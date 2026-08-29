from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.api.app as api_app
from src.api.app import DEPLOYMENT_REPORT_FILENAME
from src.api.app import FEATURE_PARITY_NOTE
from src.api.app import LOGGING_POLICY_NOTE
from src.api.app import MODEL_ARTIFACT_ENV_VAR
from src.api.app import POOL_SCOPE_NOTE
from src.api.app import DeploymentSettings
from src.api.app import create_app
from src.data import features as feature_engineering


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(features), self.probability)
        return np.column_stack((1.0 - probabilities, probabilities))


class BonusMalusRuleModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.where(features["BonusMalus"].to_numpy(dtype=float) >= 100.0, 0.5, 0.1)
        return np.column_stack((1.0 - probabilities, probabilities))


def test_health_and_model_version_report_valid_loaded_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.25))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)

    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        version = client.get("/model/version")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert health.json()["selected_model_source"] == "phase7_reference_model"
    assert version.status_code == 200
    payload = version.json()
    assert payload["feature_parity_note"] == FEATURE_PARITY_NOTE
    assert payload["logging_policy_note"] == LOGGING_POLICY_NOTE
    assert payload["pool_scope_note"] == POOL_SCOPE_NOTE
    assert payload["dvc_output_hash"] == "hash.dir"

    report = json.loads(settings.deployment_report_path.read_text(encoding="utf-8"))
    assert report["selected_model_source"] == "phase7_reference_model"
    assert report["pool_scope_note"] == POOL_SCOPE_NOTE


def test_score_contract_returns_probability_and_uses_phase4_serving_helper(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.31))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    original_builder = api_app.feature_engineering.build_serving_contract_features
    calls = {"count": 0}

    def spy_builder(contracts: pd.DataFrame) -> pd.DataFrame:
        calls["count"] += 1
        return original_builder(contracts)

    monkeypatch.setattr(
        api_app.feature_engineering,
        "build_serving_contract_features",
        spy_builder,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload(IDpol=99123))

    assert response.status_code == 200
    assert response.json()["model_source"] == "phase7_reference_model"
    assert response.json()["predicted_claim_probability"] == 0.31
    assert calls["count"] == 1


def test_phase4_serving_helper_matches_fixed_bands_and_density_transform() -> None:
    serving_features = feature_engineering.build_serving_contract_features(
        pd.DataFrame(
            [
                _contract_payload(
                    VehAge=2.0,
                    DrivAge=35.0,
                    BonusMalus=100.0,
                    Density=1500.0,
                )
            ]
        )
    )

    row = serving_features.iloc[0]
    assert row["Density_log1p"] == np.log1p(1500.0)
    assert row["VehAge_band"] == "0-2"
    assert row["DrivAge_band"] == "26-35"
    assert row["BonusMalus_band"] == "76-100"
    assert row["Density_band"] == "501-1500"


def test_score_pool_uses_exposure_weighted_score(tmp_path: Path, monkeypatch: object) -> None:
    settings = _write_phase7_fixture(tmp_path, BonusMalusRuleModel())
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    members = [
        _contract_payload(BonusMalus=60.0, Exposure=1.0),
        _contract_payload(BonusMalus=120.0, Exposure=3.0),
    ]

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/pool", json={"pool_id": "0", "members": members})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pool_risk_score"] == 0.4
    assert payload["pool_contract_count"] == 2
    assert payload["pool_scored_exposure"] == 4.0
    assert payload["pool_score_weighting_method"] == "exposure_weighted"
    assert [member["predicted_claim_probability"] for member in payload["member_scores"]] == [
        0.1,
        0.5,
    ]


def test_score_pool_uses_unweighted_fallback_for_zero_exposure(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, BonusMalusRuleModel())
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    members = [
        _contract_payload(BonusMalus=60.0, Exposure=0.0),
        _contract_payload(BonusMalus=120.0, Exposure=0.0),
    ]

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/pool", json={"pool_id": "0", "members": members})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pool_risk_score"] == 0.3
    assert payload["pool_score_weighting_method"] == "unweighted_zero_exposure_fallback"


def test_score_pool_rejects_member_rows_from_other_pool(tmp_path: Path, monkeypatch: object) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.2))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    members = [_contract_payload(pool_id="other")]

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/pool", json={"pool_id": "0", "members": members})

    assert response.status_code == 400
    assert "same pool_id" in response.json()["detail"]


def test_validation_errors_do_not_echo_payload_values(tmp_path: Path, monkeypatch: object) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.2))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    payload = _contract_payload(IDpol=771234, Exposure=-1.0)

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=payload)

    response_text = json.dumps(response.json())
    assert response.status_code == 422
    assert "771234" not in response_text
    assert "-1.0" not in response_text


def test_request_logging_excludes_payload_and_predictions(
    tmp_path: Path, monkeypatch: object, caplog: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.42))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    caplog.set_level(logging.INFO, logger=api_app.LOGGER.name)

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload(IDpol=771234))

    assert response.status_code == 200
    messages = "\n".join(
        record.getMessage() for record in caplog.records if record.name == api_app.LOGGER.name
    )
    assert "method=POST" in messages
    assert "path=/score/contract" in messages
    assert "model_source=phase7_reference_model" in messages
    assert "771234" not in messages
    assert "BonusMalus" not in messages
    assert "predicted_claim_probability" not in messages
    assert "0.42" not in messages


def test_model_artifact_path_env_override_wins(tmp_path: Path, monkeypatch: object) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.11))
    env_model_path = tmp_path / "override" / "risk_model.pkl"
    _write_model_artifact(env_model_path, ConstantProbabilityModel(0.77))
    monkeypatch.setenv(MODEL_ARTIFACT_ENV_VAR, str(env_model_path))

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_source"] == "env_override"
    assert payload["model_artifact_path"] == str(env_model_path)
    assert payload["predicted_claim_probability"] == 0.77


def test_validated_phase9_candidate_is_selected_without_env_override(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.11))
    _write_model_artifact(settings.phase9_model_path, ConstantProbabilityModel(0.66))
    settings.phase9_validation_report_path.write_text(
        json.dumps({"validation_passed": True, "candidate_source": "phase9_retrained_candidate"}),
        encoding="utf-8",
    )
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload())

    assert response.status_code == 200
    assert response.json()["model_source"] == "phase9_retrained_candidate"
    assert response.json()["predicted_claim_probability"] == 0.66


def test_phase7_fallback_is_selected_when_phase9_validation_failed(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.22))
    _write_model_artifact(settings.phase9_model_path, ConstantProbabilityModel(0.99))
    settings.phase9_validation_report_path.write_text(
        json.dumps({"validation_passed": False, "candidate_source": "phase9_retrained_candidate"}),
        encoding="utf-8",
    )
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload())

    assert response.status_code == 200
    assert response.json()["model_source"] == "phase7_reference_model"
    assert response.json()["predicted_claim_probability"] == 0.22


def test_invalid_env_override_fails_without_silent_fallback(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.22))
    monkeypatch.setenv(MODEL_ARTIFACT_ENV_VAR, str(tmp_path / "missing.pkl"))

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.post("/score/contract", json=_contract_payload())

    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert response.status_code == 503
    assert "model_unavailable" in response.text


def test_health_is_degraded_when_no_valid_model_exists(tmp_path: Path, monkeypatch: object) -> None:
    settings = _settings(tmp_path)
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.post("/score/contract", json=_contract_payload())

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert response.status_code == 503


def test_deployment_packaging_files_keep_artifacts_out_of_image() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "src.api.app:app" in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "COPY data" not in dockerfile
    assert "data/raw" in dockerignore
    assert "data/processed" in dockerignore
    assert "artifacts" in dockerignore
    assert "uv run pytest" in ci_workflow
    assert "docker build" in ci_workflow


def _write_phase7_fixture(tmp_path: Path, model: object) -> DeploymentSettings:
    settings = _settings(tmp_path)
    _write_model_artifact(settings.phase7_model_path, model)
    settings.phase7_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.phase7_report_path.write_text(
        json.dumps(
            {
                "acceptance_passed": True,
                "model_auc": 0.7,
                "model_normalized_gini": 0.4,
                "expected_calibration_error": 0.01,
            }
        ),
        encoding="utf-8",
    )
    settings.dataset_version_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.dataset_version_report_path.write_text(
        json.dumps(
            {
                "dvc_output_hash": "hash.dir",
                "dvc_tracked_path": "data/processed",
                "git_commit_sha": "git123",
            }
        ),
        encoding="utf-8",
    )
    settings.phase9_validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.phase9_pipeline_report_path.write_text(
        json.dumps({"model_registered": False, "registered_model_version": None}),
        encoding="utf-8",
    )
    return settings


def _write_model_artifact(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact_file:
        pickle.dump({"classifier_model": model, "report": {"fixture": True}}, artifact_file)


def _settings(tmp_path: Path) -> DeploymentSettings:
    return DeploymentSettings(
        phase7_model_path=tmp_path / "phase7" / "risk_model.pkl",
        phase7_report_path=tmp_path / "phase7" / "risk_modeling_report.json",
        phase9_model_path=tmp_path / "phase9" / "retrained_candidate" / "risk_model.pkl",
        phase9_validation_report_path=tmp_path / "phase9" / "candidate_validation_report.json",
        phase9_pipeline_report_path=tmp_path / "phase9" / "mlops_pipeline_report.json",
        dataset_version_report_path=tmp_path / "processed" / "dataset_version_report.json",
        deployment_report_path=tmp_path / "phase10" / DEPLOYMENT_REPORT_FILENAME,
    )


def _contract_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "pool_id": "0",
        "Exposure": 0.7,
        "VehPower": 6.0,
        "VehAge": 4.0,
        "DrivAge": 42.0,
        "BonusMalus": 68.0,
        "Density": 850.0,
        "VehBrand": "B1",
        "VehGas": "Regular",
        "Area": "C",
        "Region": "R1",
    }
    payload.update(overrides)
    return payload
