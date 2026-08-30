from __future__ import annotations

import json
import logging
import pickle
from dataclasses import replace
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
from src.api.app import TRACE_DEV_MODE_ENV_VAR
from src.api.app import TRACE_HASH_SALT_ENV_VAR
from src.api.app import TRACE_SALT_POLICY_NOTE
from src.api.app import TRACEABILITY_JOIN_NOTE
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
    assert payload["trace_logging_enabled"] is False
    assert payload["trace_salt_mode"] == "disabled_missing_trace_hash_salt"
    assert payload["trace_salt_policy_note"] == TRACE_SALT_POLICY_NOTE
    assert payload["traceability_join_note"] == TRACEABILITY_JOIN_NOTE
    assert payload["retraining_cycle_id"].startswith("phase7-reference:")

    report = json.loads(settings.deployment_report_path.read_text(encoding="utf-8"))
    assert report["selected_model_source"] == "phase7_reference_model"
    assert report["pool_scope_note"] == POOL_SCOPE_NOTE
    assert report["trace_logging_enabled"] is False


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
    assert response.json()["decision_id"]
    assert response.json()["retraining_cycle_id"].startswith("phase7-reference:")
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


def test_metrics_endpoint_exposes_safe_prometheus_metrics(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.42))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.setenv(TRACE_HASH_SALT_ENV_VAR, "test-secret-salt")

    with TestClient(create_app(settings)) as client:
        score_response = client.post("/score/contract", json=_contract_payload(IDpol=771234))
        metrics_response = client.get("/metrics")

    assert score_response.status_code == 200
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    assert "adaptive_p2p_api_requests_total" in metrics_text
    assert "adaptive_p2p_scoring_requests_total" in metrics_text
    assert "adaptive_p2p_model_loaded" in metrics_text
    assert "adaptive_p2p_trace_logging_enabled" in metrics_text
    assert 'path="/score/contract"' in metrics_text
    assert "771234" not in metrics_text
    assert "BonusMalus" not in metrics_text
    assert score_response.json()["decision_id"] not in metrics_text


def test_trace_logging_with_env_salt_writes_safe_stable_fingerprints(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.25))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.setenv(TRACE_HASH_SALT_ENV_VAR, "stable-test-salt")
    monkeypatch.delenv(TRACE_DEV_MODE_ENV_VAR, raising=False)
    payload = _contract_payload(IDpol=771234)

    with TestClient(create_app(settings)) as client:
        first = client.post("/score/contract", json=payload)
        second = client.post("/score/contract", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    records = _read_jsonl(settings.decision_log_path)
    assert len(records) == 2
    assert records[0]["decision_id"] == first.json()["decision_id"]
    assert records[0]["retraining_cycle_id"] == first.json()["retraining_cycle_id"]
    assert records[0]["input_fingerprint"] == records[1]["input_fingerprint"]
    assert records[0]["trace_salt_mode"] == "env_var"
    assert records[0]["dvc_output_hash"] == "hash.dir"
    record_text = json.dumps(records)
    assert "771234" not in record_text
    assert "BonusMalus" not in record_text
    assert "VehBrand" not in record_text
    assert "stable-test-salt" not in record_text


def test_pool_trace_log_stores_aggregate_summary_not_member_payloads(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, BonusMalusRuleModel())
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.setenv(TRACE_HASH_SALT_ENV_VAR, "pool-salt")
    members = [
        _contract_payload(IDpol=10001, BonusMalus=60.0, Exposure=1.0),
        _contract_payload(IDpol=10002, BonusMalus=120.0, Exposure=3.0),
    ]

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/pool", json={"pool_id": "0", "members": members})

    assert response.status_code == 200
    records = _read_jsonl(settings.decision_log_path)
    assert len(records) == 1
    record = records[0]
    assert record["endpoint"] == "/score/pool"
    assert record["prediction_summary"]["contract_count"] == 2
    assert record["prediction_summary"]["pool_risk_score"] == response.json()["pool_risk_score"]
    record_text = json.dumps(record)
    assert "members" not in record_text
    assert "member_scores" not in record_text
    assert "10001" not in record_text
    assert "BonusMalus" not in record_text


def test_trace_log_write_failure_increments_metric_without_breaking_score(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.25))
    settings = replace(settings, decision_log_path=tmp_path)
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.setenv(TRACE_HASH_SALT_ENV_VAR, "stable-test-salt")

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload())
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert "adaptive_p2p_trace_log_failures_total 1.0" in metrics.text


def test_trace_dev_mode_uses_ephemeral_random_salt(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.25))
    payload = _contract_payload()
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.delenv(TRACE_HASH_SALT_ENV_VAR, raising=False)
    monkeypatch.setenv(TRACE_DEV_MODE_ENV_VAR, "true")

    with TestClient(create_app(settings)) as first_client:
        first_response = first_client.post("/score/contract", json=payload)
    first_fingerprint = _read_jsonl(settings.decision_log_path)[0]["input_fingerprint"]
    settings.decision_log_path.unlink()
    with TestClient(create_app(settings)) as second_client:
        second_response = second_client.post("/score/contract", json=payload)
        version = second_client.get("/model/version").json()
    second_fingerprint = _read_jsonl(settings.decision_log_path)[0]["input_fingerprint"]

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_fingerprint != second_fingerprint
    assert version["trace_logging_enabled"] is True
    assert version["trace_salt_mode"] == "ephemeral_dev_random"


def test_trace_logging_disabled_without_salt_or_dev_mode(
    tmp_path: Path, monkeypatch: object
) -> None:
    settings = _write_phase7_fixture(tmp_path, ConstantProbabilityModel(0.25))
    monkeypatch.delenv(MODEL_ARTIFACT_ENV_VAR, raising=False)
    monkeypatch.delenv(TRACE_HASH_SALT_ENV_VAR, raising=False)
    monkeypatch.delenv(TRACE_DEV_MODE_ENV_VAR, raising=False)

    with TestClient(create_app(settings)) as client:
        response = client.post("/score/contract", json=_contract_payload())
        version = client.get("/model/version")
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["decision_id"]
    assert not settings.decision_log_path.exists()
    assert version.json()["trace_logging_enabled"] is False
    assert version.json()["trace_salt_mode"] == "disabled_missing_trace_hash_salt"
    assert "adaptive_p2p_trace_logging_enabled 0.0" in metrics.text


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
    assert payload["retraining_cycle_id"].startswith("env-override:")
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
    assert response.json()["retraining_cycle_id"].startswith("phase9:")
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
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "src.api.app:app" in dockerfile
    assert 'ENV PATH="/app/.venv/bin:$PATH"' in dockerfile
    assert 'CMD ["uvicorn"' in dockerfile
    assert "uv run" not in dockerfile
    assert "uv run" not in compose
    assert "./artifacts/phase7:/app/artifacts/phase7:ro" in compose
    assert "./artifacts/phase9:/app/artifacts/phase9:ro" in compose
    assert "./artifacts/phase10:/app/artifacts/phase10" in compose
    assert "./artifacts/phase11:/app/artifacts/phase11" in compose
    assert 'TRACE_DEV_MODE: "true"' in compose
    assert "prometheus:" in compose
    assert "grafana:" in compose
    assert "COPY artifacts" not in dockerfile
    assert "COPY data" not in dockerfile
    assert "data/raw" in dockerignore
    assert "data/processed" in dockerignore
    assert "artifacts" in dockerignore
    assert "uv run pytest" in ci_workflow
    assert "docker build" in ci_workflow


def test_monitoring_config_scrapes_api_and_contains_required_alerts() -> None:
    prometheus_config = Path("monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")
    alert_rules = Path("monitoring/prometheus/alert_rules.yml").read_text(encoding="utf-8")
    datasource = Path("monitoring/grafana/provisioning/datasources/prometheus.yml").read_text(
        encoding="utf-8"
    )
    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/adaptive-p2p-risk.json").read_text(encoding="utf-8")
    )

    assert "api:8000" in prometheus_config
    assert "metrics_path: /metrics" in prometheus_config
    assert "AdaptiveP2PModelUnavailable" in alert_rules
    assert "AdaptiveP2PDataDriftDetected" in alert_rules
    assert "AdaptiveP2PConceptDriftDetected" in alert_rules
    assert "AdaptiveP2PRetrainingValidationFailed" in alert_rules
    assert "AdaptiveP2PTraceLoggingDisabled" in alert_rules
    assert "AdaptiveP2PTraceLoggingFailures" in alert_rules
    assert "uid: Prometheus" in datasource
    assert dashboard["title"] == "Adaptive P2P Risk Monitoring"


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
    settings.phase8_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.phase8_report_path.write_text(
        json.dumps(
            {
                "acceptance_passed": True,
                "data_drift_evaluation": {"first_detected_batch": 12},
                "concept_drift_evaluation": {"first_detected_batch": 15},
            }
        ),
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
        phase8_report_path=tmp_path / "phase8" / "continual_learning_report.json",
        phase9_validation_report_path=tmp_path / "phase9" / "candidate_validation_report.json",
        phase9_pipeline_report_path=tmp_path / "phase9" / "mlops_pipeline_report.json",
        dataset_version_report_path=tmp_path / "processed" / "dataset_version_report.json",
        deployment_report_path=tmp_path / "phase10" / DEPLOYMENT_REPORT_FILENAME,
        decision_log_path=tmp_path / "phase11" / "risk_decisions.jsonl",
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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
