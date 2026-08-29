from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.continual.drift import CONTINUAL_REPORT_FILENAME
from src.data.versioning import DEFAULT_DATASET_VERSION_REPORT_FILENAME
from src.mlops.pipeline import CALIBRATION_THRESHOLD_RATIONALE
from src.mlops.pipeline import FAIR_EVALUATION_NOTE
from src.mlops.pipeline import MLOPS_REPORT_FILENAME
from src.mlops.pipeline import REGISTERED_MODEL_NAME
from src.mlops.pipeline import VALIDATION_REPORT_FILENAME
from src.mlops.pipeline import _phase7_holdout
from src.mlops.pipeline import run_mlops_pipeline
from src.models.risk import build_risk_models
from src.models.risk import write_risk_modeling_artifacts


def test_mlflow_run_and_registry_version_are_created_for_valid_reference_candidate(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False)
    tracking_uri = _tracking_uri(tmp_path)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=tracking_uri,
    )

    pipeline_report = _read_json(result_paths.report_path)
    validation_report = _read_json(result_paths.validation_report_path)
    assert pipeline_report["validation_passed"] is True
    assert pipeline_report["model_registered"] is True
    assert pipeline_report["registered_model_name"] == REGISTERED_MODEL_NAME
    assert validation_report["dvc_hash_consistent"] is True
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
    run = client.get_run(pipeline_report["mlflow_run_id"])
    assert run.data.tags["dvc_output_hash"] == "hash.dir"
    assert "candidate_auc" in run.data.metrics


def test_drift_triggered_retraining_refits_candidate_and_uses_phase7_holdout(
    tmp_path: Path,
) -> None:
    paths = _write_fixture_files(tmp_path, with_events=True)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    validation_report = _read_json(result_paths.validation_report_path)
    assert validation_report["candidate_source"] == "phase9_retrained_candidate"
    assert result_paths.candidate_model_path.exists()
    assert result_paths.candidate_predictions_path is not None
    assert result_paths.candidate_predictions_path.exists()
    assert validation_report["evaluated_on_phase7_holdout"] is True
    assert validation_report["fair_evaluation_note"] == FAIR_EVALUATION_NOTE
    assert validation_report["retraining_rows_removed_for_holdout"] > 0
    assert validation_report["phase7_holdout_test_rows"] == paths.phase7_holdout_rows


def test_validation_fails_on_dvc_hash_mismatch_and_skips_registry(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False, phase8_hash="other-hash.dir")

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    validation_report = _read_json(result_paths.validation_report_path)
    pipeline_report = _read_json(result_paths.report_path)
    assert validation_report["validation_passed"] is False
    assert "dvc_hash_consistent" in validation_report["failure_reasons"]
    assert pipeline_report["model_registered"] is False
    client = MlflowClient(tracking_uri=_tracking_uri(tmp_path))
    assert client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'") == []


def test_validation_fails_when_candidate_does_not_beat_baseline(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False, baseline_auc=1.0, baseline_gini=1.0)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    validation_report = _read_json(result_paths.validation_report_path)
    assert validation_report["validation_passed"] is False
    assert "auc_beats_baseline" in validation_report["failure_reasons"]
    assert "gini_beats_baseline" in validation_report["failure_reasons"]


def test_validation_fails_when_calibration_delta_exceeds_threshold(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False, corrupted_model=True)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    validation_report = _read_json(result_paths.validation_report_path)
    assert validation_report["validation_passed"] is False
    assert validation_report["probability_rate_delta"] > 0.005
    assert "calibration_delta_passed" in validation_report["failure_reasons"]
    assert validation_report["calibration_threshold_rationale"] == CALIBRATION_THRESHOLD_RATIONALE


def test_no_event_path_validates_existing_phase7_reference_model(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
        force_retraining_check=True,
    )

    pipeline_report = _read_json(result_paths.report_path)
    validation_report = _read_json(result_paths.validation_report_path)
    assert validation_report["candidate_source"] == "phase7_reference_model"
    assert pipeline_report["retraining_requested"] is True
    assert pipeline_report["selected_retraining_trigger_batch"] is None


def test_cli_style_pipeline_writes_expected_reports(tmp_path: Path) -> None:
    paths = _write_fixture_files(tmp_path, with_events=False)

    result_paths = run_mlops_pipeline(
        **paths.pipeline_kwargs,
        output_dir=tmp_path / "phase9",
        mlflow_tracking_uri=_tracking_uri(tmp_path),
    )

    assert result_paths.report_path.name == MLOPS_REPORT_FILENAME
    assert result_paths.validation_report_path.name == VALIDATION_REPORT_FILENAME
    assert _read_json(result_paths.report_path)["kafka_scope_note"]


class CorruptedHighProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(features), 0.95)
        return np.column_stack((1.0 - probability, probability))


class CalibratedRuleModel:
    classes_ = np.array([0, 1])

    def __init__(self, bonus_malus_threshold: float) -> None:
        self.bonus_malus_threshold = bonus_malus_threshold

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        high_risk = features["BonusMalus"].to_numpy() >= self.bonus_malus_threshold
        probabilities = np.where(high_risk, 0.7166666667, 0.05)
        return np.column_stack((1.0 - probabilities, probabilities))


class FixturePaths:
    def __init__(self, pipeline_kwargs: dict[str, Path], phase7_holdout_rows: int) -> None:
        self.pipeline_kwargs = pipeline_kwargs
        self.phase7_holdout_rows = phase7_holdout_rows


def _write_fixture_files(
    tmp_path: Path,
    with_events: bool,
    phase8_hash: str = "hash.dir",
    baseline_auc: float = 0.5,
    baseline_gini: float = 0.0,
    corrupted_model: bool = False,
) -> FixturePaths:
    contract_features, contract_targets, pool_features, pool_targets, threshold = (
        _synthetic_model_data()
    )
    processed_dir = tmp_path / "processed"
    phase7_dir = tmp_path / "phase7"
    phase8_dir = tmp_path / "phase8"
    stream_dir = processed_dir / "stream"
    processed_dir.mkdir()
    phase8_dir.mkdir()
    stream_dir.mkdir()

    contract_features_path = processed_dir / "contract_features.csv"
    contract_targets_path = processed_dir / "contract_targets.csv"
    pool_features_path = processed_dir / "pool_features.csv"
    pool_targets_path = processed_dir / "pool_targets.csv"
    contract_features.to_csv(contract_features_path, index=False)
    contract_targets.to_csv(contract_targets_path, index=False)
    pool_features.to_csv(pool_features_path, index=False)
    pool_targets.to_csv(pool_targets_path, index=False)

    risk_result = build_risk_models(
        contract_features,
        contract_targets,
        pool_features,
        pool_targets,
        dataset_version_report={
            "dvc_output_hash": "hash.dir",
            "dvc_tracked_path": "data/processed",
            "git_commit_sha": "git123",
        },
    )
    risk_paths = write_risk_modeling_artifacts(risk_result, phase7_dir)
    phase7_report = _read_json(risk_paths.report_path)
    phase7_report["baseline_auc"] = baseline_auc
    phase7_report["baseline_normalized_gini"] = baseline_gini
    phase7_report["acceptance_passed"] = True
    risk_paths.report_path.write_text(json.dumps(phase7_report), encoding="utf-8")
    if corrupted_model:
        with risk_paths.model_artifact_path.open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
        payload["classifier_model"] = CorruptedHighProbabilityModel()
        with risk_paths.model_artifact_path.open("wb") as artifact_file:
            pickle.dump(payload, artifact_file)
    elif not with_events:
        with risk_paths.model_artifact_path.open("rb") as artifact_file:
            payload = pickle.load(artifact_file)
        payload["classifier_model"] = CalibratedRuleModel(threshold)
        with risk_paths.model_artifact_path.open("wb") as artifact_file:
            pickle.dump(payload, artifact_file)

    _write_stream_batches(stream_dir, contract_features, contract_targets)
    events = (
        pd.DataFrame(
            [
                {
                    "trigger_batch": 12,
                    "triggering_detector": "data",
                    "training_window_batches": "8,9,10,11,12",
                },
                {
                    "trigger_batch": 19,
                    "triggering_detector": "data_and_concept",
                    "training_window_batches": "15,16,17,18,19",
                },
            ]
        )
        if with_events
        else pd.DataFrame(columns=["trigger_batch", "triggering_detector", "training_window_batches"])
    )
    retraining_events_path = phase8_dir / "retraining_events.csv"
    events.to_csv(retraining_events_path, index=False)
    drift_metrics_path = phase8_dir / "drift_metrics.csv"
    pd.DataFrame({"batch_id": [12], "data_drift_detected": [True]}).to_csv(
        drift_metrics_path, index=False
    )
    phase8_report_path = phase8_dir / CONTINUAL_REPORT_FILENAME
    phase8_report_path.write_text(
        json.dumps(
            {
                "acceptance_passed": True,
                "dvc_output_hash": phase8_hash,
                "data_drift_evaluation": {"detection_latency_batches": 0},
                "concept_drift_evaluation": {"detection_latency_batches": 0},
                "retraining_event_count": len(events),
                "provisional_threshold_note": "provisional",
                "overlap_batch_start": 15,
                "overlap_batch_end": 19,
            }
        ),
        encoding="utf-8",
    )
    dataset_report_path = processed_dir / DEFAULT_DATASET_VERSION_REPORT_FILENAME
    dataset_report_path.write_text(
        json.dumps(
            {
                "dvc_output_hash": "hash.dir",
                "dvc_tracked_path": "data/processed",
                "git_commit_sha": "git123",
            }
        ),
        encoding="utf-8",
    )
    dvc_metadata_path = tmp_path / "processed.dvc"
    dvc_metadata_path.write_text("outs:\n- md5: hash.dir\n  path: processed\n", encoding="utf-8")

    holdout = _phase7_holdout(contract_features, contract_targets)
    return FixturePaths(
        pipeline_kwargs={
            "phase7_model_path": risk_paths.model_artifact_path,
            "phase7_report_path": risk_paths.report_path,
            "phase8_report_path": phase8_report_path,
            "retraining_events_path": retraining_events_path,
            "drift_metrics_path": drift_metrics_path,
            "dataset_version_report_path": dataset_report_path,
            "dvc_metadata_path": dvc_metadata_path,
            "stream_dir": stream_dir,
            "contract_features_path": contract_features_path,
            "contract_targets_path": contract_targets_path,
            "pool_features_path": pool_features_path,
            "pool_targets_path": pool_targets_path,
        },
        phase7_holdout_rows=len(holdout.y_test),
    )


def _synthetic_model_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    rng = np.random.default_rng(42)
    n_rows = 240
    ids = np.arange(1, n_rows + 1)
    pool_ids = (ids % 4).astype(str)
    bonus_malus = np.linspace(45.0, 165.0, n_rows)
    veh_power = rng.integers(4, 12, size=n_rows)
    veh_age = rng.integers(0, 16, size=n_rows)
    driv_age = rng.integers(19, 75, size=n_rows)
    exposure = rng.uniform(0.3, 1.0, size=n_rows)
    density = rng.integers(50, 4000, size=n_rows)
    risk_signal = bonus_malus
    positive = np.zeros(n_rows, dtype=int)
    positive[np.argsort(risk_signal)[-36:]] = 1
    threshold = float(np.min(bonus_malus[positive == 1]))
    contract_features = pd.DataFrame(
        {
            "IDpol": ids,
            "pool_id": pool_ids,
            "Exposure": exposure,
            "VehPower": veh_power,
            "VehAge": veh_age,
            "DrivAge": driv_age,
            "BonusMalus": bonus_malus,
            "Density": density,
            "Density_log1p": np.log1p(density),
            "VehBrand": np.where(veh_power > 8, "B2", "B1"),
            "VehGas": np.where(ids % 2 == 0, "Diesel", "Regular"),
            "Area": np.where(density > 1200, "E", "C"),
            "Region": np.where(ids % 3 == 0, "R1", "R2"),
            "VehAge_band": np.where(veh_age <= 5, "0-5", "6+"),
            "DrivAge_band": np.where(driv_age <= 35, "0-35", "36+"),
            "BonusMalus_band": np.where(bonus_malus <= 100, "0-100", "101+"),
            "Density_band": np.where(density <= 1500, "0-1500", "1501+"),
        }
    )
    contract_targets = pd.DataFrame(
        {
            "IDpol": ids,
            "target_claim_count": positive,
            "target_has_claim": positive,
            "target_claim_frequency": positive / exposure,
            "target_total_claim_amount": positive * 1000.0,
            "target_avg_claim_amount": positive * 1000.0,
        }
    )
    pool_targets = (
        pd.DataFrame({"pool_id": pool_ids, "target": positive, "Exposure": exposure})
        .groupby("pool_id", as_index=False)
        .agg(pool_claim_count=("target", "sum"), pool_total_exposure=("Exposure", "sum"))
    )
    pool_targets["pool_has_claim"] = (pool_targets["pool_claim_count"] > 0).astype(int)
    pool_targets["pool_claim_rate"] = (
        pool_targets["pool_claim_count"] / pool_targets["pool_total_exposure"]
    )
    pool_targets["pool_total_claim_amount"] = pool_targets["pool_claim_count"] * 1000.0
    pool_targets["pool_avg_claim_amount"] = (
        pool_targets["pool_total_claim_amount"] / pool_targets["pool_claim_count"]
    )
    pool_features = pd.DataFrame(
        {
            "pool_id": sorted(set(pool_ids)),
            "pool_size": [int((pool_ids == pool_id).sum()) for pool_id in sorted(set(pool_ids))],
        }
    )
    return contract_features, contract_targets, pool_features, pool_targets, threshold


def _write_stream_batches(
    stream_dir: Path,
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
) -> None:
    stream = contract_features.merge(contract_targets, on="IDpol", validate="one_to_one")
    for batch_id, indices in enumerate(np.array_split(np.arange(len(stream)), 20)):
        batch = stream.iloc[indices].copy()
        batch["batch_id"] = batch_id
        batch["data_drift_injected"] = batch_id >= 12
        batch["concept_drift_injected"] = batch_id >= 15
        batch.to_csv(stream_dir / f"batch_{batch_id:03d}.csv", index=False)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}"
