from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.continual.drift import CONCEPT_THRESHOLD_NOTE
from src.continual.drift import CONTINUAL_REPORT_FILENAME
from src.continual.drift import DATA_DRIFT_DETECTOR_NOTE
from src.continual.drift import FROZEN_PREPROCESSING_NOTE
from src.continual.drift import PROVISIONAL_THRESHOLD_NOTE
from src.continual.drift import TWO_MODEL_RELATIONSHIP_NOTE
from src.continual.drift import monitor_drift
from src.continual.drift import run_drift_monitoring_pipeline
from src.models.risk import MODEL_INPUT_COLUMNS
from src.models.risk import _build_preprocessor


class ConstantProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability: float = 0.05) -> None:
        self.probability = probability

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(features), self.probability, dtype="float64")
        return np.column_stack((1.0 - probabilities, probabilities))


def test_psi_detects_abrupt_data_drift_within_provisional_latency() -> None:
    result = monitor_drift(*_monitoring_inputs())

    evaluation = result.report.data_drift_evaluation
    assert evaluation.first_detected_batch is not None
    assert evaluation.first_detected_batch <= 14
    assert evaluation.detected_within_provisional_threshold is True
    assert result.drift_metrics.loc[12, "data_drift_detected"]


def test_concept_residual_detector_detects_shift_within_provisional_latency() -> None:
    result = monitor_drift(*_monitoring_inputs())

    evaluation = result.report.concept_drift_evaluation
    assert evaluation.first_detected_batch is not None
    assert evaluation.first_detected_batch <= 17
    assert evaluation.detected_within_provisional_threshold is True
    assert result.drift_metrics.loc[15, "concept_drift_detected"]


def test_data_and_concept_drift_are_reported_separately_during_overlap() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert result.report.overlap_batch_start == 15
    assert result.report.overlap_batch_end == 19
    assert 15 in result.report.data_drift_evaluation.overlap_detected_batches
    assert 15 in result.report.concept_drift_evaluation.overlap_detected_batches


def test_monitored_features_exclude_targets_and_match_phase7_model_inputs() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert result.report.monitored_feature_columns == MODEL_INPUT_COLUMNS
    assert "target_has_claim" not in result.report.monitored_feature_columns
    assert "target_claim_frequency" not in result.report.monitored_feature_columns


def test_reference_window_threshold_notes_and_patterns_are_logged() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert result.report.reference_batch_start == 0
    assert result.report.reference_batch_end == 9
    assert result.report.provisional_latency_batches == 2
    assert result.report.provisional_threshold_note == PROVISIONAL_THRESHOLD_NOTE
    assert result.report.concept_threshold_note == CONCEPT_THRESHOLD_NOTE
    assert result.report.data_drift_pattern == "abrupt_step_function"
    assert result.report.concept_drift_pattern == "abrupt_step_function"
    assert result.report.data_drift_detector_note == DATA_DRIFT_DETECTOR_NOTE


def test_phase7_preprocessing_is_frozen_and_online_model_does_not_replace_reference() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert result.report.frozen_preprocessing_used is True
    assert result.report.frozen_preprocessing_note == FROZEN_PREPROCESSING_NOTE
    assert result.report.online_model_name == "SGDClassifier"
    assert result.report.online_partial_fit_updates == 20
    assert result.report.two_model_relationship_note == TWO_MODEL_RELATIONSHIP_NOTE
    assert not result.retraining_events["candidate_replaces_reference_model"].any()


def test_retraining_events_use_sliding_window_and_reference_replay() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert len(result.retraining_events) > 0
    first_event = result.retraining_events.iloc[0]
    assert first_event["event_type"] == "sliding_window_retraining_candidate"
    assert first_event["window_end_batch"] == first_event["trigger_batch"]
    assert first_event["window_rows"] == 500
    assert first_event["replay_rows"] == 100
    assert first_event["training_rows"] == 600
    assert result.report.sliding_window_batches == 5
    assert result.report.replay_fraction == 0.10


def test_eligible_row_fractions_are_logged_from_injection_windows() -> None:
    result = monitor_drift(*_monitoring_inputs())
    metrics = result.drift_metrics.set_index("batch_id")

    assert metrics.loc[12, "data_drift_eligible_fraction"] == 1.0
    assert metrics.loc[15, "concept_drift_eligible_fraction"] == 0.30
    assert "eligible" in result.report.eligible_fraction_note


def test_data_drift_metrics_include_density_segment_psi_signal() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert "density_segment_psi" in result.drift_metrics.columns
    assert "data_drift_signal_psi" in result.drift_metrics.columns
    assert "data_drift_signal_source" in result.drift_metrics.columns


def test_report_carries_phase7_dataset_traceability() -> None:
    result = monitor_drift(*_monitoring_inputs())

    assert result.report.dvc_output_hash == "hash.dir"
    assert result.report.dvc_tracked_path == "data/processed"
    assert result.report.dataset_git_commit_sha == "abc123"
    assert result.report.phase7_model_auc == 0.69
    assert result.report.phase7_model_normalized_gini == 0.38


def test_run_drift_monitoring_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
    batches, model_artifact, ground_truth, phase7_report = _monitoring_inputs()
    stream_dir = tmp_path / "stream"
    stream_dir.mkdir()
    for batch_id, batch in enumerate(batches):
        batch.to_csv(stream_dir / f"batch_{batch_id:03d}.csv", index=False)
    ground_truth_path = stream_dir / "drift_ground_truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")
    model_artifact_path = tmp_path / "risk_model.pkl"
    with model_artifact_path.open("wb") as artifact_file:
        pickle.dump(model_artifact, artifact_file)
    phase7_report_path = tmp_path / "risk_modeling_report.json"
    phase7_report_path.write_text(json.dumps(phase7_report), encoding="utf-8")

    paths = run_drift_monitoring_pipeline(
        stream_dir=stream_dir,
        drift_ground_truth_path=ground_truth_path,
        model_artifact_path=model_artifact_path,
        phase7_report_path=phase7_report_path,
        output_dir=tmp_path / "phase8",
    )

    assert paths.drift_metrics_path.exists()
    assert paths.retraining_events_path.exists()
    assert paths.report_path.name == CONTINUAL_REPORT_FILENAME
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    assert report["acceptance_passed"] is True


def _monitoring_inputs() -> tuple[tuple[pd.DataFrame, ...], dict[str, object], dict[str, object], dict[str, object]]:
    batches = _stream_batches()
    reference = pd.concat(batches[:10], ignore_index=True)
    preprocessor = _build_preprocessor()
    preprocessor.fit(reference.loc[:, MODEL_INPUT_COLUMNS])
    model_artifact = {
        "classifier_model": ConstantProbabilityModel(0.05),
        "feature_preprocessor": preprocessor,
    }
    phase7_report = {
        "dvc_output_hash": "hash.dir",
        "dvc_tracked_path": "data/processed",
        "dataset_git_commit_sha": "abc123",
        "model_auc": 0.69,
        "model_normalized_gini": 0.38,
    }
    return batches, model_artifact, _ground_truth(), phase7_report


def _stream_batches() -> tuple[pd.DataFrame, ...]:
    batches = []
    for batch_id in range(20):
        n_rows = 100
        ids = np.arange(batch_id * n_rows, (batch_id + 1) * n_rows)
        data_drift = batch_id >= 12
        concept_drift = batch_id >= 15
        density = np.full(n_rows, 800.0)
        density_band = np.full(n_rows, "501-1500", dtype=object)
        if data_drift:
            density = np.full(n_rows, 6500.0)
            density_band = np.full(n_rows, "5001+", dtype=object)

        target = np.zeros(n_rows, dtype=int)
        positive_count = 30 if concept_drift else 5
        target[:positive_count] = 1
        bonus_malus = np.full(n_rows, 80)
        if concept_drift:
            bonus_malus[:30] = 120

        batches.append(
            pd.DataFrame(
                {
                    "IDpol": ids,
                    "batch_id": batch_id,
                    "event_index": ids,
                    "simulated_time_step": ids,
                    "data_drift_injected": data_drift,
                    "concept_drift_injected": concept_drift,
                    "pool_id": np.where(ids % 2 == 0, "0", "1"),
                    "Exposure": np.full(n_rows, 1.0),
                    "VehPower": np.where(ids % 2 == 0, 5, 7),
                    "VehAge": np.where(ids % 3 == 0, 2, 8),
                    "DrivAge": np.where(ids % 4 == 0, 30, 50),
                    "BonusMalus": bonus_malus,
                    "Density": density,
                    "Density_log1p": np.log1p(density),
                    "VehBrand": np.where(ids % 2 == 0, "B1", "B2"),
                    "VehGas": np.where(ids % 2 == 0, "Regular", "Diesel"),
                    "Area": np.where(ids % 2 == 0, "C", "E"),
                    "Region": np.where(ids % 2 == 0, "R1", "R2"),
                    "VehAge_band": np.where(ids % 3 == 0, "0-2", "6-10"),
                    "DrivAge_band": np.where(ids % 4 == 0, "26-35", "36-50"),
                    "BonusMalus_band": np.where(bonus_malus >= 100, "101-150", "76-100"),
                    "Density_band": density_band,
                    "target_claim_count": target,
                    "target_has_claim": target,
                    "target_claim_frequency": target.astype(float),
                    "target_total_claim_amount": target.astype(float) * 1000.0,
                    "target_avg_claim_amount": target.astype(float) * 1000.0,
                }
            )
        )
    return tuple(batches)


def _ground_truth() -> dict[str, object]:
    return {
        "n_batches": 20,
        "data_drift": {"start_batch": 12, "pattern": "abrupt_step_function"},
        "concept_drift": {"start_batch": 15, "pattern": "abrupt_step_function"},
        "overlap": {"intentional": True, "batch_start": 15, "batch_end": 19},
    }
