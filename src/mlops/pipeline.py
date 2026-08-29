"""Local MLOps orchestration for Phase 9.

This module implements actual drift-triggered retraining when Phase 8 emits
events. Retrained candidates are evaluated on the reconstructed Phase 7
held-out split, with those IDs removed from the retraining window, so the gate
does not score a candidate on rows it trained on.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from src.continual.drift import DEFAULT_PHASE8_ARTIFACT_DIR
from src.continual.drift import REPLAY_FRACTION
from src.continual.drift import REFERENCE_BATCH_END
from src.continual.drift import REFERENCE_BATCH_START
from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.features import DEFAULT_CONTRACT_FEATURES_FILENAME
from src.data.features import DEFAULT_CONTRACT_TARGETS_FILENAME
from src.data.features import DEFAULT_POOL_FEATURES_FILENAME
from src.data.features import DEFAULT_POOL_TARGETS_FILENAME
from src.data.streaming import DEFAULT_STREAM_DIR
from src.data.versioning import DEFAULT_DATASET_VERSION_REPORT_FILENAME
from src.data.versioning import DEFAULT_DVC_METADATA_PATH
from src.models.risk import DEFAULT_ARTIFACT_DIR as DEFAULT_PHASE7_ARTIFACT_DIR
from src.models.risk import DEFAULT_TEST_SIZE
from src.models.risk import ID_COLUMN
from src.models.risk import MODEL_ARTIFACT_FILENAME
from src.models.risk import MODEL_INPUT_COLUMNS
from src.models.risk import POOL_COLUMN
from src.models.risk import RANDOM_STATE
from src.models.risk import RISK_REPORT_FILENAME as PHASE7_RISK_REPORT_FILENAME
from src.models.risk import TARGET_COLUMN
from src.models.risk import _build_pool_risk_scores
from src.models.risk import _compute_metrics
from src.models.risk import _fit_probability_model
from src.models.risk import _predict_positive_probability
from src.models.risk import _prepare_contract_modeling_data

DEFAULT_PHASE9_ARTIFACT_DIR = Path("artifacts/phase9")
DEFAULT_CONTRACT_FEATURES_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_FEATURES_FILENAME
DEFAULT_CONTRACT_TARGETS_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_TARGETS_FILENAME
DEFAULT_POOL_FEATURES_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_POOL_FEATURES_FILENAME
DEFAULT_POOL_TARGETS_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_POOL_TARGETS_FILENAME
DEFAULT_DATASET_VERSION_REPORT_PATH = (
    DEFAULT_PROCESSED_DIR / DEFAULT_DATASET_VERSION_REPORT_FILENAME
)
DEFAULT_PHASE7_MODEL_PATH = DEFAULT_PHASE7_ARTIFACT_DIR / MODEL_ARTIFACT_FILENAME
DEFAULT_PHASE7_REPORT_PATH = DEFAULT_PHASE7_ARTIFACT_DIR / PHASE7_RISK_REPORT_FILENAME
DEFAULT_PHASE8_REPORT_PATH = DEFAULT_PHASE8_ARTIFACT_DIR / "continual_learning_report.json"
DEFAULT_RETRAINING_EVENTS_PATH = DEFAULT_PHASE8_ARTIFACT_DIR / "retraining_events.csv"
DEFAULT_DRIFT_METRICS_PATH = DEFAULT_PHASE8_ARTIFACT_DIR / "drift_metrics.csv"
DEFAULT_MLFLOW_TRACKING_DIR = Path("mlruns")

EXPERIMENT_NAME = "adaptive-p2p-risk"
REGISTERED_MODEL_NAME = "adaptive-p2p-risk-contract-risk"
MLOPS_REPORT_FILENAME = "mlops_pipeline_report.json"
VALIDATION_REPORT_FILENAME = "candidate_validation_report.json"
PROMOTION_METADATA_FILENAME = "promotion_metadata.json"
RETRAINED_CANDIDATE_DIRNAME = "retrained_candidate"
RETRAINED_CANDIDATE_MODEL_FILENAME = "risk_model.pkl"
RETRAINED_CANDIDATE_PREDICTIONS_FILENAME = "candidate_contract_test_predictions.csv"
RETRAINED_CANDIDATE_POOL_SCORES_FILENAME = "candidate_pool_risk_scores.csv"

CALIBRATION_RATE_DELTA_THRESHOLD = 0.005
CALIBRATION_THRESHOLD_RATIONALE = (
    "The 0.005 probability-rate delta threshold is intentionally above the "
    "observed Phase 7 baseline delta of about 0.00008 to tolerate normal "
    "split/retraining variance while still catching gross miscalibration."
)
FAIR_EVALUATION_NOTE = (
    "Retrained candidates are evaluated on the reconstructed Phase 7 held-out "
    "test split; held-out IDs are removed from retraining data before fitting."
)
PROMOTION_SCOPE_NOTE = (
    "Promotion means local MLflow model registry registration only; serving and "
    "production replacement remain Phase 10."
)
FR_ML_04_NOTE = (
    "FR-ML-04 is implemented as actual local retraining orchestration: when "
    "Phase 8 emits retraining events, the latest event is used to refit a "
    "calibrated logistic candidate and send it through the validation gate."
)
KAFKA_SCOPE_NOTE = "Kafka simulation is deferred because FR-ML-06 is Could/optional."


class MLOpsPipelineError(ValueError):
    """Raised when Phase 9 orchestration inputs or validation are invalid."""


@dataclass(frozen=True)
class CandidateValidationReport:
    candidate_source: str
    validation_passed: bool
    phase7_acceptance_passed: bool
    phase8_acceptance_passed: bool
    dvc_hash_consistent: bool
    dvc_metadata_hash_matches_report: bool
    dvc_output_hash: str | None
    phase6_dvc_hash: str | None
    phase7_dvc_hash: str | None
    phase8_dvc_hash: str | None
    dvc_metadata_hash: str | None
    dataset_git_commit_sha: str | None
    pool_score_coverage: float
    pool_score_coverage_passed: bool
    candidate_auc: float
    baseline_auc: float
    auc_beats_baseline: bool
    candidate_normalized_gini: float
    baseline_normalized_gini: float
    gini_beats_baseline: bool
    probability_rate_delta: float
    calibration_delta_threshold: float
    calibration_delta_passed: bool
    calibration_threshold_rationale: str
    evaluated_on_phase7_holdout: bool
    phase7_holdout_test_rows: int
    candidate_training_rows: int
    retraining_rows_removed_for_holdout: int
    fair_evaluation_note: str
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MLOpsPipelineReport:
    experiment_name: str
    registered_model_name: str
    mlflow_tracking_uri: str
    mlflow_run_id: str
    mlflow_artifact_uri: str
    candidate_source: str
    retraining_requested: bool
    selected_retraining_trigger_batch: int | None
    selected_retraining_detector: str | None
    validation_passed: bool
    model_registered: bool
    registered_model_version: str | None
    dvc_status_checked: bool
    dvc_status_clean: bool | None
    dvc_status_output: str | None
    fr_ml_04_note: str
    promotion_scope_note: str
    kafka_scope_note: str
    validation_report_path: str
    promotion_metadata_path: str | None


@dataclass(frozen=True)
class MLOpsPipelineResult:
    validation_report: CandidateValidationReport
    pipeline_report: MLOpsPipelineReport
    candidate_model_path: Path
    candidate_predictions_path: Path | None
    candidate_pool_scores_path: Path | None


@dataclass(frozen=True)
class MLOpsPipelinePaths:
    report_path: Path
    validation_report_path: Path
    promotion_metadata_path: Path | None
    candidate_model_path: Path
    candidate_predictions_path: Path | None
    candidate_pool_scores_path: Path | None


def run_mlops_pipeline(
    phase7_model_path: Path | str = DEFAULT_PHASE7_MODEL_PATH,
    phase7_report_path: Path | str = DEFAULT_PHASE7_REPORT_PATH,
    phase8_report_path: Path | str = DEFAULT_PHASE8_REPORT_PATH,
    retraining_events_path: Path | str = DEFAULT_RETRAINING_EVENTS_PATH,
    drift_metrics_path: Path | str = DEFAULT_DRIFT_METRICS_PATH,
    dataset_version_report_path: Path | str = DEFAULT_DATASET_VERSION_REPORT_PATH,
    dvc_metadata_path: Path | str = DEFAULT_DVC_METADATA_PATH,
    stream_dir: Path | str = DEFAULT_STREAM_DIR,
    contract_features_path: Path | str = DEFAULT_CONTRACT_FEATURES_PATH,
    contract_targets_path: Path | str = DEFAULT_CONTRACT_TARGETS_PATH,
    pool_features_path: Path | str = DEFAULT_POOL_FEATURES_PATH,
    pool_targets_path: Path | str = DEFAULT_POOL_TARGETS_PATH,
    output_dir: Path | str = DEFAULT_PHASE9_ARTIFACT_DIR,
    mlflow_tracking_uri: str | None = None,
    dvc_status_check: bool = False,
    force_retraining_check: bool = False,
) -> MLOpsPipelinePaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    phase7_report = _read_json(Path(phase7_report_path))
    phase8_report = _read_json(Path(phase8_report_path))
    dataset_report = _read_json(Path(dataset_version_report_path))
    dvc_metadata_hash = _read_dvc_metadata_hash(Path(dvc_metadata_path))
    retraining_events = _read_optional_csv(Path(retraining_events_path))
    drift_metrics_path = Path(drift_metrics_path)
    phase7_model_path = Path(phase7_model_path)

    contract_features = pd.read_csv(contract_features_path)
    contract_targets = pd.read_csv(contract_targets_path)
    pool_features = pd.read_csv(pool_features_path)
    pool_targets = pd.read_csv(pool_targets_path)
    holdout = _phase7_holdout(contract_features, contract_targets)

    selected_event = _select_latest_retraining_event(retraining_events)
    retraining_requested = selected_event is not None or force_retraining_check
    phase7_model_artifact = _read_pickle(phase7_model_path)
    candidate_predictions_path: Path | None = None
    candidate_pool_scores_path: Path | None = None

    if selected_event is not None:
        candidate_source = "phase9_retrained_candidate"
        candidate_result = _retrain_candidate_from_event(
            selected_event=selected_event,
            stream_dir=Path(stream_dir),
            holdout_ids=holdout.test_ids,
            output_dir=output_path / RETRAINED_CANDIDATE_DIRNAME,
            random_state=RANDOM_STATE,
        )
        candidate_model = candidate_result["model"]
        candidate_model_path = candidate_result["model_path"]
        candidate_training_rows = int(candidate_result["training_rows"])
        removed_for_holdout = int(candidate_result["removed_for_holdout"])
    else:
        candidate_source = "phase7_reference_model"
        candidate_model = phase7_model_artifact["classifier_model"]
        candidate_model_path = phase7_model_path
        candidate_training_rows = int(phase7_report.get("train_rows", 0))
        removed_for_holdout = 0

    validation_report, candidate_predictions, candidate_pool_scores = _validate_candidate(
        candidate_source=candidate_source,
        candidate_model=candidate_model,
        holdout=holdout,
        phase7_report=phase7_report,
        phase8_report=phase8_report,
        dataset_report=dataset_report,
        dvc_metadata_hash=dvc_metadata_hash,
        contract_features=contract_features,
        contract_targets=contract_targets,
        pool_features=pool_features,
        pool_targets=pool_targets,
        candidate_training_rows=candidate_training_rows,
        removed_for_holdout=removed_for_holdout,
    )

    validation_report_path = output_path / VALIDATION_REPORT_FILENAME
    _write_json(validation_report_path, asdict(validation_report))

    if selected_event is not None:
        candidate_predictions_path = (
            output_path / RETRAINED_CANDIDATE_DIRNAME / RETRAINED_CANDIDATE_PREDICTIONS_FILENAME
        )
        candidate_pool_scores_path = (
            output_path / RETRAINED_CANDIDATE_DIRNAME / RETRAINED_CANDIDATE_POOL_SCORES_FILENAME
        )
        candidate_predictions.to_csv(candidate_predictions_path, index=False)
        candidate_pool_scores.to_csv(candidate_pool_scores_path, index=False)

    dvc_status_clean, dvc_status_output = _dvc_status(dvc_status_check)
    tracking_uri = mlflow_tracking_uri or _default_mlflow_tracking_uri()
    _ensure_local_tracking_parent(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"phase9-{candidate_source}") as run:
        run_id = run.info.run_id
        _log_mlflow_run(
            phase7_report=phase7_report,
            phase8_report=phase8_report,
            validation_report=validation_report,
            retraining_requested=retraining_requested,
            selected_event=selected_event,
        )
        _log_existing_artifact(phase7_report_path, "phase7")
        _log_existing_artifact(phase8_report_path, "phase8")
        _log_existing_artifact(retraining_events_path, "phase8")
        _log_existing_artifact(drift_metrics_path, "phase8")
        _log_existing_artifact(phase7_model_path, "phase7_model")
        _log_existing_artifact(validation_report_path, "phase9")
        if candidate_model_path.exists():
            _log_existing_artifact(candidate_model_path, "candidate_model")

        registered_version, promotion_metadata_path = _promote_if_valid(
            validation_report=validation_report,
            run_id=run_id,
            candidate_model_path=candidate_model_path,
            output_dir=output_path,
            phase7_report_path=Path(phase7_report_path),
            phase8_report_path=Path(phase8_report_path),
        )
        pipeline_report = MLOpsPipelineReport(
            experiment_name=EXPERIMENT_NAME,
            registered_model_name=REGISTERED_MODEL_NAME,
            mlflow_tracking_uri=tracking_uri,
            mlflow_run_id=run_id,
            mlflow_artifact_uri=run.info.artifact_uri,
            candidate_source=candidate_source,
            retraining_requested=retraining_requested,
            selected_retraining_trigger_batch=(
                None if selected_event is None else int(selected_event["trigger_batch"])
            ),
            selected_retraining_detector=(
                None if selected_event is None else str(selected_event["triggering_detector"])
            ),
            validation_passed=validation_report.validation_passed,
            model_registered=registered_version is not None,
            registered_model_version=registered_version,
            dvc_status_checked=dvc_status_check,
            dvc_status_clean=dvc_status_clean,
            dvc_status_output=dvc_status_output,
            fr_ml_04_note=FR_ML_04_NOTE,
            promotion_scope_note=PROMOTION_SCOPE_NOTE,
            kafka_scope_note=KAFKA_SCOPE_NOTE,
            validation_report_path=str(validation_report_path),
            promotion_metadata_path=None if promotion_metadata_path is None else str(promotion_metadata_path),
        )

    report_path = output_path / MLOPS_REPORT_FILENAME
    _write_json(report_path, asdict(pipeline_report))
    return MLOpsPipelinePaths(
        report_path=report_path,
        validation_report_path=validation_report_path,
        promotion_metadata_path=promotion_metadata_path,
        candidate_model_path=candidate_model_path,
        candidate_predictions_path=candidate_predictions_path,
        candidate_pool_scores_path=candidate_pool_scores_path,
    )


@dataclass(frozen=True)
class HoldoutSplit:
    train_ids: tuple[Any, ...]
    test_ids: tuple[Any, ...]
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_train: np.ndarray


def _phase7_holdout(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
) -> HoldoutSplit:
    modeling_data = _prepare_contract_modeling_data(contract_features, contract_targets)
    X = modeling_data.loc[:, MODEL_INPUT_COLUMNS].copy()
    y = modeling_data[TARGET_COLUMN].astype("int64")
    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X,
        y,
        modeling_data[ID_COLUMN],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return HoldoutSplit(
        train_ids=tuple(train_ids.tolist()),
        test_ids=tuple(test_ids.tolist()),
        X_test=X_test,
        y_test=y_test.to_numpy(),
        y_train=y_train.to_numpy(),
    )


def _retrain_candidate_from_event(
    selected_event: pd.Series,
    stream_dir: Path,
    holdout_ids: tuple[Any, ...],
    output_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_batch_ids = _parse_batch_ids(str(selected_event["training_window_batches"]))
    window_data = pd.concat(
        [_read_stream_batch(stream_dir, batch_id) for batch_id in training_batch_ids],
        ignore_index=True,
    )
    reference_data = pd.concat(
        [
            _read_stream_batch(stream_dir, batch_id)
            for batch_id in range(REFERENCE_BATCH_START, REFERENCE_BATCH_END + 1)
        ],
        ignore_index=True,
    )
    replay_count = int(round(len(reference_data) * REPLAY_FRACTION))
    rng = np.random.default_rng(random_state)
    replay = (
        reference_data.iloc[rng.choice(len(reference_data), size=replay_count, replace=False)]
        if replay_count > 0
        else reference_data.iloc[[]]
    )
    candidate_training = pd.concat([window_data, replay], ignore_index=True)
    holdout_id_set = set(holdout_ids)
    rows_before_filter = len(candidate_training)
    candidate_training = candidate_training.loc[
        ~candidate_training[ID_COLUMN].isin(holdout_id_set)
    ].copy()
    removed_for_holdout = rows_before_filter - len(candidate_training)
    if candidate_training[TARGET_COLUMN].nunique() < 2:
        raise MLOpsPipelineError("retrained candidate data must contain both target classes")

    model, calibration_cv_folds, calibration_skipped, calibration_note = _fit_probability_model(
        candidate_training.loc[:, MODEL_INPUT_COLUMNS].copy(),
        candidate_training[TARGET_COLUMN].astype("int64"),
    )
    model_path = output_dir / RETRAINED_CANDIDATE_MODEL_FILENAME
    with model_path.open("wb") as model_file:
        pickle.dump(
            {
                "classifier_model": model,
                "training_window_batches": training_batch_ids,
                "calibration_cv_folds": calibration_cv_folds,
                "calibration_skipped": calibration_skipped,
                "calibration_note": calibration_note,
                "fair_evaluation_note": FAIR_EVALUATION_NOTE,
            },
            model_file,
        )
    return {
        "model": model,
        "model_path": model_path,
        "training_rows": len(candidate_training),
        "removed_for_holdout": removed_for_holdout,
    }


def _validate_candidate(
    candidate_source: str,
    candidate_model: Any,
    holdout: HoldoutSplit,
    phase7_report: dict[str, Any],
    phase8_report: dict[str, Any],
    dataset_report: dict[str, Any],
    dvc_metadata_hash: str | None,
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    pool_features: pd.DataFrame,
    pool_targets: pd.DataFrame,
    candidate_training_rows: int,
    removed_for_holdout: int,
) -> tuple[CandidateValidationReport, pd.DataFrame, pd.DataFrame]:
    probabilities = _predict_positive_probability(candidate_model, holdout.X_test)
    metrics = _compute_metrics(holdout.y_test, probabilities, holdout.y_train)
    candidate_predictions = pd.DataFrame(
        {
            ID_COLUMN: holdout.test_ids,
            TARGET_COLUMN: holdout.y_test,
            "predicted_claim_probability": probabilities,
        }
    )
    all_contract_predictions = _score_all_contracts(candidate_model, contract_features, contract_targets)
    pool_scores, _ = _build_pool_risk_scores(all_contract_predictions, pool_features, pool_targets)
    pool_score_coverage = float(pool_scores["pool_risk_score"].notna().mean())

    phase6_hash = _string_or_none(dataset_report, "dvc_output_hash")
    phase7_hash = _string_or_none(phase7_report, "dvc_output_hash")
    phase8_hash = _string_or_none(phase8_report, "dvc_output_hash")
    dvc_hashes = (phase6_hash, phase7_hash, phase8_hash)
    dvc_hash_consistent = None not in dvc_hashes and len(set(dvc_hashes)) == 1
    dvc_metadata_hash_matches_report = dvc_metadata_hash is not None and dvc_metadata_hash == phase6_hash

    baseline_auc = float(phase7_report["baseline_auc"])
    baseline_gini = float(phase7_report["baseline_normalized_gini"])
    auc_beats_baseline = metrics["model_auc"] > baseline_auc
    gini_beats_baseline = metrics["model_normalized_gini"] > baseline_gini
    calibration_delta_passed = (
        metrics["test_probability_rate_delta"] <= CALIBRATION_RATE_DELTA_THRESHOLD
    )
    pool_score_coverage_passed = pool_score_coverage == 1.0
    phase7_acceptance_passed = bool(phase7_report.get("acceptance_passed"))
    phase8_acceptance_passed = bool(phase8_report.get("acceptance_passed"))

    failure_reasons = _validation_failure_reasons(
        phase7_acceptance_passed=phase7_acceptance_passed,
        phase8_acceptance_passed=phase8_acceptance_passed,
        dvc_hash_consistent=dvc_hash_consistent,
        dvc_metadata_hash_matches_report=dvc_metadata_hash_matches_report,
        pool_score_coverage_passed=pool_score_coverage_passed,
        auc_beats_baseline=auc_beats_baseline,
        gini_beats_baseline=gini_beats_baseline,
        calibration_delta_passed=calibration_delta_passed,
    )
    validation_passed = len(failure_reasons) == 0
    return (
        CandidateValidationReport(
            candidate_source=candidate_source,
            validation_passed=validation_passed,
            phase7_acceptance_passed=phase7_acceptance_passed,
            phase8_acceptance_passed=phase8_acceptance_passed,
            dvc_hash_consistent=dvc_hash_consistent,
            dvc_metadata_hash_matches_report=dvc_metadata_hash_matches_report,
            dvc_output_hash=phase6_hash if dvc_hash_consistent else None,
            phase6_dvc_hash=phase6_hash,
            phase7_dvc_hash=phase7_hash,
            phase8_dvc_hash=phase8_hash,
            dvc_metadata_hash=dvc_metadata_hash,
            dataset_git_commit_sha=_string_or_none(phase7_report, "dataset_git_commit_sha"),
            pool_score_coverage=pool_score_coverage,
            pool_score_coverage_passed=pool_score_coverage_passed,
            candidate_auc=metrics["model_auc"],
            baseline_auc=baseline_auc,
            auc_beats_baseline=auc_beats_baseline,
            candidate_normalized_gini=metrics["model_normalized_gini"],
            baseline_normalized_gini=baseline_gini,
            gini_beats_baseline=gini_beats_baseline,
            probability_rate_delta=metrics["test_probability_rate_delta"],
            calibration_delta_threshold=CALIBRATION_RATE_DELTA_THRESHOLD,
            calibration_delta_passed=calibration_delta_passed,
            calibration_threshold_rationale=CALIBRATION_THRESHOLD_RATIONALE,
            evaluated_on_phase7_holdout=True,
            phase7_holdout_test_rows=len(holdout.y_test),
            candidate_training_rows=candidate_training_rows,
            retraining_rows_removed_for_holdout=removed_for_holdout,
            fair_evaluation_note=FAIR_EVALUATION_NOTE,
            failure_reasons=failure_reasons,
        ),
        candidate_predictions,
        pool_scores,
    )


def _score_all_contracts(
    model: Any,
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
) -> pd.DataFrame:
    modeling_data = _prepare_contract_modeling_data(contract_features, contract_targets)
    probabilities = _predict_positive_probability(
        model, modeling_data.loc[:, MODEL_INPUT_COLUMNS].copy()
    )
    predictions = modeling_data.loc[:, [ID_COLUMN, POOL_COLUMN, "Exposure"]].copy()
    predictions[TARGET_COLUMN] = modeling_data[TARGET_COLUMN].to_numpy()
    predictions["predicted_claim_probability"] = probabilities
    return predictions


def _validation_failure_reasons(**checks: bool) -> tuple[str, ...]:
    return tuple(name for name, passed in checks.items() if not passed)


def _select_latest_retraining_event(events: pd.DataFrame) -> pd.Series | None:
    if events.empty:
        return None
    return events.sort_values("trigger_batch").iloc[-1]


def _parse_batch_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _read_stream_batch(stream_dir: Path, batch_id: int) -> pd.DataFrame:
    path = stream_dir / f"batch_{batch_id:03d}.csv"
    if not path.exists():
        raise MLOpsPipelineError(f"stream batch does not exist: {path}")
    return pd.read_csv(path)


def _log_mlflow_run(
    phase7_report: dict[str, Any],
    phase8_report: dict[str, Any],
    validation_report: CandidateValidationReport,
    retraining_requested: bool,
    selected_event: pd.Series | None,
) -> None:
    for name in (
        "model_auc",
        "model_normalized_gini",
        "baseline_auc",
        "baseline_normalized_gini",
        "brier_score",
        "expected_calibration_error",
        "test_empirical_claim_rate",
        "test_mean_predicted_probability",
        "pool_score_coverage",
    ):
        _log_metric_if_present(f"phase7_{name}", phase7_report.get(name))
    data_eval = phase8_report.get("data_drift_evaluation", {})
    concept_eval = phase8_report.get("concept_drift_evaluation", {})
    _log_metric_if_present("phase8_data_detection_latency", data_eval.get("detection_latency_batches"))
    _log_metric_if_present(
        "phase8_concept_detection_latency", concept_eval.get("detection_latency_batches")
    )
    _log_metric_if_present("phase8_retraining_event_count", phase8_report.get("retraining_event_count"))
    mlflow.log_metric("candidate_auc", validation_report.candidate_auc)
    mlflow.log_metric("candidate_normalized_gini", validation_report.candidate_normalized_gini)
    mlflow.log_metric("candidate_probability_rate_delta", validation_report.probability_rate_delta)
    mlflow.log_metric("candidate_pool_score_coverage", validation_report.pool_score_coverage)
    mlflow.log_metric("retraining_requested", float(retraining_requested))
    if selected_event is not None:
        mlflow.log_metric("selected_retraining_trigger_batch", float(selected_event["trigger_batch"]))

    tags = {
        "dvc_output_hash": validation_report.dvc_output_hash or "",
        "dvc_tracked_path": str(phase7_report.get("dvc_tracked_path", "")),
        "dataset_git_commit_sha": str(phase7_report.get("dataset_git_commit_sha", "")),
        "phase7_acceptance_passed": str(validation_report.phase7_acceptance_passed),
        "phase8_acceptance_passed": str(validation_report.phase8_acceptance_passed),
        "candidate_source": validation_report.candidate_source,
        "validation_passed": str(validation_report.validation_passed),
    }
    mlflow.set_tags(tags)


def _log_metric_if_present(name: str, value: Any) -> None:
    if value is None:
        return
    mlflow.log_metric(name, float(value))


def _log_existing_artifact(path: Path | str, artifact_path: str) -> None:
    candidate = Path(path)
    if candidate.exists():
        mlflow.log_artifact(str(candidate), artifact_path=artifact_path)


def _promote_if_valid(
    validation_report: CandidateValidationReport,
    run_id: str,
    candidate_model_path: Path,
    output_dir: Path,
    phase7_report_path: Path,
    phase8_report_path: Path,
) -> tuple[str | None, Path | None]:
    if not validation_report.validation_passed:
        return None, None
    client = MlflowClient()
    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        pass
    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=str(candidate_model_path.resolve()),
        run_id=run_id,
    )
    version = str(model_version.version)
    tags = {
        "dvc_output_hash": validation_report.dvc_output_hash or "",
        "dataset_git_commit_sha": validation_report.dataset_git_commit_sha or "",
        "source_phase": "phase9",
        "validation_status": "passed",
        "phase7_report_path": str(phase7_report_path),
        "phase8_report_path": str(phase8_report_path),
    }
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, key, value)

    promotion_metadata_path = output_dir / PROMOTION_METADATA_FILENAME
    _write_json(
        promotion_metadata_path,
        {
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": version,
            "mlflow_run_id": run_id,
            "candidate_model_path": str(candidate_model_path),
            "dvc_output_hash": validation_report.dvc_output_hash,
            "promotion_scope_note": PROMOTION_SCOPE_NOTE,
        },
    )
    _log_existing_artifact(promotion_metadata_path, "phase9")
    return version, promotion_metadata_path


def _dvc_status(enabled: bool) -> tuple[bool | None, str | None]:
    if not enabled:
        return None, None
    result = subprocess.run(
        ["uv", "run", "dvc", "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0 and "up to date" in output.lower(), output


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MLOpsPipelineError(f"JSON input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_pickle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MLOpsPipelineError(f"pickle artifact does not exist: {path}")
    with path.open("rb") as artifact_file:
        payload = pickle.load(artifact_file)
    if not isinstance(payload, dict):
        raise MLOpsPipelineError(f"pickle artifact is invalid: {path}")
    return payload


def _read_dvc_metadata_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        return None
    outs = metadata.get("outs")
    if not isinstance(outs, list) or not outs or not isinstance(outs[0], dict):
        return None
    first_output = outs[0]
    value = first_output.get("md5") or first_output.get("etag") or first_output.get("checksum")
    return None if value is None else str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _string_or_none(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _default_mlflow_tracking_uri() -> str:
    return f"sqlite:///{(DEFAULT_MLFLOW_TRACKING_DIR.resolve() / 'mlflow.db').as_posix()}"


def _ensure_local_tracking_parent(tracking_uri: str) -> None:
    if not tracking_uri.startswith("sqlite:///"):
        return
    Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Phase 9 MLOps pipeline.")
    parser.add_argument("--phase7-model-path", type=Path, default=DEFAULT_PHASE7_MODEL_PATH)
    parser.add_argument("--phase7-report-path", type=Path, default=DEFAULT_PHASE7_REPORT_PATH)
    parser.add_argument("--phase8-report-path", type=Path, default=DEFAULT_PHASE8_REPORT_PATH)
    parser.add_argument("--retraining-events-path", type=Path, default=DEFAULT_RETRAINING_EVENTS_PATH)
    parser.add_argument("--drift-metrics-path", type=Path, default=DEFAULT_DRIFT_METRICS_PATH)
    parser.add_argument(
        "--dataset-version-report-path",
        type=Path,
        default=DEFAULT_DATASET_VERSION_REPORT_PATH,
    )
    parser.add_argument("--dvc-metadata-path", type=Path, default=DEFAULT_DVC_METADATA_PATH)
    parser.add_argument("--stream-dir", type=Path, default=DEFAULT_STREAM_DIR)
    parser.add_argument("--contract-features-path", type=Path, default=DEFAULT_CONTRACT_FEATURES_PATH)
    parser.add_argument("--contract-targets-path", type=Path, default=DEFAULT_CONTRACT_TARGETS_PATH)
    parser.add_argument("--pool-features-path", type=Path, default=DEFAULT_POOL_FEATURES_PATH)
    parser.add_argument("--pool-targets-path", type=Path, default=DEFAULT_POOL_TARGETS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE9_ARTIFACT_DIR)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--dvc-status-check", action="store_true")
    parser.add_argument(
        "--force-retraining-check",
        action="store_true",
        help="Record a scheduled retraining check even when no drift event exists.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_mlops_pipeline(
        phase7_model_path=args.phase7_model_path,
        phase7_report_path=args.phase7_report_path,
        phase8_report_path=args.phase8_report_path,
        retraining_events_path=args.retraining_events_path,
        drift_metrics_path=args.drift_metrics_path,
        dataset_version_report_path=args.dataset_version_report_path,
        dvc_metadata_path=args.dvc_metadata_path,
        stream_dir=args.stream_dir,
        contract_features_path=args.contract_features_path,
        contract_targets_path=args.contract_targets_path,
        pool_features_path=args.pool_features_path,
        pool_targets_path=args.pool_targets_path,
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        dvc_status_check=args.dvc_status_check,
        force_retraining_check=args.force_retraining_check,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
