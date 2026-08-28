"""Continual learning and drift detection for Phase 8.

The 2-batch detection latency is provisional: it was chosen on Codex's
technical recommendation and is pending supervisor confirmation by Dr. Jbilou.
Phase 8 keeps the Phase 7 reference model fixed; online SGD updates are
candidate adaptation artifacts only and are not promoted here.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.streaming import CONCEPT_DRIFT_PATTERN
from src.data.streaming import DATA_DRIFT_PATTERN
from src.data.streaming import DEFAULT_DRIFT_GROUND_TRUTH_FILENAME
from src.data.streaming import DEFAULT_N_BATCHES
from src.data.streaming import DEFAULT_STREAM_DIR
from src.models.risk import DEFAULT_ARTIFACT_DIR as DEFAULT_PHASE7_ARTIFACT_DIR
from src.models.risk import MODEL_ARTIFACT_FILENAME
from src.models.risk import MODEL_INPUT_COLUMNS
from src.models.risk import NUMERIC_FEATURE_COLUMNS
from src.models.risk import RISK_REPORT_FILENAME as PHASE7_RISK_REPORT_FILENAME
from src.models.risk import TARGET_COLUMN

DEFAULT_PHASE8_ARTIFACT_DIR = Path("artifacts/phase8")
DEFAULT_MODEL_ARTIFACT_PATH = DEFAULT_PHASE7_ARTIFACT_DIR / MODEL_ARTIFACT_FILENAME
DEFAULT_RISK_REPORT_PATH = DEFAULT_PHASE7_ARTIFACT_DIR / PHASE7_RISK_REPORT_FILENAME
DEFAULT_DRIFT_GROUND_TRUTH_PATH = DEFAULT_STREAM_DIR / DEFAULT_DRIFT_GROUND_TRUTH_FILENAME

DRIFT_METRICS_FILENAME = "drift_metrics.csv"
RETRAINING_EVENTS_FILENAME = "retraining_events.csv"
CONTINUAL_REPORT_FILENAME = "continual_learning_report.json"

REFERENCE_BATCH_START = 0
REFERENCE_BATCH_END = 9
PROVISIONAL_LATENCY_BATCHES = 2
PSI_WARNING_THRESHOLD = 0.10
PSI_TRIGGER_THRESHOLD = 0.25
CONCEPT_RESIDUAL_DELTA_THRESHOLD = 0.01
CONCEPT_RESIDUAL_Z_THRESHOLD = 3.0
SLIDING_WINDOW_BATCHES = 5
REPLAY_FRACTION = 0.10
RANDOM_STATE = 42
EPSILON = 1e-6
PSI_NUMERIC_BINS = 10
HIGH_DENSITY_BANDS = frozenset(("1501-5000", "5001+"))

PROVISIONAL_THRESHOLD_NOTE = (
    "The 2-batch detection latency threshold is provisional, chosen on Codex's "
    "technical recommendation, and pending supervisor confirmation by Dr. Jbilou."
)
CONCEPT_THRESHOLD_NOTE = (
    "Concept drift thresholds z-score >= 3.0 and absolute residual delta >= 0.01 "
    "are independent technical defaults, not tuned against the known batch-15 "
    "injection point."
)
FROZEN_PREPROCESSING_NOTE = (
    "The online learner reuses the fitted Phase 7 StandardScaler/OneHotEncoder "
    "preprocessor; the transformer is never refit during stream processing, so "
    "drift is not absorbed by changing scaling or category mappings."
)
TWO_MODEL_RELATIONSHIP_NOTE = (
    "The Phase 7 model remains the monitored reference/scoring model throughout "
    "Phase 8. The online SGDClassifier is a candidate adaptation artifact only; "
    "replacement and promotion are deferred to Phase 9."
)
REPLAY_NOTE = (
    "Sliding-window retraining uses the last 5 batches plus a deterministic 10% "
    "replay sample from reference batches 0-9 to reduce catastrophic forgetting."
)
DATA_DRIFT_DETECTOR_NOTE = (
    "Data drift is detected with standard PSI thresholds on whole-batch model "
    "inputs plus a Density_log1p PSI on the predeclared high-density segment "
    "used by Phase 5's data-drift rule. The segment PSI prevents a known "
    "subpopulation shift from being diluted in whole-batch PSI and should be "
    "read as controlled-simulation instrumentation, not validated general "
    "sensitivity."
)


class DriftMonitoringError(ValueError):
    """Raised when Phase 8 drift monitoring inputs are invalid."""


@dataclass(frozen=True)
class DriftEvaluation:
    drift_type: str
    ground_truth_start_batch: int
    first_detected_batch: int | None
    detection_latency_batches: int | None
    detected_within_provisional_threshold: bool
    false_positive_batches: tuple[int, ...]
    overlap_detected_batches: tuple[int, ...]


@dataclass(frozen=True)
class ContinualLearningReport:
    batch_count: int
    reference_batch_start: int
    reference_batch_end: int
    monitored_feature_columns: tuple[str, ...]
    psi_warning_threshold: float
    psi_trigger_threshold: float
    psi_numeric_bins: int
    concept_residual_delta_threshold: float
    concept_residual_z_threshold: float
    data_drift_detector_note: str
    provisional_latency_batches: int
    provisional_threshold_note: str
    concept_threshold_note: str
    data_drift_pattern: str
    concept_drift_pattern: str
    data_drift_evaluation: DriftEvaluation
    concept_drift_evaluation: DriftEvaluation
    overlap_batch_start: int | None
    overlap_batch_end: int | None
    eligible_fraction_note: str
    frozen_preprocessing_used: bool
    frozen_preprocessing_note: str
    online_model_name: str
    online_partial_fit_updates: int
    sliding_window_batches: int
    replay_fraction: float
    replay_note: str
    retraining_event_count: int
    two_model_relationship_note: str
    dvc_output_hash: str | None
    dvc_tracked_path: str | None
    dataset_git_commit_sha: str | None
    phase7_model_auc: float | None
    phase7_model_normalized_gini: float | None
    acceptance_passed: bool


@dataclass(frozen=True)
class DriftMonitoringResult:
    drift_metrics: pd.DataFrame
    retraining_events: pd.DataFrame
    report: ContinualLearningReport


@dataclass(frozen=True)
class DriftMonitoringPaths:
    drift_metrics_path: Path
    retraining_events_path: Path
    report_path: Path


@dataclass(frozen=True)
class PsiReference:
    numeric_edges: dict[str, tuple[float, ...]]
    numeric_distribution: dict[str, tuple[float, ...]]
    categorical_categories: dict[str, tuple[str, ...]]
    categorical_distribution: dict[str, tuple[float, ...]]


def monitor_drift(
    batches: tuple[pd.DataFrame, ...],
    model_artifact: dict[str, Any],
    drift_ground_truth: dict[str, Any],
    phase7_report: dict[str, Any] | None = None,
    provisional_latency_batches: int = PROVISIONAL_LATENCY_BATCHES,
    random_state: int = RANDOM_STATE,
) -> DriftMonitoringResult:
    """Detect data/concept drift and simulate candidate continual updates."""
    _validate_batches(batches)
    classifier_model = model_artifact["classifier_model"]
    frozen_preprocessor = _extract_frozen_preprocessor(model_artifact)
    reference_batches = batches[REFERENCE_BATCH_START : REFERENCE_BATCH_END + 1]
    reference_data = pd.concat(reference_batches, ignore_index=True)
    _validate_reference_window(reference_data)

    psi_reference = _build_psi_reference(reference_data)
    reference_predictions = _predict_reference_probabilities(classifier_model, reference_data)
    reference_residuals = _batch_residuals(reference_batches, classifier_model)
    reference_residual_mean = float(np.mean(reference_residuals))
    reference_residual_std = float(np.std(reference_residuals, ddof=0))
    if reference_residual_std < EPSILON:
        reference_residual_std = EPSILON

    online_model = SGDClassifier(loss="log_loss", random_state=random_state)
    reference_encoded = frozen_preprocessor.transform(
        _prepare_model_input(reference_data.loc[:, MODEL_INPUT_COLUMNS].copy())
    )
    online_model.partial_fit(
        reference_encoded,
        reference_data[TARGET_COLUMN].astype("int64").to_numpy(),
        classes=np.array([0, 1]),
    )

    rng = np.random.default_rng(random_state)
    metric_rows: list[dict[str, Any]] = []
    retraining_rows: list[dict[str, Any]] = []
    online_updates = 0

    for batch in batches:
        batch_id = int(batch["batch_id"].iloc[0])
        probabilities = _predict_reference_probabilities(classifier_model, batch)
        empirical_claim_rate = float(batch[TARGET_COLUMN].mean())
        mean_predicted_probability = float(np.mean(probabilities))
        residual = empirical_claim_rate - mean_predicted_probability
        residual_delta = residual - reference_residual_mean
        residual_z_score = abs(residual_delta) / reference_residual_std
        feature_psi = _compute_feature_psi(batch, psi_reference)
        max_feature = max(feature_psi, key=feature_psi.get)
        max_psi = float(feature_psi[max_feature])
        density_segment_psi = _density_segment_psi(batch, reference_data)
        data_drift_signal_psi = max(max_psi, density_segment_psi)
        data_drift_signal_source = (
            "density_segment_psi" if density_segment_psi > max_psi else "whole_batch_feature_psi"
        )
        data_drift_detected = data_drift_signal_psi >= PSI_TRIGGER_THRESHOLD
        concept_drift_detected = (
            abs(residual_delta) >= CONCEPT_RESIDUAL_DELTA_THRESHOLD
            and residual_z_score >= CONCEPT_RESIDUAL_Z_THRESHOLD
        )
        data_eligible_fraction = float(batch["Density_band"].isin(HIGH_DENSITY_BANDS).mean())
        concept_eligible_fraction = float((batch["BonusMalus"] >= 100).mean())

        metric_rows.append(
            {
                "batch_id": batch_id,
                "row_count": len(batch),
                "max_psi_feature": max_feature,
                "max_psi": max_psi,
                "density_segment_psi": density_segment_psi,
                "data_drift_signal_psi": data_drift_signal_psi,
                "data_drift_signal_source": data_drift_signal_source,
                "data_drift_detected": data_drift_detected,
                "empirical_claim_rate": empirical_claim_rate,
                "mean_predicted_probability": mean_predicted_probability,
                "concept_residual": residual,
                "concept_residual_delta": residual_delta,
                "concept_residual_z_score": residual_z_score,
                "concept_drift_detected": concept_drift_detected,
                "data_drift_ground_truth": bool(batch["data_drift_injected"].any()),
                "concept_drift_ground_truth": bool(batch["concept_drift_injected"].any()),
                "overlap_ground_truth": bool(
                    (batch["data_drift_injected"] & batch["concept_drift_injected"]).any()
                ),
                "data_drift_eligible_fraction": data_eligible_fraction,
                "concept_drift_eligible_fraction": concept_eligible_fraction,
            }
        )

        encoded_batch = frozen_preprocessor.transform(
            _prepare_model_input(batch.loc[:, MODEL_INPUT_COLUMNS].copy())
        )
        online_model.partial_fit(encoded_batch, batch[TARGET_COLUMN].astype("int64").to_numpy())
        online_updates += 1

        if data_drift_detected or concept_drift_detected:
            retraining_rows.append(
                _build_retraining_event(
                    batch_id=batch_id,
                    triggering_detector=_triggering_detector(
                        data_drift_detected, concept_drift_detected
                    ),
                    batches=batches,
                    frozen_preprocessor=frozen_preprocessor,
                    rng=rng,
                )
            )

    drift_metrics = pd.DataFrame(metric_rows)
    retraining_events = pd.DataFrame(retraining_rows, columns=_retraining_event_columns())
    overlap_start, overlap_end = _overlap_range_from_ground_truth(drift_ground_truth)
    data_eval = _evaluate_detection(
        drift_type="data",
        ground_truth_start_batch=_ground_truth_start(drift_ground_truth, "data_drift"),
        detected_batches=tuple(
            drift_metrics.loc[drift_metrics["data_drift_detected"], "batch_id"].astype(int)
        ),
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        latency_batches=provisional_latency_batches,
    )
    concept_eval = _evaluate_detection(
        drift_type="concept",
        ground_truth_start_batch=_ground_truth_start(drift_ground_truth, "concept_drift"),
        detected_batches=tuple(
            drift_metrics.loc[drift_metrics["concept_drift_detected"], "batch_id"].astype(int)
        ),
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        latency_batches=provisional_latency_batches,
    )
    report = ContinualLearningReport(
        batch_count=len(batches),
        reference_batch_start=REFERENCE_BATCH_START,
        reference_batch_end=REFERENCE_BATCH_END,
        monitored_feature_columns=MODEL_INPUT_COLUMNS,
        psi_warning_threshold=PSI_WARNING_THRESHOLD,
        psi_trigger_threshold=PSI_TRIGGER_THRESHOLD,
        psi_numeric_bins=PSI_NUMERIC_BINS,
        concept_residual_delta_threshold=CONCEPT_RESIDUAL_DELTA_THRESHOLD,
        concept_residual_z_threshold=CONCEPT_RESIDUAL_Z_THRESHOLD,
        data_drift_detector_note=DATA_DRIFT_DETECTOR_NOTE,
        provisional_latency_batches=provisional_latency_batches,
        provisional_threshold_note=PROVISIONAL_THRESHOLD_NOTE,
        concept_threshold_note=CONCEPT_THRESHOLD_NOTE,
        data_drift_pattern=str(drift_ground_truth.get("data_drift", {}).get("pattern", DATA_DRIFT_PATTERN)),
        concept_drift_pattern=str(
            drift_ground_truth.get("concept_drift", {}).get("pattern", CONCEPT_DRIFT_PATTERN)
        ),
        data_drift_evaluation=data_eval,
        concept_drift_evaluation=concept_eval,
        overlap_batch_start=overlap_start,
        overlap_batch_end=overlap_end,
        eligible_fraction_note=(
            "data_drift_eligible_fraction is logged from batch 12 onward for high-density "
            "rows; concept_drift_eligible_fraction is logged from batch 15 onward for "
            "BonusMalus >= 100 rows."
        ),
        frozen_preprocessing_used=True,
        frozen_preprocessing_note=FROZEN_PREPROCESSING_NOTE,
        online_model_name=online_model.__class__.__name__,
        online_partial_fit_updates=online_updates,
        sliding_window_batches=SLIDING_WINDOW_BATCHES,
        replay_fraction=REPLAY_FRACTION,
        replay_note=REPLAY_NOTE,
        retraining_event_count=len(retraining_events),
        two_model_relationship_note=TWO_MODEL_RELATIONSHIP_NOTE,
        dvc_output_hash=_string_or_none(phase7_report, "dvc_output_hash"),
        dvc_tracked_path=_string_or_none(phase7_report, "dvc_tracked_path"),
        dataset_git_commit_sha=_string_or_none(phase7_report, "dataset_git_commit_sha"),
        phase7_model_auc=_float_or_none(phase7_report, "model_auc"),
        phase7_model_normalized_gini=_float_or_none(phase7_report, "model_normalized_gini"),
        acceptance_passed=(
            data_eval.detected_within_provisional_threshold
            and concept_eval.detected_within_provisional_threshold
        ),
    )
    return DriftMonitoringResult(
        drift_metrics=drift_metrics,
        retraining_events=retraining_events,
        report=report,
    )


def write_drift_monitoring_artifacts(
    result: DriftMonitoringResult,
    output_dir: Path | str = DEFAULT_PHASE8_ARTIFACT_DIR,
) -> DriftMonitoringPaths:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = DriftMonitoringPaths(
        drift_metrics_path=artifact_dir / DRIFT_METRICS_FILENAME,
        retraining_events_path=artifact_dir / RETRAINING_EVENTS_FILENAME,
        report_path=artifact_dir / CONTINUAL_REPORT_FILENAME,
    )
    result.drift_metrics.to_csv(paths.drift_metrics_path, index=False)
    result.retraining_events.to_csv(paths.retraining_events_path, index=False)
    paths.report_path.write_text(
        json.dumps(asdict(result.report), indent=2, default=str),
        encoding="utf-8",
    )
    return paths


def run_drift_monitoring_pipeline(
    stream_dir: Path | str = DEFAULT_STREAM_DIR,
    drift_ground_truth_path: Path | str = DEFAULT_DRIFT_GROUND_TRUTH_PATH,
    model_artifact_path: Path | str = DEFAULT_MODEL_ARTIFACT_PATH,
    phase7_report_path: Path | str = DEFAULT_RISK_REPORT_PATH,
    output_dir: Path | str = DEFAULT_PHASE8_ARTIFACT_DIR,
    enforce_acceptance: bool = True,
    provisional_latency_batches: int = PROVISIONAL_LATENCY_BATCHES,
) -> DriftMonitoringPaths:
    batches = _read_stream_batches(Path(stream_dir))
    drift_ground_truth = json.loads(Path(drift_ground_truth_path).read_text(encoding="utf-8"))
    model_artifact = _read_pickle_artifact(Path(model_artifact_path))
    phase7_report = _read_optional_json_report(Path(phase7_report_path))
    result = monitor_drift(
        batches=batches,
        model_artifact=model_artifact,
        drift_ground_truth=drift_ground_truth,
        phase7_report=phase7_report,
        provisional_latency_batches=provisional_latency_batches,
    )
    paths = write_drift_monitoring_artifacts(result, output_dir)
    if enforce_acceptance and not result.report.acceptance_passed:
        raise DriftMonitoringError(
            "Phase 8 provisional acceptance failed after writing artifacts: injected drifts "
            "were not both detected within the provisional latency threshold."
        )
    return paths


def _validate_batches(batches: tuple[pd.DataFrame, ...]) -> None:
    if len(batches) <= REFERENCE_BATCH_END + 1:
        raise DriftMonitoringError("stream must contain enough batches for reference and monitoring")
    required = (
        "batch_id",
        "Density_band",
        "BonusMalus",
        "data_drift_injected",
        "concept_drift_injected",
        TARGET_COLUMN,
        *MODEL_INPUT_COLUMNS,
    )
    for index, batch in enumerate(batches):
        missing = tuple(column for column in required if column not in batch.columns)
        if missing:
            raise DriftMonitoringError(
                f"batch {index} is missing drift monitoring columns: {', '.join(missing)}"
            )
        if batch.empty:
            raise DriftMonitoringError(f"batch {index} must not be empty")


def _validate_reference_window(reference_data: pd.DataFrame) -> None:
    if reference_data["data_drift_injected"].any() or reference_data["concept_drift_injected"].any():
        raise DriftMonitoringError("reference batches must not contain injected drift")
    if reference_data[TARGET_COLUMN].nunique() < 2:
        raise DriftMonitoringError("reference batches must contain both target classes")


def _extract_frozen_preprocessor(model_artifact: dict[str, Any]) -> Any:
    preprocessor = model_artifact.get("feature_preprocessor")
    if preprocessor is not None:
        return preprocessor
    anomaly_preprocessor = model_artifact.get("anomaly_preprocessor")
    if anomaly_preprocessor is not None:
        return anomaly_preprocessor
    raise DriftMonitoringError(
        "Phase 7 artifact does not contain a reusable fitted preprocessor; rerun Phase 7."
    )


def _predict_reference_probabilities(classifier_model: Any, batch: pd.DataFrame) -> np.ndarray:
    features = _prepare_model_input(batch.loc[:, MODEL_INPUT_COLUMNS].copy())
    classes = list(classifier_model.classes_)
    positive_index = classes.index(1)
    return classifier_model.predict_proba(features)[:, positive_index]


def _prepare_model_input(features: pd.DataFrame) -> pd.DataFrame:
    prepared = features.copy()
    for column in MODEL_INPUT_COLUMNS:
        if column not in prepared.columns:
            raise DriftMonitoringError(f"missing model input column: {column}")
    for column in set(MODEL_INPUT_COLUMNS) - set(NUMERIC_FEATURE_COLUMNS):
        prepared[column] = prepared[column].astype("string")
    return prepared.loc[:, MODEL_INPUT_COLUMNS]


def _batch_residuals(batches: tuple[pd.DataFrame, ...], classifier_model: Any) -> tuple[float, ...]:
    residuals: list[float] = []
    for batch in batches:
        probabilities = _predict_reference_probabilities(classifier_model, batch)
        residuals.append(float(batch[TARGET_COLUMN].mean() - np.mean(probabilities)))
    return tuple(residuals)


def _build_psi_reference(reference_data: pd.DataFrame) -> PsiReference:
    numeric_edges: dict[str, tuple[float, ...]] = {}
    numeric_distribution: dict[str, tuple[float, ...]] = {}
    categorical_categories: dict[str, tuple[str, ...]] = {}
    categorical_distribution: dict[str, tuple[float, ...]] = {}

    for column in NUMERIC_FEATURE_COLUMNS:
        values = reference_data[column].astype("float64").to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, PSI_NUMERIC_BINS + 1)))
        if len(edges) < 3:
            low = float(np.min(values))
            high = float(np.max(values))
            if low == high:
                high = low + EPSILON
            edges = np.array([low, high])
        edges[0] = -np.inf
        edges[-1] = np.inf
        numeric_edges[column] = tuple(float(edge) for edge in edges)
        numeric_distribution[column] = _numeric_distribution(values, numeric_edges[column])

    for column in set(MODEL_INPUT_COLUMNS) - set(NUMERIC_FEATURE_COLUMNS):
        categories = tuple(
            sorted(reference_data[column].astype("string").fillna("__missing__").unique().tolist())
        )
        categorical_categories[column] = categories
        categorical_distribution[column] = _categorical_distribution(
            reference_data[column], categories
        )

    return PsiReference(
        numeric_edges=numeric_edges,
        numeric_distribution=numeric_distribution,
        categorical_categories=categorical_categories,
        categorical_distribution=categorical_distribution,
    )


def _compute_feature_psi(batch: pd.DataFrame, reference: PsiReference) -> dict[str, float]:
    feature_psi: dict[str, float] = {}
    for column in NUMERIC_FEATURE_COLUMNS:
        actual = _numeric_distribution(
            batch[column].astype("float64").to_numpy(), reference.numeric_edges[column]
        )
        feature_psi[column] = _population_stability_index(
            reference.numeric_distribution[column], actual
        )
    for column in set(MODEL_INPUT_COLUMNS) - set(NUMERIC_FEATURE_COLUMNS):
        actual = _categorical_distribution(batch[column], reference.categorical_categories[column])
        feature_psi[column] = _population_stability_index(reference.categorical_distribution[column], actual)
    return feature_psi


def _density_segment_psi(batch: pd.DataFrame, reference_data: pd.DataFrame) -> float:
    reference_segment = reference_data.loc[
        reference_data["Density_band"].isin(HIGH_DENSITY_BANDS), "Density_log1p"
    ].astype("float64")
    actual_segment = batch.loc[
        batch["Density_band"].isin(HIGH_DENSITY_BANDS), "Density_log1p"
    ].astype("float64")
    if len(reference_segment) < 2 or len(actual_segment) < 2:
        return 0.0
    edges = np.unique(
        np.quantile(reference_segment.to_numpy(), np.linspace(0.0, 1.0, PSI_NUMERIC_BINS + 1))
    )
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_distribution = _numeric_distribution(reference_segment.to_numpy(), tuple(edges))
    actual_distribution = _numeric_distribution(actual_segment.to_numpy(), tuple(edges))
    return _population_stability_index(reference_distribution, actual_distribution)


def _numeric_distribution(values: np.ndarray, edges: tuple[float, ...]) -> tuple[float, ...]:
    counts, _ = np.histogram(values, bins=np.array(edges))
    return _share_distribution(counts)


def _categorical_distribution(values: pd.Series, categories: tuple[str, ...]) -> tuple[float, ...]:
    normalized = values.astype("string").fillna("__missing__")
    counts = [int((normalized == category).sum()) for category in categories]
    counts.append(int((~normalized.isin(categories)).sum()))
    return _share_distribution(np.array(counts, dtype="int64"))


def _share_distribution(counts: np.ndarray) -> tuple[float, ...]:
    total = int(counts.sum())
    if total == 0:
        return tuple(float(EPSILON) for _ in counts)
    shares = counts.astype("float64") / total
    shares = np.clip(shares, EPSILON, None)
    shares = shares / shares.sum()
    return tuple(float(value) for value in shares)


def _population_stability_index(expected: tuple[float, ...], actual: tuple[float, ...]) -> float:
    expected_array = np.clip(np.array(expected, dtype="float64"), EPSILON, None)
    actual_array = np.clip(np.array(actual, dtype="float64"), EPSILON, None)
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


def _build_retraining_event(
    batch_id: int,
    triggering_detector: str,
    batches: tuple[pd.DataFrame, ...],
    frozen_preprocessor: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    window_start = max(0, batch_id - SLIDING_WINDOW_BATCHES + 1)
    window_batch_ids = tuple(range(window_start, batch_id + 1))
    window_data = pd.concat([batches[index] for index in window_batch_ids], ignore_index=True)
    reference_data = pd.concat(
        batches[REFERENCE_BATCH_START : REFERENCE_BATCH_END + 1], ignore_index=True
    )
    replay_rows = int(round(len(reference_data) * REPLAY_FRACTION))
    if replay_rows > 0:
        replay_indices = rng.choice(len(reference_data), size=replay_rows, replace=False)
        replay_data = reference_data.iloc[replay_indices]
        training_data = pd.concat([window_data, replay_data], ignore_index=True)
    else:
        training_data = window_data

    encoded_training = frozen_preprocessor.transform(
        _prepare_model_input(training_data.loc[:, MODEL_INPUT_COLUMNS].copy())
    )
    candidate = SGDClassifier(loss="log_loss", random_state=RANDOM_STATE)
    candidate.fit(encoded_training, training_data[TARGET_COLUMN].astype("int64").to_numpy())
    candidate_auc = _candidate_auc(candidate, encoded_training, training_data[TARGET_COLUMN].to_numpy())
    return {
        "trigger_batch": batch_id,
        "triggering_detector": triggering_detector,
        "event_type": "sliding_window_retraining_candidate",
        "window_start_batch": window_start,
        "window_end_batch": batch_id,
        "training_window_batches": ",".join(str(batch) for batch in window_batch_ids),
        "window_rows": len(window_data),
        "replay_rows": replay_rows,
        "training_rows": len(training_data),
        "candidate_model_name": candidate.__class__.__name__,
        "candidate_training_auc": candidate_auc,
        "candidate_replaces_reference_model": False,
    }


def _candidate_auc(model: SGDClassifier, encoded_features: Any, y: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    probabilities = model.predict_proba(encoded_features)[:, list(model.classes_).index(1)]
    return float(roc_auc_score(y, probabilities))


def _triggering_detector(data_drift_detected: bool, concept_drift_detected: bool) -> str:
    if data_drift_detected and concept_drift_detected:
        return "data_and_concept"
    if data_drift_detected:
        return "data"
    return "concept"


def _evaluate_detection(
    drift_type: str,
    ground_truth_start_batch: int,
    detected_batches: tuple[int, ...],
    overlap_start: int | None,
    overlap_end: int | None,
    latency_batches: int,
) -> DriftEvaluation:
    post_injection = tuple(batch for batch in detected_batches if batch >= ground_truth_start_batch)
    first_detected = min(post_injection) if post_injection else None
    latency = None if first_detected is None else first_detected - ground_truth_start_batch
    false_positives = tuple(batch for batch in detected_batches if batch < ground_truth_start_batch)
    overlap_detected = tuple(
        batch
        for batch in detected_batches
        if overlap_start is not None and overlap_end is not None and overlap_start <= batch <= overlap_end
    )
    return DriftEvaluation(
        drift_type=drift_type,
        ground_truth_start_batch=ground_truth_start_batch,
        first_detected_batch=first_detected,
        detection_latency_batches=latency,
        detected_within_provisional_threshold=latency is not None and latency <= latency_batches,
        false_positive_batches=false_positives,
        overlap_detected_batches=overlap_detected,
    )


def _ground_truth_start(drift_ground_truth: dict[str, Any], key: str) -> int:
    return int(drift_ground_truth[key]["start_batch"])


def _overlap_range_from_ground_truth(
    drift_ground_truth: dict[str, Any]
) -> tuple[int | None, int | None]:
    overlap = drift_ground_truth.get("overlap", {})
    return overlap.get("batch_start"), overlap.get("batch_end")


def _read_stream_batches(stream_dir: Path) -> tuple[pd.DataFrame, ...]:
    batch_paths = sorted(stream_dir.glob("batch_*.csv"))
    if not batch_paths:
        raise DriftMonitoringError(f"no stream batch files found in {stream_dir}")
    return tuple(pd.read_csv(path) for path in batch_paths)


def _read_pickle_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DriftMonitoringError(f"model artifact does not exist: {path}")
    with path.open("rb") as artifact_file:
        payload = pickle.load(artifact_file)
    if not isinstance(payload, dict):
        raise DriftMonitoringError(f"model artifact is invalid: {path}")
    return payload


def _read_optional_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _string_or_none(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _float_or_none(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None or payload.get(key) is None:
        return None
    return float(payload[key])


def _retraining_event_columns() -> list[str]:
    return [
        "trigger_batch",
        "triggering_detector",
        "event_type",
        "window_start_batch",
        "window_end_batch",
        "training_window_batches",
        "window_rows",
        "replay_rows",
        "training_rows",
        "candidate_model_name",
        "candidate_training_auc",
        "candidate_replaces_reference_model",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 8 drift monitoring.")
    parser.add_argument("--stream-dir", type=Path, default=DEFAULT_STREAM_DIR)
    parser.add_argument(
        "--drift-ground-truth-path",
        type=Path,
        default=DEFAULT_DRIFT_GROUND_TRUTH_PATH,
    )
    parser.add_argument("--model-artifact-path", type=Path, default=DEFAULT_MODEL_ARTIFACT_PATH)
    parser.add_argument("--phase7-report-path", type=Path, default=DEFAULT_RISK_REPORT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE8_ARTIFACT_DIR)
    parser.add_argument(
        "--provisional-latency-batches",
        type=int,
        default=PROVISIONAL_LATENCY_BATCHES,
    )
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="Write artifacts without raising if provisional detection acceptance fails.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_drift_monitoring_pipeline(
        stream_dir=args.stream_dir,
        drift_ground_truth_path=args.drift_ground_truth_path,
        model_artifact_path=args.model_artifact_path,
        phase7_report_path=args.phase7_report_path,
        output_dir=args.output_dir,
        enforce_acceptance=not args.no_enforce_acceptance,
        provisional_latency_batches=args.provisional_latency_batches,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
