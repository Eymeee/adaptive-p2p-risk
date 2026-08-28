from __future__ import annotations

"""Temporal stream simulation for Phase 5.

freMTPL2 has no event timestamp, so this module creates simulated time through
a fixed-seed shuffle. Drift injections are abrupt step functions with logged
ground truth for Phase 8; detection thresholds remain out of scope here.
"""

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.features import DEFAULT_CONTRACT_FEATURES_FILENAME
from src.data.features import DEFAULT_CONTRACT_TARGETS_FILENAME

DEFAULT_CONTRACT_FEATURES_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_FEATURES_FILENAME
DEFAULT_CONTRACT_TARGETS_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_TARGETS_FILENAME
DEFAULT_STREAM_DIR = DEFAULT_PROCESSED_DIR / "stream"
DEFAULT_STREAM_REPORT_FILENAME = "stream_simulation_report.json"
DEFAULT_DRIFT_GROUND_TRUTH_FILENAME = "drift_ground_truth.json"
DEFAULT_N_BATCHES = 20
DEFAULT_RANDOM_SEED = 42
DEFAULT_SEVERITY_SEED = 42
DEFAULT_DATA_DRIFT_START_BATCH = 12
DEFAULT_CONCEPT_DRIFT_START_BATCH = 15
DATA_DRIFT_PATTERN = "abrupt_step_function"
CONCEPT_DRIFT_PATTERN = "abrupt_step_function"
DATA_DRIFT_RULE = (
    "From the start batch onward, rows with Density_band in {'1501-5000', '5001+'} "
    "have Density multiplied by 1.35, then Density_log1p and Density_band are recomputed."
)
CONCEPT_DRIFT_RULE = (
    "From the start batch onward, rows with BonusMalus >= 100 have target_claim_count "
    "incremented by 1 and capped at 4; target indicators, frequency, and severity "
    "aggregates are recomputed."
)
OVERLAP_NOTE = (
    "The overlap is intentional: batches 15-19 have both abrupt data drift and abrupt "
    "concept drift active, possibly on the same rows, so Phase 8 must account for "
    "overlap rather than treating those batches as a single drift source."
)
SYNTHETIC_SEVERITY_STRATEGY = (
    "Added claim severity is sampled deterministically from the population-wide "
    "distribution of positive target_avg_claim_amount values across all contracts, "
    "using an independent numpy random generator seeded by severity_seed. If no "
    "positive observed average severity exists, the fallback amount is 1000.0."
)
PHASE_8_THRESHOLD_NOTE = (
    "Phase 8 detection thresholds, including detected-within-N-batches acceptance "
    "criteria, remain TBD and are not defined by this stream simulation."
)
DENSITY_DRIFT_FACTOR = 1.35
CLAIM_COUNT_CAP = 4
SYNTHETIC_SEVERITY_FALLBACK = 1000.0
DENSITY_BAND_EDGES: tuple[float, ...] = (0.0, 100.0, 500.0, 1500.0, 5000.0, np.inf)
DENSITY_BAND_LABELS: tuple[str, ...] = (
    "0-100",
    "101-500",
    "501-1500",
    "1501-5000",
    "5001+",
)
REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "IDpol",
    "Density",
    "Density_log1p",
    "Density_band",
    "BonusMalus",
    "Exposure",
)
REQUIRED_TARGET_COLUMNS: tuple[str, ...] = (
    "IDpol",
    "target_claim_count",
    "target_has_claim",
    "target_claim_frequency",
    "target_total_claim_amount",
    "target_avg_claim_amount",
)
DRIFT_METADATA_COLUMNS: tuple[str, ...] = (
    "batch_id",
    "event_index",
    "simulated_time_step",
    "data_drift_injected",
    "concept_drift_injected",
)


class StreamSimulationError(ValueError):
    """Raised when feature and target artifacts cannot be simulated as a stream."""


@dataclass(frozen=True)
class StreamBatchReport:
    batch_id: int
    row_count: int
    data_drift_rows: int
    concept_drift_rows: int
    overlap_rows: int


@dataclass(frozen=True)
class DriftInjectionReport:
    drift_type: str
    start_batch: int
    pattern: str
    rule: str
    affected_rows_by_batch: dict[int, int]
    total_affected_rows: int


@dataclass(frozen=True)
class StreamSimulationReport:
    feature_input_rows: int
    target_input_rows: int
    joined_rows: int
    n_batches: int
    random_seed: int
    severity_seed: int
    batch_size_min: int
    batch_size_max: int
    batch_size_mean: float
    output_batch_files: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    drift_metadata_columns: tuple[str, ...]
    zero_exposure_rows_during_concept_recompute: int
    data_drift_pattern: str
    concept_drift_pattern: str
    overlap_intentional: bool
    overlap_batch_start: int | None
    overlap_batch_end: int | None
    overlap_note: str
    synthetic_severity_strategy: str
    synthetic_severity_fallback_used: bool
    total_synthetic_severity_added: float
    batch_reports: tuple[StreamBatchReport, ...]


@dataclass(frozen=True)
class StreamSimulationResult:
    batches: tuple[pd.DataFrame, ...]
    report: StreamSimulationReport
    drift_ground_truth: dict[str, Any]


@dataclass(frozen=True)
class StreamDataPaths:
    batch_paths: tuple[Path, ...]
    report_path: Path
    drift_ground_truth_path: Path


def simulate_stream(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    n_batches: int = DEFAULT_N_BATCHES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    severity_seed: int = DEFAULT_SEVERITY_SEED,
    data_drift_start_batch: int = DEFAULT_DATA_DRIFT_START_BATCH,
    concept_drift_start_batch: int = DEFAULT_CONCEPT_DRIFT_START_BATCH,
) -> StreamSimulationResult:
    _validate_inputs(
        contract_features,
        contract_targets,
        n_batches,
        data_drift_start_batch,
        concept_drift_start_batch,
    )

    joined = contract_features.merge(
        contract_targets,
        on="IDpol",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(contract_features) or len(joined) != len(contract_targets):
        raise StreamSimulationError("contract features and targets must join one-to-one by IDpol")

    shuffle_rng = np.random.default_rng(random_seed)
    shuffled = joined.iloc[shuffle_rng.permutation(len(joined))].reset_index(drop=True)
    _prepare_mutable_drift_columns(shuffled)
    shuffled["event_index"] = np.arange(len(shuffled), dtype="int64")
    shuffled["simulated_time_step"] = shuffled["event_index"]
    shuffled["batch_id"] = _assign_batch_ids(len(shuffled), n_batches)
    shuffled["data_drift_injected"] = False
    shuffled["concept_drift_injected"] = False

    severity_rng = np.random.default_rng(severity_seed)
    positive_average_severity = shuffled.loc[
        shuffled["target_avg_claim_amount"] > 0, "target_avg_claim_amount"
    ].to_numpy(dtype="float64")
    severity_fallback_used = len(positive_average_severity) == 0

    data_affected_by_batch: dict[int, int] = {}
    concept_affected_by_batch: dict[int, int] = {}
    overlap_affected_by_batch: dict[int, int] = {}
    total_synthetic_severity_added = 0.0
    zero_exposure_rows_during_concept_recompute = 0

    for batch_id in range(n_batches):
        batch_mask = shuffled["batch_id"] == batch_id

        data_mask = batch_mask & (batch_id >= data_drift_start_batch) & (
            shuffled["Density_band"].isin(("1501-5000", "5001+"))
        )
        if data_mask.any():
            shuffled.loc[data_mask, "Density"] = (
                shuffled.loc[data_mask, "Density"] * DENSITY_DRIFT_FACTOR
            )
            shuffled.loc[data_mask, "Density_log1p"] = np.log1p(
                shuffled.loc[data_mask, "Density"]
            )
            shuffled.loc[data_mask, "Density_band"] = _density_band(
                shuffled.loc[data_mask, "Density"]
            )
            shuffled.loc[data_mask, "data_drift_injected"] = True

        concept_mask = batch_mask & (batch_id >= concept_drift_start_batch) & (
            shuffled["BonusMalus"] >= 100
        )
        changed_claim_mask = concept_mask & (shuffled["target_claim_count"] < CLAIM_COUNT_CAP)
        changed_claim_count = int(changed_claim_mask.sum())
        if concept_mask.any():
            shuffled.loc[concept_mask, "concept_drift_injected"] = True
            zero_exposure_rows_during_concept_recompute += int(
                (shuffled.loc[concept_mask, "Exposure"] == 0).sum()
            )
            if changed_claim_count > 0:
                added_severity = _sample_added_severity(
                    rng=severity_rng,
                    positive_average_severity=positive_average_severity,
                    size=changed_claim_count,
                    fallback_used=severity_fallback_used,
                )
                total_synthetic_severity_added += float(added_severity.sum())
                shuffled.loc[changed_claim_mask, "target_claim_count"] = (
                    shuffled.loc[changed_claim_mask, "target_claim_count"] + 1
                )
                shuffled.loc[changed_claim_mask, "target_total_claim_amount"] = (
                    shuffled.loc[changed_claim_mask, "target_total_claim_amount"]
                    + added_severity
                )

            shuffled.loc[concept_mask, "target_has_claim"] = (
                shuffled.loc[concept_mask, "target_claim_count"] > 0
            ).astype("int64")
            shuffled.loc[concept_mask, "target_claim_frequency"] = np.where(
                shuffled.loc[concept_mask, "Exposure"] > 0,
                shuffled.loc[concept_mask, "target_claim_count"]
                / shuffled.loc[concept_mask, "Exposure"],
                0.0,
            )
            shuffled.loc[concept_mask, "target_avg_claim_amount"] = np.where(
                shuffled.loc[concept_mask, "target_claim_count"] > 0,
                shuffled.loc[concept_mask, "target_total_claim_amount"]
                / shuffled.loc[concept_mask, "target_claim_count"],
                0.0,
            )

        data_affected_by_batch[batch_id] = int(data_mask.sum())
        concept_affected_by_batch[batch_id] = int(concept_mask.sum())
        overlap_affected_by_batch[batch_id] = int((data_mask & concept_mask).sum())

    batches = tuple(
        shuffled.loc[shuffled["batch_id"] == batch_id].reset_index(drop=True)
        for batch_id in range(n_batches)
    )
    batch_reports = tuple(
        StreamBatchReport(
            batch_id=batch_id,
            row_count=len(batches[batch_id]),
            data_drift_rows=data_affected_by_batch[batch_id],
            concept_drift_rows=concept_affected_by_batch[batch_id],
            overlap_rows=overlap_affected_by_batch[batch_id],
        )
        for batch_id in range(n_batches)
    )
    batch_sizes = [len(batch) for batch in batches]
    overlap_start, overlap_end = _overlap_range(
        data_drift_start_batch, concept_drift_start_batch, n_batches
    )

    report = StreamSimulationReport(
        feature_input_rows=len(contract_features),
        target_input_rows=len(contract_targets),
        joined_rows=len(joined),
        n_batches=n_batches,
        random_seed=random_seed,
        severity_seed=severity_seed,
        batch_size_min=min(batch_sizes),
        batch_size_max=max(batch_sizes),
        batch_size_mean=float(np.mean(batch_sizes)),
        output_batch_files=tuple(_batch_filename(batch_id) for batch_id in range(n_batches)),
        feature_columns=tuple(contract_features.columns),
        target_columns=tuple(contract_targets.columns),
        drift_metadata_columns=DRIFT_METADATA_COLUMNS,
        zero_exposure_rows_during_concept_recompute=zero_exposure_rows_during_concept_recompute,
        data_drift_pattern=DATA_DRIFT_PATTERN,
        concept_drift_pattern=CONCEPT_DRIFT_PATTERN,
        overlap_intentional=overlap_start is not None,
        overlap_batch_start=overlap_start,
        overlap_batch_end=overlap_end,
        overlap_note=OVERLAP_NOTE,
        synthetic_severity_strategy=SYNTHETIC_SEVERITY_STRATEGY,
        synthetic_severity_fallback_used=severity_fallback_used,
        total_synthetic_severity_added=round(total_synthetic_severity_added, 12),
        batch_reports=batch_reports,
    )
    drift_ground_truth = _build_drift_ground_truth(
        n_batches=n_batches,
        random_seed=random_seed,
        severity_seed=severity_seed,
        data_drift_start_batch=data_drift_start_batch,
        concept_drift_start_batch=concept_drift_start_batch,
        data_affected_by_batch=data_affected_by_batch,
        concept_affected_by_batch=concept_affected_by_batch,
        overlap_affected_by_batch=overlap_affected_by_batch,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        severity_fallback_used=severity_fallback_used,
        total_synthetic_severity_added=total_synthetic_severity_added,
    )
    return StreamSimulationResult(
        batches=batches,
        report=report,
        drift_ground_truth=drift_ground_truth,
    )


def write_stream_batches(
    result: StreamSimulationResult,
    output_dir: Path | str = DEFAULT_STREAM_DIR,
) -> StreamDataPaths:
    stream_dir = Path(output_dir)
    stream_dir.mkdir(parents=True, exist_ok=True)

    batch_paths: list[Path] = []
    for batch_id, batch in enumerate(result.batches):
        batch_path = stream_dir / _batch_filename(batch_id)
        batch.to_csv(batch_path, index=False)
        batch_paths.append(batch_path)

    report_path = stream_dir / DEFAULT_STREAM_REPORT_FILENAME
    drift_ground_truth_path = stream_dir / DEFAULT_DRIFT_GROUND_TRUTH_FILENAME
    _write_json_report(report_path, asdict(result.report))
    _write_json_report(drift_ground_truth_path, result.drift_ground_truth)
    return StreamDataPaths(
        batch_paths=tuple(batch_paths),
        report_path=report_path,
        drift_ground_truth_path=drift_ground_truth_path,
    )


def run_stream_simulation_pipeline(
    contract_features_path: Path | str = DEFAULT_CONTRACT_FEATURES_PATH,
    contract_targets_path: Path | str = DEFAULT_CONTRACT_TARGETS_PATH,
    output_dir: Path | str = DEFAULT_STREAM_DIR,
    n_batches: int = DEFAULT_N_BATCHES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    severity_seed: int = DEFAULT_SEVERITY_SEED,
    data_drift_start_batch: int = DEFAULT_DATA_DRIFT_START_BATCH,
    concept_drift_start_batch: int = DEFAULT_CONCEPT_DRIFT_START_BATCH,
) -> StreamDataPaths:
    contract_features = pd.read_csv(contract_features_path)
    contract_targets = pd.read_csv(contract_targets_path)
    result = simulate_stream(
        contract_features=contract_features,
        contract_targets=contract_targets,
        n_batches=n_batches,
        random_seed=random_seed,
        severity_seed=severity_seed,
        data_drift_start_batch=data_drift_start_batch,
        concept_drift_start_batch=concept_drift_start_batch,
    )
    return write_stream_batches(result, output_dir)


def _validate_inputs(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    n_batches: int,
    data_drift_start_batch: int,
    concept_drift_start_batch: int,
) -> None:
    _validate_required_columns(contract_features, REQUIRED_FEATURE_COLUMNS, "contract features")
    _validate_required_columns(contract_targets, REQUIRED_TARGET_COLUMNS, "contract targets")
    if contract_features.empty:
        raise StreamSimulationError("contract features must contain at least one row")
    if contract_targets.empty:
        raise StreamSimulationError("contract targets must contain at least one row")
    if contract_features["IDpol"].duplicated().any():
        raise StreamSimulationError("contract features must have unique IDpol values")
    if contract_targets["IDpol"].duplicated().any():
        raise StreamSimulationError("contract targets must have unique IDpol values")
    if n_batches < 1:
        raise StreamSimulationError("n_batches must be at least 1")
    if n_batches > len(contract_features):
        raise StreamSimulationError("n_batches must be no greater than the number of rows")
    if not 0 <= data_drift_start_batch < n_batches:
        raise StreamSimulationError("data_drift_start_batch must be within the batch range")
    if not 0 <= concept_drift_start_batch < n_batches:
        raise StreamSimulationError("concept_drift_start_batch must be within the batch range")


def _validate_required_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], table_name: str
) -> None:
    missing_columns = tuple(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise StreamSimulationError(
            f"{table_name} data is missing required stream simulation columns: "
            f"{', '.join(missing_columns)}"
        )


def _prepare_mutable_drift_columns(stream: pd.DataFrame) -> None:
    for column in (
        "Density",
        "Density_log1p",
        "target_claim_frequency",
        "target_total_claim_amount",
        "target_avg_claim_amount",
    ):
        stream[column] = stream[column].astype("float64")
    stream["target_claim_count"] = stream["target_claim_count"].astype("int64")
    stream["target_has_claim"] = stream["target_has_claim"].astype("int64")


def _assign_batch_ids(row_count: int, n_batches: int) -> np.ndarray:
    split_indices = np.array_split(np.arange(row_count), n_batches)
    return np.concatenate(
        [
            np.full(len(indices), batch_id, dtype="int64")
            for batch_id, indices in enumerate(split_indices)
        ]
    )


def _density_band(density: pd.Series) -> pd.Series:
    return pd.cut(
        density,
        bins=DENSITY_BAND_EDGES,
        labels=DENSITY_BAND_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")


def _sample_added_severity(
    rng: np.random.Generator,
    positive_average_severity: np.ndarray,
    size: int,
    fallback_used: bool,
) -> np.ndarray:
    if fallback_used:
        return np.full(size, SYNTHETIC_SEVERITY_FALLBACK, dtype="float64")
    return rng.choice(positive_average_severity, size=size, replace=True).astype("float64")


def _overlap_range(
    data_drift_start_batch: int, concept_drift_start_batch: int, n_batches: int
) -> tuple[int | None, int | None]:
    start = max(data_drift_start_batch, concept_drift_start_batch)
    if start >= n_batches:
        return None, None
    return start, n_batches - 1


def _build_drift_ground_truth(
    n_batches: int,
    random_seed: int,
    severity_seed: int,
    data_drift_start_batch: int,
    concept_drift_start_batch: int,
    data_affected_by_batch: dict[int, int],
    concept_affected_by_batch: dict[int, int],
    overlap_affected_by_batch: dict[int, int],
    overlap_start: int | None,
    overlap_end: int | None,
    severity_fallback_used: bool,
    total_synthetic_severity_added: float,
) -> dict[str, Any]:
    return {
        "n_batches": n_batches,
        "random_seed": random_seed,
        "severity_seed": severity_seed,
        "data_drift": asdict(
            DriftInjectionReport(
                drift_type="data",
                start_batch=data_drift_start_batch,
                pattern=DATA_DRIFT_PATTERN,
                rule=DATA_DRIFT_RULE,
                affected_rows_by_batch=data_affected_by_batch,
                total_affected_rows=sum(data_affected_by_batch.values()),
            )
        ),
        "concept_drift": asdict(
            DriftInjectionReport(
                drift_type="concept",
                start_batch=concept_drift_start_batch,
                pattern=CONCEPT_DRIFT_PATTERN,
                rule=CONCEPT_DRIFT_RULE,
                affected_rows_by_batch=concept_affected_by_batch,
                total_affected_rows=sum(concept_affected_by_batch.values()),
            )
        ),
        "overlap": {
            "intentional": overlap_start is not None,
            "batch_start": overlap_start,
            "batch_end": overlap_end,
            "note": OVERLAP_NOTE,
            "affected_rows_by_batch": overlap_affected_by_batch,
            "total_affected_rows": sum(overlap_affected_by_batch.values()),
        },
        "synthetic_severity": {
            "strategy": SYNTHETIC_SEVERITY_STRATEGY,
            "fallback_used": severity_fallback_used,
            "total_synthetic_severity_added": round(total_synthetic_severity_added, 12),
        },
        "phase_8_threshold_note": PHASE_8_THRESHOLD_NOTE,
    }


def _batch_filename(batch_id: int) -> str:
    return f"batch_{batch_id:03d}.csv"


def _write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic temporal stream batches with logged drift ground truth."
    )
    parser.add_argument(
        "--contract-features-path",
        type=Path,
        default=DEFAULT_CONTRACT_FEATURES_PATH,
        help=f"Contract features CSV path. Defaults to {DEFAULT_CONTRACT_FEATURES_PATH}.",
    )
    parser.add_argument(
        "--contract-targets-path",
        type=Path,
        default=DEFAULT_CONTRACT_TARGETS_PATH,
        help=f"Contract targets CSV path. Defaults to {DEFAULT_CONTRACT_TARGETS_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STREAM_DIR,
        help=f"Stream output directory. Defaults to {DEFAULT_STREAM_DIR}.",
    )
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--severity-seed", type=int, default=DEFAULT_SEVERITY_SEED)
    parser.add_argument(
        "--data-drift-start-batch",
        type=int,
        default=DEFAULT_DATA_DRIFT_START_BATCH,
    )
    parser.add_argument(
        "--concept-drift-start-batch",
        type=int,
        default=DEFAULT_CONCEPT_DRIFT_START_BATCH,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_stream_simulation_pipeline(
        contract_features_path=args.contract_features_path,
        contract_targets_path=args.contract_targets_path,
        output_dir=args.output_dir,
        n_batches=args.n_batches,
        random_seed=args.random_seed,
        severity_seed=args.severity_seed,
        data_drift_start_batch=args.data_drift_start_batch,
        concept_drift_start_batch=args.concept_drift_start_batch,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
