from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.data.cleaning import DEFAULT_PROCESSED_DIR # pyrefly: ignore

DEFAULT_CLEANED_FREQUENCY_PATH = DEFAULT_PROCESSED_DIR / "freMTPL2freq_cleaned.csv"
DEFAULT_POOLED_FREQUENCY_FILENAME = "freMTPL2freq_pooled.csv"
DEFAULT_POOL_REPORT_FILENAME = "pool_construction_report.json"
NUMERIC_FEATURES: tuple[str, ...] = (
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density_log1p",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("VehBrand", "VehGas", "Area", "Region")
EXCLUDED_FEATURES: tuple[str, ...] = (
    "IDpol",
    "ClaimNb",
    "ClaimNb_declared",
    "Exposure",
    "ClaimAmount",
)
DEFAULT_K_MIN = 10
DEFAULT_K_MAX = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_SILHOUETTE_SAMPLE_SIZE = 10_000
DEFAULT_SILHOUETTE_SAMPLE_SEED = 42
SCALING_STRATEGY = "StandardScaler applied to numeric clustering features"
ENCODING_STRATEGY = "OneHotEncoder(handle_unknown='ignore') applied to categorical features"
CLUSTERING_ALGORITHM = "MiniBatchKMeans"
MIXED_TYPE_CLUSTERING_TRADEOFF = (
    "One-hot encoding plus k-means is reproducible and keeps this phase within "
    "standard scikit-learn tooling, but it is an approximation for mixed-type "
    "data. High-cardinality categoricals can numerically dominate Euclidean "
    "distance by column count alone; for example, Region can expand to about "
    "20 binary columns compared with only 5 scaled numeric columns."
)
FEATURE_RATIONALE: dict[str, str] = {
    "VehPower": "Vehicle power is a proxy for performance and risk-taking exposure.",
    "VehAge": "Vehicle age captures condition and safety-technology differences.",
    "DrivAge": "Driver age represents a core shared risk segment for motor insurance.",
    "BonusMalus": "Bonus-malus summarizes prior insurance risk behavior.",
    "Density_log1p": (
        "Traffic density captures urban/rural exposure; log1p reduces skew before scaling."
    ),
    "VehBrand": (
        "Vehicle brand can proxy repair cost, vehicle segment, and owner usage profile."
    ),
    "VehGas": "Fuel type can proxy vehicle category and usage pattern.",
    "Area": "Area groups local traffic intensity and claim-environment conditions.",
    "Region": "Region groups broader geographic traffic and claim-environment patterns.",
}


class PoolConstructionError(ValueError):
    """Raised when cleaned contract data cannot be clustered into valid pools."""


@dataclass(frozen=True)
class KScanResult:
    k: int
    silhouette_score: float | None
    inertia: float


@dataclass(frozen=True)
class PoolConstructionReport:
    input_rows: int
    output_rows: int
    k_min: int
    k_max: int
    selected_k: int
    random_seed: int
    silhouette_sample_size: int
    silhouette_sample_seed: int
    k_scan_results: tuple[KScanResult, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    excluded_features: tuple[str, ...]
    scaling_strategy: str
    encoding_strategy: str
    clustering_algorithm: str
    feature_rationale: dict[str, str]
    mixed_type_clustering_tradeoff: str
    pool_size_min: int
    pool_size_max: int
    pool_size_mean: float


@dataclass(frozen=True)
class PoolConstructionResult:
    frequency: pd.DataFrame
    report: PoolConstructionReport


@dataclass(frozen=True)
class PooledDataPaths:
    frequency_path: Path
    report_path: Path


def construct_pools(
    frequency: pd.DataFrame,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    random_seed: int = DEFAULT_RANDOM_SEED,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    silhouette_sample_seed: int = DEFAULT_SILHOUETTE_SAMPLE_SEED,
) -> PoolConstructionResult:
    """Construct reproducible risk pools from shared non-claim contract attributes."""
    _validate_inputs(frequency, k_min, k_max, silhouette_sample_size)

    clustering_frame = _build_clustering_frame(frequency)
    transformed = _build_preprocessor().fit_transform(clustering_frame)
    sample_indices = _sample_indices(
        row_count=len(frequency),
        sample_size=silhouette_sample_size,
        sample_seed=silhouette_sample_seed,
    )

    k_scan_results: list[KScanResult] = []
    best_k: int | None = None
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        model = _build_model(k, random_seed)
        labels = model.fit_predict(transformed)
        score = _score_labels(transformed, labels, sample_indices)
        k_scan_results.append(
            KScanResult(
                k=k,
                silhouette_score=None if np.isnan(score) else round(float(score), 12),
                inertia=round(float(model.inertia_), 12),
            )
        )

        if not np.isnan(score) and score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        raise PoolConstructionError("no valid silhouette score was produced for the k scan")

    selected_model = _build_model(best_k, random_seed)
    selected_labels = selected_model.fit_predict(transformed)
    pooled_frequency = frequency.copy()
    pooled_frequency["pool_id"] = selected_labels.astype("int64")

    pool_sizes = pooled_frequency["pool_id"].value_counts()
    report = PoolConstructionReport(
        input_rows=len(frequency),
        output_rows=len(pooled_frequency),
        k_min=k_min,
        k_max=k_max,
        selected_k=best_k,
        random_seed=random_seed,
        silhouette_sample_size=len(sample_indices),
        silhouette_sample_seed=silhouette_sample_seed,
        k_scan_results=tuple(k_scan_results),
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        excluded_features=EXCLUDED_FEATURES,
        scaling_strategy=SCALING_STRATEGY,
        encoding_strategy=ENCODING_STRATEGY,
        clustering_algorithm=CLUSTERING_ALGORITHM,
        feature_rationale=FEATURE_RATIONALE,
        mixed_type_clustering_tradeoff=MIXED_TYPE_CLUSTERING_TRADEOFF,
        pool_size_min=int(pool_sizes.min()),
        pool_size_max=int(pool_sizes.max()),
        pool_size_mean=float(pool_sizes.mean()),
    )
    return PoolConstructionResult(frequency=pooled_frequency, report=report)


def write_pooled_data(
    result: PoolConstructionResult,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> PooledDataPaths:
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frequency_path = processed_dir / DEFAULT_POOLED_FREQUENCY_FILENAME
    report_path = processed_dir / DEFAULT_POOL_REPORT_FILENAME
    result.frequency.to_csv(frequency_path, index=False)
    _write_json_report(report_path, asdict(result.report))
    return PooledDataPaths(frequency_path=frequency_path, report_path=report_path)


def run_pool_construction_pipeline(
    input_path: Path | str = DEFAULT_CLEANED_FREQUENCY_PATH,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    random_seed: int = DEFAULT_RANDOM_SEED,
    silhouette_sample_size: int = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    silhouette_sample_seed: int = DEFAULT_SILHOUETTE_SAMPLE_SEED,
) -> PooledDataPaths:
    frequency = pd.read_csv(input_path)
    result = construct_pools(
        frequency=frequency,
        k_min=k_min,
        k_max=k_max,
        random_seed=random_seed,
        silhouette_sample_size=silhouette_sample_size,
        silhouette_sample_seed=silhouette_sample_seed,
    )
    return write_pooled_data(result, output_dir)


def _validate_inputs(
    frequency: pd.DataFrame, k_min: int, k_max: int, silhouette_sample_size: int
) -> None:
    required_columns = tuple(
        column for column in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES) if column != "Density_log1p"
    )
    missing_columns = tuple(column for column in required_columns if column not in frequency.columns)
    if missing_columns:
        raise PoolConstructionError(
            f"frequency data is missing required pool construction columns: "
            f"{', '.join(missing_columns)}"
        )
    if frequency.empty:
        raise PoolConstructionError("frequency data must contain at least one row")
    if k_min < 2:
        raise PoolConstructionError("k_min must be at least 2")
    if k_max < k_min:
        raise PoolConstructionError("k_max must be greater than or equal to k_min")
    if k_max >= len(frequency):
        raise PoolConstructionError("k_max must be smaller than the number of rows")
    if silhouette_sample_size < 2:
        raise PoolConstructionError("silhouette_sample_size must be at least 2")
    if frequency.loc[:, required_columns].isna().any().any():
        raise PoolConstructionError("pool construction features must not contain missing values")


def _build_clustering_frame(frequency: pd.DataFrame) -> pd.DataFrame:
    clustering_frame = frequency.copy()
    clustering_frame["Density_log1p"] = np.log1p(clustering_frame["Density"])
    return clustering_frame.loc[:, (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def _build_model(k: int, random_seed: int) -> MiniBatchKMeans:
    return MiniBatchKMeans(
        n_clusters=k,
        random_state=random_seed,
        n_init=3,
        batch_size=4096,
    )


def _sample_indices(row_count: int, sample_size: int, sample_seed: int) -> np.ndarray:
    effective_sample_size = min(sample_size, row_count)
    rng = np.random.default_rng(sample_seed)
    return np.sort(rng.choice(row_count, size=effective_sample_size, replace=False))


def _score_labels(matrix: Any, labels: np.ndarray, sample_indices: np.ndarray) -> float:
    sampled_labels = labels[sample_indices]
    unique_label_count = len(np.unique(sampled_labels))
    if unique_label_count < 2 or unique_label_count >= len(sampled_labels):
        return float("nan")
    return float(silhouette_score(matrix[sample_indices], sampled_labels))


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct deterministic simulated pools from cleaned freMTPL2 contracts."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_CLEANED_FREQUENCY_PATH,
        help=f"Cleaned frequency CSV path. Defaults to {DEFAULT_CLEANED_FREQUENCY_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed output directory. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--silhouette-sample-seed",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE_SEED,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_pool_construction_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        random_seed=args.random_seed,
        silhouette_sample_size=args.silhouette_sample_size,
        silhouette_sample_seed=args.silhouette_sample_seed,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
