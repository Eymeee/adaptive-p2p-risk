from __future__ import annotations

"""Feature engineering for Phase 4.

The CdCT names average claim rate as a pool feature, but this module stores
`pool_claim_rate` in the target table. Claim rate is derived from observed
claims, so using it as a model input would leak the outcome we later want to
predict. The same separation is applied at contract level for `ClaimNb`.
"""

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning import DEFAULT_CLEANED_SEVERITY_FILENAME
from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.pools import DEFAULT_POOLED_FREQUENCY_FILENAME

DEFAULT_POOLED_FREQUENCY_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_POOLED_FREQUENCY_FILENAME
DEFAULT_CLEANED_SEVERITY_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CLEANED_SEVERITY_FILENAME
DEFAULT_CONTRACT_FEATURES_FILENAME = "contract_features.csv"
DEFAULT_CONTRACT_TARGETS_FILENAME = "contract_targets.csv"
DEFAULT_POOL_FEATURES_FILENAME = "pool_features.csv"
DEFAULT_POOL_TARGETS_FILENAME = "pool_targets.csv"
DEFAULT_FEATURE_REPORT_FILENAME = "feature_engineering_report.json"

CONTRACT_FEATURE_COLUMNS: tuple[str, ...] = (
    "IDpol",
    "pool_id",
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density",
    "Density_log1p",
    "VehBrand",
    "VehGas",
    "Area",
    "Region",
    "VehAge_band",
    "DrivAge_band",
    "BonusMalus_band",
    "Density_band",
)
SERVING_REQUIRED_CONTRACT_COLUMNS: tuple[str, ...] = (
    "pool_id",
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region",
)
REQUIRED_FREQUENCY_COLUMNS: tuple[str, ...] = (
    "IDpol",
    "pool_id",
    "ClaimNb",
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region",
)
SEVERITY_COLUMNS: tuple[str, ...] = ("IDpol", "ClaimAmount")
LEAKAGE_EXCLUDED_COLUMNS: tuple[str, ...] = (
    "ClaimNb",
    "ClaimNb_declared",
    "ClaimAmount",
    "target_claim_count",
    "target_has_claim",
    "target_claim_frequency",
    "target_total_claim_amount",
    "target_avg_claim_amount",
    "pool_claim_count",
    "pool_has_claim",
    "pool_claim_rate",
    "pool_total_claim_amount",
    "pool_avg_claim_amount",
)
CLAIM_RATE_RECLASSIFICATION_NOTE = (
    "The CdCT lists average claim rate as a pool feature, but this implementation "
    "stores pool_claim_rate in pool_targets because it is calculated from observed "
    "claims. Using it as an input feature would leak the outcome, analogous to "
    "using ClaimNb as a contract-level predictor."
)
FEATURE_RATIONALE_NOTE = (
    "Features describe policyholder, vehicle, geography, exposure, and simulated "
    "pool context. Observed claim outcomes and severity totals are kept in target "
    "tables to preserve a clean modeling boundary."
)
BANDING_METHODOLOGY = (
    "Fixed domain thresholds are used instead of quantiles so the feature schema "
    "is stable for future inference and easy to audit."
)


@dataclass(frozen=True)
class BandDefinition:
    edges: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class FeatureEngineeringReport:
    frequency_input_rows: int
    severity_input_rows: int | None
    contract_feature_rows: int
    contract_target_rows: int
    pool_feature_rows: int
    pool_target_rows: int
    contract_feature_columns: tuple[str, ...]
    contract_target_columns: tuple[str, ...]
    pool_feature_columns: tuple[str, ...]
    pool_target_columns: tuple[str, ...]
    leakage_excluded_columns: tuple[str, ...]
    claim_rate_reclassification_note: str
    banding_methodology: str
    band_definitions: dict[str, BandDefinition]
    zero_exposure_contract_rows: int
    zero_exposure_pools: int
    number_of_pools: int
    severity_input_used: bool
    feature_rationale_note: str


@dataclass(frozen=True)
class FeatureEngineeringResult:
    contract_features: pd.DataFrame
    contract_targets: pd.DataFrame
    pool_features: pd.DataFrame
    pool_targets: pd.DataFrame
    report: FeatureEngineeringReport


@dataclass(frozen=True)
class FeatureDataPaths:
    contract_features_path: Path
    contract_targets_path: Path
    pool_features_path: Path
    pool_targets_path: Path
    report_path: Path


class FeatureEngineeringError(ValueError):
    """Raised when pooled data cannot be transformed into feature tables."""


_BAND_EDGES: dict[str, tuple[float, ...]] = {
    "VehAge_band": (0.0, 2.0, 5.0, 10.0, np.inf),
    "DrivAge_band": (0.0, 25.0, 35.0, 50.0, 65.0, np.inf),
    "BonusMalus_band": (0.0, 50.0, 75.0, 100.0, 150.0, np.inf),
    "Density_band": (0.0, 100.0, 500.0, 1500.0, 5000.0, np.inf),
}
_BAND_SOURCE_COLUMNS: dict[str, str] = {
    "VehAge_band": "VehAge",
    "DrivAge_band": "DrivAge",
    "BonusMalus_band": "BonusMalus",
    "Density_band": "Density",
}
_BAND_LABELS: dict[str, tuple[str, ...]] = {
    "VehAge_band": ("0-2", "3-5", "6-10", "11+"),
    "DrivAge_band": ("0-25", "26-35", "36-50", "51-65", "66+"),
    "BonusMalus_band": ("0-50", "51-75", "76-100", "101-150", "151+"),
    "Density_band": ("0-100", "101-500", "501-1500", "1501-5000", "5001+"),
}


def build_features(
    frequency: pd.DataFrame, severity: pd.DataFrame | None = None
) -> FeatureEngineeringResult:
    """Build leak-free contract and pool feature/target tables."""
    _validate_frequency(frequency)
    _validate_severity(severity)

    working_frequency = frequency.copy()
    working_frequency["Density_log1p"] = np.log1p(working_frequency["Density"])
    _add_bands(working_frequency)

    severity_aggregates = _build_severity_aggregates(severity)
    contract_features = _build_contract_features(working_frequency)
    contract_targets = _build_contract_targets(working_frequency, severity_aggregates)
    pool_features = _build_pool_features(working_frequency)
    pool_targets = _build_pool_targets(working_frequency, contract_targets)

    zero_exposure_contract_rows = int((working_frequency["Exposure"] == 0).sum())
    zero_exposure_pools = int(
        (working_frequency.groupby("pool_id", observed=True)["Exposure"].sum() == 0).sum()
    )

    report = FeatureEngineeringReport(
        frequency_input_rows=len(frequency),
        severity_input_rows=None if severity is None else len(severity),
        contract_feature_rows=len(contract_features),
        contract_target_rows=len(contract_targets),
        pool_feature_rows=len(pool_features),
        pool_target_rows=len(pool_targets),
        contract_feature_columns=tuple(contract_features.columns),
        contract_target_columns=tuple(contract_targets.columns),
        pool_feature_columns=tuple(pool_features.columns),
        pool_target_columns=tuple(pool_targets.columns),
        leakage_excluded_columns=LEAKAGE_EXCLUDED_COLUMNS,
        claim_rate_reclassification_note=CLAIM_RATE_RECLASSIFICATION_NOTE,
        banding_methodology=BANDING_METHODOLOGY,
        band_definitions=_serializable_band_definitions(),
        zero_exposure_contract_rows=zero_exposure_contract_rows,
        zero_exposure_pools=zero_exposure_pools,
        number_of_pools=int(working_frequency["pool_id"].nunique()),
        severity_input_used=severity is not None,
        feature_rationale_note=FEATURE_RATIONALE_NOTE,
    )
    return FeatureEngineeringResult(
        contract_features=contract_features,
        contract_targets=contract_targets,
        pool_features=pool_features,
        pool_targets=pool_targets,
        report=report,
    )


def build_serving_contract_features(contracts: pd.DataFrame) -> pd.DataFrame:
    """Build contract features for inference using the Phase 4 feature policy.

    Serving deliberately imports this helper instead of duplicating thresholds in
    the API layer, so training and inference cannot drift apart when Phase 4
    banding definitions change.
    """
    _validate_serving_contracts(contracts)
    working_contracts = contracts.copy()
    working_contracts["Density_log1p"] = np.log1p(working_contracts["Density"])
    _add_bands(working_contracts)
    output_columns = tuple(
        column for column in CONTRACT_FEATURE_COLUMNS if column in working_contracts.columns
    )
    return working_contracts.loc[:, output_columns].copy()


def write_feature_data(
    result: FeatureEngineeringResult,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> FeatureDataPaths:
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = FeatureDataPaths(
        contract_features_path=processed_dir / DEFAULT_CONTRACT_FEATURES_FILENAME,
        contract_targets_path=processed_dir / DEFAULT_CONTRACT_TARGETS_FILENAME,
        pool_features_path=processed_dir / DEFAULT_POOL_FEATURES_FILENAME,
        pool_targets_path=processed_dir / DEFAULT_POOL_TARGETS_FILENAME,
        report_path=processed_dir / DEFAULT_FEATURE_REPORT_FILENAME,
    )
    result.contract_features.to_csv(paths.contract_features_path, index=False)
    result.contract_targets.to_csv(paths.contract_targets_path, index=False)
    result.pool_features.to_csv(paths.pool_features_path, index=False)
    result.pool_targets.to_csv(paths.pool_targets_path, index=False)
    _write_json_report(paths.report_path, asdict(result.report))
    return paths


def run_feature_engineering_pipeline(
    frequency_path: Path | str = DEFAULT_POOLED_FREQUENCY_PATH,
    severity_path: Path | str | None = DEFAULT_CLEANED_SEVERITY_PATH,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> FeatureDataPaths:
    frequency = pd.read_csv(frequency_path)
    severity = _read_optional_severity(severity_path)
    result = build_features(frequency, severity)
    return write_feature_data(result, output_dir)


def _validate_frequency(frequency: pd.DataFrame) -> None:
    missing_columns = tuple(
        column for column in REQUIRED_FREQUENCY_COLUMNS if column not in frequency.columns
    )
    if missing_columns:
        raise FeatureEngineeringError(
            "frequency data is missing required feature engineering columns: "
            f"{', '.join(missing_columns)}"
        )
    if frequency.empty:
        raise FeatureEngineeringError("frequency data must contain at least one row")
    if frequency.loc[:, REQUIRED_FREQUENCY_COLUMNS].isna().any().any():
        raise FeatureEngineeringError("feature engineering inputs must not contain missing values")


def _validate_severity(severity: pd.DataFrame | None) -> None:
    if severity is None:
        return
    missing_columns = tuple(column for column in SEVERITY_COLUMNS if column not in severity.columns)
    if missing_columns:
        raise FeatureEngineeringError(
            f"severity data is missing required columns: {', '.join(missing_columns)}"
        )


def _validate_serving_contracts(contracts: pd.DataFrame) -> None:
    missing_columns = tuple(
        column for column in SERVING_REQUIRED_CONTRACT_COLUMNS if column not in contracts.columns
    )
    if missing_columns:
        raise FeatureEngineeringError(
            "serving contracts are missing required columns: "
            f"{', '.join(missing_columns)}"
        )
    if contracts.empty:
        raise FeatureEngineeringError("serving contracts must contain at least one row")
    if contracts.loc[:, SERVING_REQUIRED_CONTRACT_COLUMNS].isna().any().any():
        raise FeatureEngineeringError("serving contract inputs must not contain missing values")


def _add_bands(frequency: pd.DataFrame) -> None:
    for band_column, source_column in _BAND_SOURCE_COLUMNS.items():
        frequency[band_column] = pd.cut(
            frequency[source_column],
            bins=_BAND_EDGES[band_column],
            labels=_BAND_LABELS[band_column],
            include_lowest=True,
            right=True,
        ).astype("string")


def _build_contract_features(frequency: pd.DataFrame) -> pd.DataFrame:
    return frequency.loc[:, CONTRACT_FEATURE_COLUMNS].copy()


def _build_contract_targets(
    frequency: pd.DataFrame, severity_aggregates: pd.DataFrame
) -> pd.DataFrame:
    targets = frequency.loc[:, ["IDpol", "ClaimNb", "Exposure"]].copy()
    targets["target_claim_count"] = targets["ClaimNb"].astype("int64")
    targets["target_has_claim"] = (targets["target_claim_count"] > 0).astype("int64")
    targets["target_claim_frequency"] = np.where(
        targets["Exposure"] > 0,
        targets["target_claim_count"] / targets["Exposure"],
        0.0,
    )
    targets = targets.merge(severity_aggregates, on="IDpol", how="left")
    targets["target_total_claim_amount"] = targets["target_total_claim_amount"].fillna(0.0)
    targets["target_avg_claim_amount"] = targets["target_avg_claim_amount"].fillna(0.0)
    return targets.loc[
        :,
        [
            "IDpol",
            "target_claim_count",
            "target_has_claim",
            "target_claim_frequency",
            "target_total_claim_amount",
            "target_avg_claim_amount",
        ],
    ]


def _build_severity_aggregates(severity: pd.DataFrame | None) -> pd.DataFrame:
    if severity is None:
        return pd.DataFrame(
            columns=["IDpol", "target_total_claim_amount", "target_avg_claim_amount"]
        )
    return (
        severity.groupby("IDpol", as_index=False, observed=True)
        .agg(
            target_total_claim_amount=("ClaimAmount", "sum"),
            target_avg_claim_amount=("ClaimAmount", "mean"),
        )
        .astype({"target_total_claim_amount": "float64", "target_avg_claim_amount": "float64"})
    )


def _build_pool_features(frequency: pd.DataFrame) -> pd.DataFrame:
    pool_features = (
        frequency.groupby("pool_id", as_index=False, observed=True)
        .agg(
            pool_size=("IDpol", "size"),
            pool_total_exposure=("Exposure", "sum"),
            pool_mean_exposure=("Exposure", "mean"),
            pool_mean_veh_power=("VehPower", "mean"),
            pool_std_veh_power=("VehPower", "std"),
            pool_mean_veh_age=("VehAge", "mean"),
            pool_std_veh_age=("VehAge", "std"),
            pool_mean_driv_age=("DrivAge", "mean"),
            pool_std_driv_age=("DrivAge", "std"),
            pool_mean_bonus_malus=("BonusMalus", "mean"),
            pool_std_bonus_malus=("BonusMalus", "std"),
            pool_mean_density=("Density", "mean"),
            pool_std_density=("Density", "std"),
            pool_mean_density_log1p=("Density_log1p", "mean"),
            pool_std_density_log1p=("Density_log1p", "std"),
            pool_veh_brand_diversity=("VehBrand", "nunique"),
            pool_veh_gas_diversity=("VehGas", "nunique"),
            pool_area_diversity=("Area", "nunique"),
            pool_region_diversity=("Region", "nunique"),
        )
        .fillna(0.0)
    )
    return pool_features


def _build_pool_targets(
    frequency: pd.DataFrame, contract_targets: pd.DataFrame
) -> pd.DataFrame:
    target_source = frequency.loc[:, ["IDpol", "pool_id", "Exposure"]].merge(
        contract_targets,
        on="IDpol",
        how="left",
    )
    pool_targets = (
        target_source.groupby("pool_id", as_index=False, observed=True)
        .agg(
            pool_claim_count=("target_claim_count", "sum"),
            pool_total_exposure=("Exposure", "sum"),
            pool_total_claim_amount=("target_total_claim_amount", "sum"),
        )
    )
    pool_targets["pool_has_claim"] = (pool_targets["pool_claim_count"] > 0).astype("int64")
    pool_targets["pool_claim_rate"] = np.where(
        pool_targets["pool_total_exposure"] > 0,
        pool_targets["pool_claim_count"] / pool_targets["pool_total_exposure"],
        0.0,
    )
    pool_targets["pool_avg_claim_amount"] = np.where(
        pool_targets["pool_claim_count"] > 0,
        pool_targets["pool_total_claim_amount"] / pool_targets["pool_claim_count"],
        0.0,
    )
    return pool_targets.loc[
        :,
        [
            "pool_id",
            "pool_claim_count",
            "pool_has_claim",
            "pool_claim_rate",
            "pool_total_claim_amount",
            "pool_avg_claim_amount",
        ],
    ]


def _serializable_band_definitions() -> dict[str, BandDefinition]:
    return {
        band_column: BandDefinition(
            edges=tuple(_format_edge(edge) for edge in edges),
            labels=_BAND_LABELS[band_column],
        )
        for band_column, edges in _BAND_EDGES.items()
    }


def _format_edge(edge: float) -> str:
    if np.isposinf(edge):
        return "inf"
    return str(int(edge)) if float(edge).is_integer() else str(edge)


def _read_optional_severity(severity_path: Path | str | None) -> pd.DataFrame | None:
    if severity_path is None:
        return None
    path = Path(severity_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build contract and pool feature/target tables from pooled freMTPL2 data."
    )
    parser.add_argument(
        "--frequency-path",
        type=Path,
        default=DEFAULT_POOLED_FREQUENCY_PATH,
        help=f"Pooled frequency CSV path. Defaults to {DEFAULT_POOLED_FREQUENCY_PATH}.",
    )
    parser.add_argument(
        "--severity-path",
        type=Path,
        default=DEFAULT_CLEANED_SEVERITY_PATH,
        help=(
            f"Cleaned severity CSV path. Defaults to {DEFAULT_CLEANED_SEVERITY_PATH}; "
            "if absent, severity targets are filled with zero."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed output directory. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_feature_engineering_pipeline(
        frequency_path=args.frequency_path,
        severity_path=args.severity_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
