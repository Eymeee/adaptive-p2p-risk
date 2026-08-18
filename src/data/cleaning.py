from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.ingestion import DEFAULT_FREQUENCY_PATH # pyrefly: ignore
from src.data.ingestion import DEFAULT_SEVERITY_PATH # pyrefly: ignore
from src.data.ingestion import IngestionReport # pyrefly: ignore
from src.data.ingestion import load_raw_data # pyrefly: ignore

CATEGORICAL_COLUMNS: tuple[str, ...] = ("VehBrand", "VehGas", "Area", "Region")
EXPOSURE_CAP = 1.0
CLAIMNB_CAP = 4
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CLEANED_FREQUENCY_FILENAME = "freMTPL2freq_cleaned.csv"
DEFAULT_CLEANED_SEVERITY_FILENAME = "freMTPL2sev_cleaned.csv"
DEFAULT_INGESTION_REPORT_FILENAME = "ingestion_report.json"
DEFAULT_CLEANING_REPORT_FILENAME = "cleaning_report.json"


@dataclass(frozen=True)
class CleaningReport:
    frequency_rows_before: int
    frequency_rows_after: int
    severity_rows_before: int
    severity_rows_after: int
    unmatched_severity_rows_dropped: int
    claimnb_mismatched_policies: int
    exposure_values_capped: int
    claimnb_values_capped: int
    categorical_columns_cast: tuple[str, ...]
    exposure_cap: float
    claimnb_cap: int


@dataclass(frozen=True)
class CleaningResult:
    frequency: pd.DataFrame
    severity: pd.DataFrame
    report: CleaningReport


@dataclass(frozen=True)
class ProcessedDataPaths:
    frequency_path: Path
    severity_path: Path
    ingestion_report_path: Path
    cleaning_report_path: Path


def clean_raw_data(frequency: pd.DataFrame, severity: pd.DataFrame) -> CleaningResult:
    """Clean documented freMTPL2 issues while preserving audit fields for traceability."""
    cleaned_frequency = frequency.copy()
    cleaned_severity = severity.copy()

    frequency_ids = set(cleaned_frequency["IDpol"])
    matched_severity_mask = cleaned_severity["IDpol"].isin(frequency_ids)
    unmatched_severity_rows_dropped = int((~matched_severity_mask).sum())
    cleaned_severity = cleaned_severity.loc[matched_severity_mask].copy()

    observed_claim_counts = cleaned_severity.groupby("IDpol").size()
    cleaned_frequency["ClaimNb_declared"] = cleaned_frequency["ClaimNb"]
    reconciled_claimnb = (
        cleaned_frequency["IDpol"].map(observed_claim_counts).fillna(0).astype("int64")
    )

    claimnb_mismatched_policies = int(
        (cleaned_frequency["ClaimNb_declared"] != reconciled_claimnb).sum()
    )
    claimnb_values_capped = int((reconciled_claimnb > CLAIMNB_CAP).sum())

    cleaned_frequency["ClaimNb"] = reconciled_claimnb.clip(upper=CLAIMNB_CAP)

    exposure_values_capped = int((cleaned_frequency["Exposure"] > EXPOSURE_CAP).sum())
    cleaned_frequency["Exposure"] = cleaned_frequency["Exposure"].clip(upper=EXPOSURE_CAP)

    cast_columns = tuple(
        column for column in CATEGORICAL_COLUMNS if column in cleaned_frequency.columns
    )
    for column in cast_columns:
        cleaned_frequency[column] = cleaned_frequency[column].astype("category")

    report = CleaningReport(
        frequency_rows_before=len(frequency),
        frequency_rows_after=len(cleaned_frequency),
        severity_rows_before=len(severity),
        severity_rows_after=len(cleaned_severity),
        unmatched_severity_rows_dropped=unmatched_severity_rows_dropped,
        claimnb_mismatched_policies=claimnb_mismatched_policies,
        exposure_values_capped=exposure_values_capped,
        claimnb_values_capped=claimnb_values_capped,
        categorical_columns_cast=cast_columns,
        exposure_cap=EXPOSURE_CAP,
        claimnb_cap=CLAIMNB_CAP,
    )
    return CleaningResult(
        frequency=cleaned_frequency,
        severity=cleaned_severity,
        report=report,
    )


def write_cleaned_data(
    cleaning_result: CleaningResult,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
    ingestion_report: IngestionReport | None = None,
) -> ProcessedDataPaths:
    """Persist cleaned datasets and audit reports as the handoff to later phases."""
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frequency_path = processed_dir / DEFAULT_CLEANED_FREQUENCY_FILENAME
    severity_path = processed_dir / DEFAULT_CLEANED_SEVERITY_FILENAME
    ingestion_report_path = processed_dir / DEFAULT_INGESTION_REPORT_FILENAME
    cleaning_report_path = processed_dir / DEFAULT_CLEANING_REPORT_FILENAME

    cleaning_result.frequency.to_csv(frequency_path, index=False)
    cleaning_result.severity.to_csv(severity_path, index=False)

    if ingestion_report is not None:
        _write_json_report(ingestion_report_path, asdict(ingestion_report))
    _write_json_report(cleaning_report_path, asdict(cleaning_result.report))

    return ProcessedDataPaths(
        frequency_path=frequency_path,
        severity_path=severity_path,
        ingestion_report_path=ingestion_report_path,
        cleaning_report_path=cleaning_report_path,
    )


def run_cleaning_pipeline(
    frequency_path: Path | str = DEFAULT_FREQUENCY_PATH,
    severity_path: Path | str = DEFAULT_SEVERITY_PATH,
    output_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> ProcessedDataPaths:
    ingested = load_raw_data(frequency_path, severity_path)
    cleaned = clean_raw_data(ingested.frequency, ingested.severity)
    return write_cleaned_data(cleaned, output_dir, ingestion_report=ingested.report)


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean raw freMTPL2 CSVs and write processed outputs."
    )
    parser.add_argument(
        "--frequency-path",
        type=Path,
        default=DEFAULT_FREQUENCY_PATH,
        help=f"Frequency CSV path. Defaults to {DEFAULT_FREQUENCY_PATH}.",
    )
    parser.add_argument(
        "--severity-path",
        type=Path,
        default=DEFAULT_SEVERITY_PATH,
        help=f"Severity CSV path. Defaults to {DEFAULT_SEVERITY_PATH}.",
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
    paths = run_cleaning_pipeline(args.frequency_path, args.severity_path, args.output_dir)
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
