from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FREQUENCY_COLUMNS: tuple[str, ...] = (
    "IDpol",
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
DEFAULT_FREQUENCY_PATH = Path("data/raw/freMTPL2freq.csv")
DEFAULT_SEVERITY_PATH = Path("data/raw/freMTPL2sev.csv")


class DataIngestionError(ValueError):
    """Raised when raw freMTPL2 files cannot be loaded into a valid dataset."""


@dataclass(frozen=True)
class IngestionReport:
    frequency_path: Path
    severity_path: Path
    frequency_rows: int
    severity_rows: int
    frequency_columns: tuple[str, ...]
    severity_columns: tuple[str, ...]
    duplicate_frequency_idpol_count: int


@dataclass(frozen=True)
class IngestionResult:
    frequency: pd.DataFrame
    severity: pd.DataFrame
    report: IngestionReport


def load_raw_data(frequency_path: Path | str, severity_path: Path | str) -> IngestionResult:
    """Load and validate the two freMTPL2 raw tables before any cleaning decisions."""
    frequency_file = Path(frequency_path)
    severity_file = Path(severity_path)

    frequency = _read_csv(frequency_file, "frequency")
    severity = _read_csv(severity_file, "severity")

    duplicate_count = _validate_frequency(frequency)
    _validate_severity(severity)

    report = IngestionReport(
        frequency_path=frequency_file,
        severity_path=severity_file,
        frequency_rows=len(frequency),
        severity_rows=len(severity),
        frequency_columns=tuple(frequency.columns),
        severity_columns=tuple(severity.columns),
        duplicate_frequency_idpol_count=duplicate_count,
    )
    return IngestionResult(frequency=frequency, severity=severity, report=report)


def load_default_raw_data() -> IngestionResult:
    """Load the project-standard raw freMTPL2 files from data/raw."""
    return load_raw_data(DEFAULT_FREQUENCY_PATH, DEFAULT_SEVERITY_PATH)


def _read_csv(path: Path, table_name: str) -> pd.DataFrame:
    if not path.exists():
        raise DataIngestionError(f"{table_name} CSV does not exist: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataIngestionError(f"{table_name} CSV is empty: {path}") from exc


def _validate_frequency(frequency: pd.DataFrame) -> int:
    _validate_required_columns(frequency, FREQUENCY_COLUMNS, "frequency")
    _validate_non_empty(frequency, "frequency")

    duplicate_count = int(frequency["IDpol"].duplicated().sum())
    if duplicate_count > 0:
        raise DataIngestionError(
            f"frequency CSV contains {duplicate_count} duplicate IDpol values"
        )
    return duplicate_count


def _validate_severity(severity: pd.DataFrame) -> None:
    _validate_required_columns(severity, SEVERITY_COLUMNS, "severity")
    _validate_non_empty(severity, "severity")


def _validate_required_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], table_name: str
) -> None:
    missing_columns = tuple(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise DataIngestionError(
            f"{table_name} CSV is missing required columns: {', '.join(missing_columns)}"
        )


def _validate_non_empty(frame: pd.DataFrame, table_name: str) -> None:
    if frame.empty:
        raise DataIngestionError(f"{table_name} CSV has headers but no rows")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and validate the raw freMTPL2 frequency and severity CSVs."
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
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = load_raw_data(args.frequency_path, args.severity_path)
    print(json.dumps(asdict(result.report), indent=2, default=str))


if __name__ == "__main__":
    main()
