from __future__ import annotations

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
