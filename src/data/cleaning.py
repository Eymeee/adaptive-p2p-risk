from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CATEGORICAL_COLUMNS: tuple[str, ...] = ("VehBrand", "VehGas", "Area", "Region")
EXPOSURE_CAP = 1.0
CLAIMNB_CAP = 4


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
