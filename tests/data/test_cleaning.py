from __future__ import annotations

import pandas as pd

from src.data.cleaning import CLAIMNB_CAP, EXPOSURE_CAP, clean_raw_data


def test_clean_raw_data_resolves_documented_quality_issues() -> None:
    frequency = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [0, 1, 10],
            "Exposure": [0.4, 1.25, 2.0],
            "VehPower": [5, 6, 7],
            "VehAge": [2, 3, 4],
            "DrivAge": [35, 48, 52],
            "BonusMalus": [50, 60, 70],
            "VehBrand": ["B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular"],
            "Area": ["A", "B", "C"],
            "Density": [100, 200, 300],
            "Region": ["R1", "R2", "R3"],
        }
    )
    severity = pd.DataFrame(
        {
            "IDpol": [2, 3, 3, 3, 3, 3, 999],
            "ClaimAmount": [100.0, 10.0, 20.0, 30.0, 40.0, 50.0, 9999.0],
        }
    )

    result = clean_raw_data(frequency, severity)
    cleaned_frequency = result.frequency.sort_values("IDpol").reset_index(drop=True)

    assert result.severity["IDpol"].tolist() == [2, 3, 3, 3, 3, 3]
    assert cleaned_frequency["ClaimNb_declared"].tolist() == [0, 1, 10]
    assert cleaned_frequency["ClaimNb"].tolist() == [0, 1, CLAIMNB_CAP]
    assert cleaned_frequency["Exposure"].tolist() == [0.4, EXPOSURE_CAP, EXPOSURE_CAP]

    assert result.report.frequency_rows_before == 3
    assert result.report.frequency_rows_after == 3
    assert result.report.severity_rows_before == 7
    assert result.report.severity_rows_after == 6
    assert result.report.unmatched_severity_rows_dropped == 1
    assert result.report.claimnb_mismatched_policies == 1
    assert result.report.exposure_values_capped == 2
    assert result.report.claimnb_values_capped == 1
    assert result.report.exposure_cap == EXPOSURE_CAP
    assert result.report.claimnb_cap == CLAIMNB_CAP
    assert result.report.categorical_columns_cast == ("VehBrand", "VehGas", "Area", "Region")

    for column in result.report.categorical_columns_cast:
        assert str(result.frequency[column].dtype) == "category"


def test_clean_raw_data_reconciles_claimnb_to_zero_for_policy_without_observed_claims() -> None:
    frequency = pd.DataFrame(
        {
            "IDpol": [1],
            "ClaimNb": [2],
            "Exposure": [0.7],
            "VehBrand": ["B1"],
            "VehGas": ["Regular"],
            "Area": ["A"],
            "Region": ["R1"],
        }
    )
    severity = pd.DataFrame({"IDpol": [999], "ClaimAmount": [12.0]})

    result = clean_raw_data(frequency, severity)

    assert result.frequency.loc[0, "ClaimNb_declared"] == 2
    assert result.frequency.loc[0, "ClaimNb"] == 0
    assert result.severity.empty
    assert result.report.unmatched_severity_rows_dropped == 1
    assert result.report.claimnb_mismatched_policies == 1
