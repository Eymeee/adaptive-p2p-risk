from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import DataIngestionError, load_raw_data


def test_load_raw_data_returns_frames_and_report(tmp_path: Path) -> None:
    frequency_path = tmp_path / "freMTPL2freq.csv"
    severity_path = tmp_path / "freMTPL2sev.csv"
    _frequency_frame().to_csv(frequency_path, index=False)
    _severity_frame().to_csv(severity_path, index=False)

    result = load_raw_data(frequency_path, severity_path)

    assert len(result.frequency) == 2
    assert len(result.severity) == 2
    assert result.report.frequency_path == frequency_path
    assert result.report.severity_path == severity_path
    assert result.report.frequency_rows == 2
    assert result.report.severity_rows == 2
    assert result.report.duplicate_frequency_idpol_count == 0


def test_load_raw_data_raises_for_missing_file(tmp_path: Path) -> None:
    severity_path = tmp_path / "freMTPL2sev.csv"
    _severity_frame().to_csv(severity_path, index=False)

    with pytest.raises(DataIngestionError, match="frequency CSV does not exist"):
        load_raw_data(tmp_path / "missing.csv", severity_path)


def test_load_raw_data_raises_for_missing_required_column(tmp_path: Path) -> None:
    frequency_path = tmp_path / "freMTPL2freq.csv"
    severity_path = tmp_path / "freMTPL2sev.csv"
    _frequency_frame().drop(columns=["Exposure"]).to_csv(frequency_path, index=False)
    _severity_frame().to_csv(severity_path, index=False)

    with pytest.raises(DataIngestionError, match="missing required columns: Exposure"):
        load_raw_data(frequency_path, severity_path)


def test_load_raw_data_raises_for_header_only_csv(tmp_path: Path) -> None:
    frequency_path = tmp_path / "freMTPL2freq.csv"
    severity_path = tmp_path / "freMTPL2sev.csv"
    _frequency_frame().iloc[0:0].to_csv(frequency_path, index=False)
    _severity_frame().to_csv(severity_path, index=False)

    with pytest.raises(DataIngestionError, match="frequency CSV has headers but no rows"):
        load_raw_data(frequency_path, severity_path)


def test_load_raw_data_raises_for_duplicate_frequency_idpol(tmp_path: Path) -> None:
    frequency_path = tmp_path / "freMTPL2freq.csv"
    severity_path = tmp_path / "freMTPL2sev.csv"
    frequency = _frequency_frame()
    frequency.loc[1, "IDpol"] = frequency.loc[0, "IDpol"]
    frequency.to_csv(frequency_path, index=False)
    _severity_frame().to_csv(severity_path, index=False)

    with pytest.raises(DataIngestionError, match="duplicate IDpol"):
        load_raw_data(frequency_path, severity_path)


def _frequency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [0, 1],
            "Exposure": [0.5, 1.0],
            "VehPower": [5, 6],
            "VehAge": [2, 3],
            "DrivAge": [35, 48],
            "BonusMalus": [50, 60],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Regular", "Diesel"],
            "Area": ["A", "B"],
            "Density": [100, 200],
            "Region": ["R1", "R2"],
        }
    )


def _severity_frame() -> pd.DataFrame:
    return pd.DataFrame({"IDpol": [2, 2], "ClaimAmount": [1000.0, 500.0]})
