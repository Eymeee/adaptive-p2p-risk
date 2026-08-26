from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.features import CLAIM_RATE_RECLASSIFICATION_NOTE
from src.data.features import FeatureEngineeringError
from src.data.features import build_features
from src.data.features import run_feature_engineering_pipeline
from src.data.features import write_feature_data


def test_build_features_creates_contract_and_pool_tables() -> None:
    result = build_features(_frequency_frame(), _severity_frame())

    assert len(result.contract_features) == 5
    assert len(result.contract_targets) == 5
    assert len(result.pool_features) == 3
    assert len(result.pool_targets) == 3
    assert result.report.contract_feature_rows == 5
    assert result.report.contract_target_rows == 5
    assert result.report.pool_feature_rows == 3
    assert result.report.pool_target_rows == 3
    assert result.report.number_of_pools == 3
    assert result.report.severity_input_used is True


def test_build_features_keeps_targets_out_of_feature_tables() -> None:
    result = build_features(_frequency_frame(), _severity_frame())

    forbidden_contract_columns = {
        "ClaimNb",
        "ClaimNb_declared",
        "ClaimAmount",
        "target_claim_count",
        "target_claim_frequency",
    }
    forbidden_pool_columns = {
        "pool_claim_count",
        "pool_claim_rate",
        "pool_total_claim_amount",
        "pool_avg_claim_amount",
    }

    assert forbidden_contract_columns.isdisjoint(result.contract_features.columns)
    assert forbidden_pool_columns.isdisjoint(result.pool_features.columns)
    assert {"target_claim_count", "target_claim_frequency"}.issubset(
        result.contract_targets.columns
    )
    assert {"pool_claim_count", "pool_claim_rate"}.issubset(result.pool_targets.columns)


def test_build_features_computes_contract_targets_and_severity_aggregates() -> None:
    result = build_features(_frequency_frame(), _severity_frame())
    targets = result.contract_targets.set_index("IDpol")

    assert targets.loc[1, "target_claim_count"] == 0
    assert targets.loc[2, "target_has_claim"] == 1
    assert targets.loc[2, "target_claim_frequency"] == 2.5
    assert targets.loc[2, "target_total_claim_amount"] == 300.0
    assert targets.loc[2, "target_avg_claim_amount"] == 150.0
    assert targets.loc[3, "target_claim_frequency"] == 0.0
    assert targets.loc[3, "target_total_claim_amount"] == 0.0


def test_build_features_computes_pool_aggregates_and_zero_exposure_guard() -> None:
    result = build_features(_frequency_frame(), _severity_frame())
    pool_features = result.pool_features.set_index("pool_id")
    pool_targets = result.pool_targets.set_index("pool_id")

    assert pool_features.loc[10, "pool_size"] == 2
    assert pool_features.loc[10, "pool_total_exposure"] == pytest.approx(1.2)
    assert pool_features.loc[10, "pool_veh_brand_diversity"] == 2
    assert pool_targets.loc[10, "pool_claim_count"] == 2
    assert pool_targets.loc[10, "pool_claim_rate"] == pytest.approx(2 / 1.2)
    assert pool_targets.loc[10, "pool_total_claim_amount"] == 300.0
    assert pool_targets.loc[20, "pool_claim_rate"] == 0.0
    assert pool_targets.loc[20, "pool_total_claim_amount"] == 0.0
    assert pool_targets.loc[30, "pool_total_claim_amount"] == 50.0
    assert result.report.zero_exposure_contract_rows == 2
    assert result.report.zero_exposure_pools == 1


def test_build_features_uses_fixed_bands_and_logs_edges() -> None:
    result = build_features(_frequency_frame(), _severity_frame())
    features = result.contract_features.set_index("IDpol")

    assert features.loc[1, "VehAge_band"] == "0-2"
    assert features.loc[2, "VehAge_band"] == "3-5"
    assert features.loc[5, "VehAge_band"] == "11+"
    assert features.loc[1, "DrivAge_band"] == "0-25"
    assert features.loc[3, "DrivAge_band"] == "36-50"
    assert features.loc[5, "Density_band"] == "5001+"

    assert result.report.banding_methodology.startswith("Fixed domain thresholds")
    assert result.report.band_definitions["VehAge_band"].edges == ("0", "2", "5", "10", "inf")
    assert result.report.band_definitions["Density_band"].labels[-1] == "5001+"


def test_build_features_reports_claim_rate_reclassification_note() -> None:
    result = build_features(_frequency_frame(), _severity_frame())

    assert result.report.claim_rate_reclassification_note == CLAIM_RATE_RECLASSIFICATION_NOTE
    assert "CdCT lists average claim rate as a pool feature" in (
        result.report.claim_rate_reclassification_note
    )
    assert "leak" in result.report.claim_rate_reclassification_note


def test_build_features_allows_missing_optional_severity() -> None:
    result = build_features(_frequency_frame())

    assert result.report.severity_input_used is False
    assert result.report.severity_input_rows is None
    assert result.contract_targets["target_total_claim_amount"].tolist() == [0.0] * 5
    assert result.pool_targets["pool_total_claim_amount"].tolist() == [0.0, 0.0, 0.0]


def test_build_features_raises_for_missing_required_column() -> None:
    frequency = _frequency_frame().drop(columns=["pool_id"])

    with pytest.raises(FeatureEngineeringError, match="pool_id"):
        build_features(frequency, _severity_frame())


def test_write_feature_data_persists_all_outputs(tmp_path: Path) -> None:
    result = build_features(_frequency_frame(), _severity_frame())

    paths = write_feature_data(result, tmp_path)

    assert paths.contract_features_path.exists()
    assert paths.contract_targets_path.exists()
    assert paths.pool_features_path.exists()
    assert paths.pool_targets_path.exists()
    assert paths.report_path.exists()

    written_report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    written_contract_features = pd.read_csv(paths.contract_features_path)
    assert "Density_log1p" in written_contract_features.columns
    assert written_report["zero_exposure_pools"] == 1
    assert written_report["claim_rate_reclassification_note"] == CLAIM_RATE_RECLASSIFICATION_NOTE


def test_run_feature_engineering_pipeline_reads_inputs_and_writes_outputs(
    tmp_path: Path,
) -> None:
    frequency_path = tmp_path / "freMTPL2freq_pooled.csv"
    severity_path = tmp_path / "freMTPL2sev_cleaned.csv"
    output_dir = tmp_path / "processed"
    _frequency_frame().to_csv(frequency_path, index=False)
    _severity_frame().to_csv(severity_path, index=False)

    paths = run_feature_engineering_pipeline(
        frequency_path=frequency_path,
        severity_path=severity_path,
        output_dir=output_dir,
    )

    assert paths.contract_features_path == output_dir / "contract_features.csv"
    assert paths.contract_targets_path == output_dir / "contract_targets.csv"
    assert paths.pool_features_path == output_dir / "pool_features.csv"
    assert paths.pool_targets_path == output_dir / "pool_targets.csv"
    assert paths.report_path == output_dir / "feature_engineering_report.json"
    assert paths.report_path.exists()


def _frequency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4, 5],
            "pool_id": [10, 10, 20, 20, 30],
            "ClaimNb": [0, 2, 1, 0, 3],
            "ClaimNb_declared": [0, 2, 1, 0, 4],
            "Exposure": [0.4, 0.8, 0.0, 0.0, 1.0],
            "VehPower": [4, 6, 8, 5, 9],
            "VehAge": [1, 5, 8, 10, 12],
            "DrivAge": [22, 35, 50, 65, 70],
            "BonusMalus": [50, 75, 100, 150, 180],
            "VehBrand": ["B1", "B2", "B1", "B1", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular", "Regular", "Diesel"],
            "Area": ["A", "B", "A", "A", "C"],
            "Density": [50, 500, 1500, 5000, 7000],
            "Region": ["R1", "R2", "R1", "R1", "R3"],
        }
    )


def _severity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": [2, 2, 5],
            "ClaimAmount": [100.0, 200.0, 50.0],
        }
    )
