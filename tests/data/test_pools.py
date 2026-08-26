from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.pools import ENCODING_STRATEGY
from src.data.pools import EXCLUDED_FEATURES
from src.data.pools import MIXED_TYPE_CLUSTERING_TRADEOFF
from src.data.pools import SCALING_STRATEGY
from src.data.pools import PoolConstructionError
from src.data.pools import construct_pools
from src.data.pools import run_pool_construction_pipeline
from src.data.pools import write_pooled_data


def test_construct_pools_assigns_one_pool_per_contract() -> None:
    frequency = _frequency_frame()

    result = construct_pools(
        frequency,
        k_min=2,
        k_max=4,
        random_seed=42,
        silhouette_sample_size=8,
        silhouette_sample_seed=7,
    )

    assert len(result.frequency) == len(frequency)
    assert result.frequency["pool_id"].notna().all()
    assert result.frequency["pool_id"].nunique() == result.report.selected_k
    assert result.report.input_rows == len(frequency)
    assert result.report.output_rows == len(frequency)
    assert 2 <= result.report.selected_k <= 4


def test_construct_pools_is_reproducible_for_same_seed() -> None:
    frequency = _frequency_frame()

    first = construct_pools(
        frequency,
        k_min=2,
        k_max=4,
        random_seed=42,
        silhouette_sample_size=10,
        silhouette_sample_seed=11,
    )
    second = construct_pools(
        frequency,
        k_min=2,
        k_max=4,
        random_seed=42,
        silhouette_sample_size=10,
        silhouette_sample_seed=11,
    )

    assert first.frequency["pool_id"].tolist() == second.frequency["pool_id"].tolist()
    assert first.report.selected_k == second.report.selected_k
    assert first.report.k_scan_results == second.report.k_scan_results


def test_construct_pools_records_full_methodology_report() -> None:
    result = construct_pools(
        _frequency_frame(),
        k_min=2,
        k_max=5,
        random_seed=42,
        silhouette_sample_size=6,
        silhouette_sample_seed=123,
    )

    assert [entry.k for entry in result.report.k_scan_results] == [2, 3, 4, 5]
    assert all(entry.inertia > 0 for entry in result.report.k_scan_results)
    assert result.report.silhouette_sample_size == 6
    assert result.report.silhouette_sample_seed == 123
    assert result.report.scaling_strategy == SCALING_STRATEGY
    assert result.report.encoding_strategy == ENCODING_STRATEGY
    assert result.report.excluded_features == EXCLUDED_FEATURES
    assert "ClaimNb" not in result.report.numeric_features
    assert "ClaimNb_declared" not in result.report.numeric_features
    assert "Exposure" not in result.report.numeric_features
    assert result.report.feature_rationale["VehBrand"]
    assert "High-cardinality categoricals" in result.report.mixed_type_clustering_tradeoff
    assert "20 binary columns" in result.report.mixed_type_clustering_tradeoff
    assert result.report.mixed_type_clustering_tradeoff == MIXED_TYPE_CLUSTERING_TRADEOFF


def test_construct_pools_raises_for_missing_required_feature() -> None:
    frequency = _frequency_frame().drop(columns=["BonusMalus"])

    with pytest.raises(PoolConstructionError, match="BonusMalus"):
        construct_pools(frequency, k_min=2, k_max=4)


def test_write_pooled_data_persists_csv_and_report(tmp_path: Path) -> None:
    result = construct_pools(
        _frequency_frame(),
        k_min=2,
        k_max=3,
        random_seed=42,
        silhouette_sample_size=8,
        silhouette_sample_seed=7,
    )

    paths = write_pooled_data(result, tmp_path)

    assert paths.frequency_path.exists()
    assert paths.report_path.exists()

    written_frequency = pd.read_csv(paths.frequency_path)
    written_report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    assert "pool_id" in written_frequency.columns
    assert written_report["selected_k"] == result.report.selected_k
    assert written_report["silhouette_sample_size"] == 8


def test_run_pool_construction_pipeline_reads_cleaned_frequency_and_writes_outputs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "freMTPL2freq_cleaned.csv"
    output_dir = tmp_path / "processed"
    _frequency_frame().to_csv(input_path, index=False)

    paths = run_pool_construction_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        k_min=2,
        k_max=3,
        random_seed=42,
        silhouette_sample_size=8,
        silhouette_sample_seed=7,
    )

    assert paths.frequency_path == output_dir / "freMTPL2freq_pooled.csv"
    assert paths.report_path == output_dir / "pool_construction_report.json"
    assert paths.frequency_path.exists()
    assert paths.report_path.exists()


def _frequency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": list(range(1, 19)),
            "ClaimNb": [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1],
            "ClaimNb_declared": [0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1],
            "Exposure": [0.5, 0.8, 1.0, 0.6, 0.7, 0.9, 0.4, 1.0, 0.5, 0.8, 0.9, 0.6, 0.7, 1.0, 0.4, 0.9, 0.5, 0.8],
            "VehPower": [4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9],
            "VehAge": [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 1, 4, 7, 10, 2, 5],
            "DrivAge": [22, 25, 31, 38, 45, 52, 24, 29, 35, 41, 48, 55, 23, 33, 43, 53, 28, 58],
            "BonusMalus": [50, 54, 60, 68, 76, 90, 51, 56, 63, 70, 82, 95, 52, 61, 72, 85, 58, 98],
            "VehBrand": ["B1", "B1", "B2", "B2", "B3", "B3", "B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3"],
            "VehGas": ["Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel", "Regular", "Diesel"],
            "Area": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "Density": [20, 40, 80, 120, 250, 500, 25, 60, 110, 180, 320, 640, 30, 90, 210, 420, 70, 700],
            "Region": ["R1", "R1", "R2", "R2", "R3", "R3", "R1", "R2", "R3", "R1", "R2", "R3", "R1", "R2", "R3", "R1", "R2", "R3"],
        }
    )
