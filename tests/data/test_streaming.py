from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.streaming import CONCEPT_DRIFT_PATTERN
from src.data.streaming import DATA_DRIFT_PATTERN
from src.data.streaming import DEFAULT_CONCEPT_DRIFT_START_BATCH
from src.data.streaming import DEFAULT_DATA_DRIFT_START_BATCH
from src.data.streaming import DEFAULT_N_BATCHES
from src.data.streaming import OVERLAP_NOTE
from src.data.streaming import PHASE_8_THRESHOLD_NOTE
from src.data.streaming import SYNTHETIC_SEVERITY_STRATEGY
from src.data.streaming import StreamSimulationError
from src.data.streaming import run_stream_simulation_pipeline
from src.data.streaming import simulate_stream
from src.data.streaming import write_stream_batches


def test_simulate_stream_joins_shuffles_and_batches_deterministically() -> None:
    features, targets = _stream_inputs()

    first = simulate_stream(features, targets)
    second = simulate_stream(features, targets)

    first_stream = pd.concat(first.batches, ignore_index=True)
    second_stream = pd.concat(second.batches, ignore_index=True)
    assert first_stream["IDpol"].tolist() == second_stream["IDpol"].tolist()
    assert sorted(first_stream["batch_id"].unique().tolist()) == list(range(DEFAULT_N_BATCHES))
    assert first.report.batch_size_min == 2
    assert first.report.batch_size_max == 2
    assert first.report.batch_size_mean == 2.0
    assert len(first_stream) == len(features)


def test_simulate_stream_injects_default_abrupt_data_and_concept_drift() -> None:
    features, targets = _stream_inputs()

    result = simulate_stream(features, targets)
    stream = pd.concat(result.batches, ignore_index=True)

    assert not stream.loc[stream["batch_id"] < DEFAULT_DATA_DRIFT_START_BATCH, "data_drift_injected"].any()
    assert stream.loc[stream["batch_id"] >= DEFAULT_DATA_DRIFT_START_BATCH, "data_drift_injected"].all()
    assert not stream.loc[
        stream["batch_id"] < DEFAULT_CONCEPT_DRIFT_START_BATCH, "concept_drift_injected"
    ].any()
    assert stream.loc[
        stream["batch_id"] >= DEFAULT_CONCEPT_DRIFT_START_BATCH, "concept_drift_injected"
    ].all()
    assert result.report.data_drift_pattern == DATA_DRIFT_PATTERN
    assert result.report.concept_drift_pattern == CONCEPT_DRIFT_PATTERN
    assert DATA_DRIFT_PATTERN == "abrupt_step_function"
    assert CONCEPT_DRIFT_PATTERN == "abrupt_step_function"


def test_simulate_stream_updates_density_fields_for_data_drift() -> None:
    features, targets = _stream_inputs()

    result = simulate_stream(features, targets)
    stream = pd.concat(result.batches, ignore_index=True)
    drifted = stream[stream["data_drift_injected"]]

    assert drifted["Density"].eq(5400.0).all()
    assert np.allclose(drifted["Density_log1p"], np.log1p(5400.0))
    assert drifted["Density_band"].eq("5001+").all()


def test_simulate_stream_accepts_integer_density_before_data_drift() -> None:
    features, targets = _stream_inputs()
    features["Density"] = features["Density"].astype("int64")

    result = simulate_stream(features, targets)
    stream = pd.concat(result.batches, ignore_index=True)

    assert stream.loc[stream["data_drift_injected"], "Density"].eq(5400.0).all()


def test_simulate_stream_updates_concept_targets_and_synthetic_severity() -> None:
    features, targets = _stream_inputs()

    result = simulate_stream(features, targets)
    stream = pd.concat(result.batches, ignore_index=True)
    drifted = stream[stream["concept_drift_injected"]]

    assert drifted["target_claim_count"].eq(2).all()
    assert drifted["target_has_claim"].eq(1).all()
    assert drifted["target_claim_frequency"].eq(0.0).all()
    assert (drifted["target_total_claim_amount"] > drifted["target_avg_claim_amount"]).all()
    assert set(
        drifted["target_total_claim_amount"] - targets.set_index("IDpol").loc[
            drifted["IDpol"], "target_total_claim_amount"
        ].to_numpy()
    ).issubset({100.0, 200.0, 300.0})
    assert result.report.zero_exposure_rows_during_concept_recompute == 10
    assert result.report.synthetic_severity_fallback_used is False
    assert result.report.total_synthetic_severity_added > 0
    assert "population-wide distribution" in result.report.synthetic_severity_strategy
    assert result.report.synthetic_severity_strategy == SYNTHETIC_SEVERITY_STRATEGY


def test_simulate_stream_logs_intentional_overlap_ground_truth() -> None:
    result = simulate_stream(*_stream_inputs())
    truth = result.drift_ground_truth

    assert result.report.overlap_intentional is True
    assert result.report.overlap_batch_start == 15
    assert result.report.overlap_batch_end == 19
    assert result.report.overlap_note == OVERLAP_NOTE
    assert truth["overlap"]["intentional"] is True
    assert truth["overlap"]["batch_start"] == 15
    assert truth["overlap"]["batch_end"] == 19
    assert truth["overlap"]["affected_rows_by_batch"][14] == 0
    assert truth["overlap"]["affected_rows_by_batch"][15] == 2
    assert truth["overlap"]["total_affected_rows"] == 10
    assert truth["phase_8_threshold_note"] == PHASE_8_THRESHOLD_NOTE


def test_simulate_stream_uses_separate_reproducible_severity_rng() -> None:
    features, targets = _stream_inputs()

    first = simulate_stream(features, targets, random_seed=42, severity_seed=7)
    second = simulate_stream(features, targets, random_seed=42, severity_seed=7)
    third = simulate_stream(features, targets, random_seed=42, severity_seed=8)

    first_stream = pd.concat(first.batches, ignore_index=True)
    second_stream = pd.concat(second.batches, ignore_index=True)
    third_stream = pd.concat(third.batches, ignore_index=True)
    first_added = first_stream.loc[
        first_stream["concept_drift_injected"], "target_total_claim_amount"
    ].tolist()
    second_added = second_stream.loc[
        second_stream["concept_drift_injected"], "target_total_claim_amount"
    ].tolist()
    third_added = third_stream.loc[
        third_stream["concept_drift_injected"], "target_total_claim_amount"
    ].tolist()

    assert first_added == second_added
    assert first_added != third_added
    assert first.report.random_seed == 42
    assert first.report.severity_seed == 7


def test_simulate_stream_uses_documented_severity_fallback_when_needed() -> None:
    features, targets = _stream_inputs()
    targets["target_avg_claim_amount"] = 0.0
    targets["target_total_claim_amount"] = 0.0

    result = simulate_stream(features, targets)
    stream = pd.concat(result.batches, ignore_index=True)
    drifted = stream[stream["concept_drift_injected"]]

    assert result.report.synthetic_severity_fallback_used is True
    assert drifted["target_total_claim_amount"].eq(1000.0).all()
    assert drifted["target_avg_claim_amount"].eq(500.0).all()
    assert result.drift_ground_truth["synthetic_severity"]["fallback_used"] is True


def test_simulate_stream_raises_for_missing_required_column() -> None:
    features, targets = _stream_inputs()

    with pytest.raises(StreamSimulationError, match="Density_band"):
        simulate_stream(features.drop(columns=["Density_band"]), targets)


def test_write_stream_batches_persists_batches_and_reports(tmp_path: Path) -> None:
    result = simulate_stream(*_stream_inputs())

    paths = write_stream_batches(result, tmp_path)

    assert len(paths.batch_paths) == DEFAULT_N_BATCHES
    assert all(path.exists() for path in paths.batch_paths)
    assert paths.report_path.exists()
    assert paths.drift_ground_truth_path.exists()

    written_batch = pd.read_csv(paths.batch_paths[15])
    written_truth = json.loads(paths.drift_ground_truth_path.read_text(encoding="utf-8"))
    assert {"batch_id", "event_index", "data_drift_injected"}.issubset(written_batch.columns)
    assert written_truth["data_drift"]["pattern"] == DATA_DRIFT_PATTERN
    assert written_truth["concept_drift"]["pattern"] == CONCEPT_DRIFT_PATTERN
    assert written_truth["overlap"]["batch_start"] == 15


def test_run_stream_simulation_pipeline_reads_inputs_and_writes_outputs(
    tmp_path: Path,
) -> None:
    features, targets = _stream_inputs()
    features_path = tmp_path / "contract_features.csv"
    targets_path = tmp_path / "contract_targets.csv"
    output_dir = tmp_path / "stream"
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path, index=False)

    paths = run_stream_simulation_pipeline(
        contract_features_path=features_path,
        contract_targets_path=targets_path,
        output_dir=output_dir,
    )

    assert paths.batch_paths[0] == output_dir / "batch_000.csv"
    assert paths.batch_paths[-1] == output_dir / "batch_019.csv"
    assert paths.report_path == output_dir / "stream_simulation_report.json"
    assert paths.drift_ground_truth_path == output_dir / "drift_ground_truth.json"
    assert paths.report_path.exists()


def _stream_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    idpol = list(range(1, 41))
    features = pd.DataFrame(
        {
            "IDpol": idpol,
            "pool_id": [index % 4 for index in idpol],
            "Exposure": [0.0] * 40,
            "VehPower": [5] * 40,
            "VehAge": [4] * 40,
            "DrivAge": [40] * 40,
            "BonusMalus": [100] * 40,
            "Density": [4000.0] * 40,
            "Density_log1p": [float(np.log1p(4000.0))] * 40,
            "VehBrand": ["B1"] * 40,
            "VehGas": ["Regular"] * 40,
            "Area": ["A"] * 40,
            "Region": ["R1"] * 40,
            "VehAge_band": ["3-5"] * 40,
            "DrivAge_band": ["36-50"] * 40,
            "BonusMalus_band": ["76-100"] * 40,
            "Density_band": ["1501-5000"] * 40,
        }
    )
    targets = pd.DataFrame(
        {
            "IDpol": idpol,
            "target_claim_count": [1] * 40,
            "target_has_claim": [1] * 40,
            "target_claim_frequency": [0.0] * 40,
            "target_total_claim_amount": [100.0, 200.0, 300.0, 0.0] * 10,
            "target_avg_claim_amount": [100.0, 200.0, 300.0, 0.0] * 10,
        }
    )
    return features, targets
