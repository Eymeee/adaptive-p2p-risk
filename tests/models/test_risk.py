from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.risk import ANOMALY_ENCODING_NOTE
from src.models.risk import CATEGORICAL_FEATURE_COLUMNS
from src.models.risk import MODEL_INPUT_COLUMNS
from src.models.risk import NUMERIC_FEATURE_COLUMNS
from src.models.risk import POOL_SCORE_CAVEAT_NOTE
from src.models.risk import RISK_REPORT_FILENAME
from src.models.risk import TARGET_CHOICE_NOTE
from src.models.risk import build_risk_models
from src.models.risk import run_risk_modeling_pipeline
from src.models.risk import write_risk_modeling_artifacts


def test_build_risk_models_uses_real_calibration_path_with_enough_positives() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(
        contract_features,
        contract_targets,
        pool_features,
        pool_targets,
        dataset_version_report=_dataset_version_report(),
    )

    probabilities = result.contract_test_predictions["predicted_claim_probability"]
    assert probabilities.between(0.0, 1.0).all()
    assert result.report.positive_rows == 25
    assert result.report.train_positive_rows == 20
    assert result.report.test_positive_rows == 5
    assert result.report.calibration_skipped is False
    assert result.report.calibration_cv_folds == 5


def test_risk_report_documents_target_choice_and_calibration_without_class_weighting() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(
        contract_features,
        contract_targets,
        pool_features,
        pool_targets,
    )

    assert result.report.target_column == "target_has_claim"
    assert result.report.target_choice_note == TARGET_CHOICE_NOTE
    assert result.report.classifier_class_weight is None
    assert result.report.test_empirical_claim_rate == 0.05
    assert result.report.test_mean_predicted_probability >= 0.0
    assert result.report.test_probability_rate_delta >= 0.0


def test_risk_model_inputs_exclude_target_and_leakage_columns() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    forbidden = set(result.report.leakage_excluded_columns)
    assert set(result.report.model_input_columns) == set(MODEL_INPUT_COLUMNS)
    assert forbidden.isdisjoint(result.report.model_input_columns)
    assert "target_has_claim" not in result.report.model_input_columns
    assert "target_claim_frequency" not in result.report.model_input_columns


def test_report_tracks_auc_gini_brier_ece_and_baseline_metrics() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    assert 0.0 <= result.report.model_auc <= 1.0
    assert result.report.baseline_auc == 0.5
    assert result.report.model_normalized_gini == 2 * result.report.model_auc - 1
    assert result.report.baseline_normalized_gini == 0.0
    assert result.report.brier_score >= 0.0
    assert result.report.expected_calibration_error >= 0.0
    assert result.report.calibration_bin_count == 10
    assert result.report.acceptance_passed is True


def test_pool_risk_scores_cover_every_pool_and_document_all_contract_scoring_caveat() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    assert set(result.pool_risk_scores["pool_id"]) == set(pool_features["pool_id"])
    assert result.report.pool_score_coverage == 1.0
    assert result.report.pool_score_rows == len(pool_features)
    assert result.report.pool_score_caveat_note == POOL_SCORE_CAVEAT_NOTE


def test_zero_exposure_pool_uses_unweighted_fallback_and_is_reported() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()
    contract_features.loc[contract_features["pool_id"] == "3", "Exposure"] = 0.0

    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    zero_pool = result.pool_risk_scores.set_index("pool_id").loc["3"]
    assert zero_pool["pool_score_weighting_method"] == "unweighted_zero_exposure_fallback"
    assert result.report.zero_exposure_pools == 1


def test_isolation_forest_uses_scaled_numeric_and_one_hot_categorical_inputs() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    assert result.report.anomaly_encoding_note == ANOMALY_ENCODING_NOTE
    assert result.report.numeric_feature_columns == NUMERIC_FEATURE_COLUMNS
    assert result.report.categorical_feature_columns == CATEGORICAL_FEATURE_COLUMNS
    assert result.report.anomaly_encoded_feature_count > len(NUMERIC_FEATURE_COLUMNS)
    assert len(result.anomaly_scores) == len(contract_features)
    assert {"IDpol", "pool_id", "anomaly_score", "anomaly_flag"}.issubset(
        result.anomaly_scores.columns
    )


def test_dataset_version_metadata_is_copied_into_report() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()

    result = build_risk_models(
        contract_features,
        contract_targets,
        pool_features,
        pool_targets,
        dataset_version_report=_dataset_version_report(),
    )

    assert result.report.dvc_output_hash == "dataset-hash.dir"
    assert result.report.dvc_tracked_path == "data/processed"
    assert result.report.dataset_git_commit_sha == "abc123"
    assert "MLflow tag" in result.report.traceability_note


def test_calibration_skips_only_when_training_class_support_is_too_small() -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data(
        n_rows=8,
        n_positive=2,
        n_pools=2,
    )

    result = build_risk_models(
        contract_features,
        contract_targets,
        pool_features,
        pool_targets,
        test_size=0.5,
    )

    assert result.report.train_positive_rows == 1
    assert result.report.calibration_skipped is True
    assert result.report.calibration_cv_folds is None


def test_write_risk_modeling_artifacts_serializes_outputs(tmp_path: Path) -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()
    result = build_risk_models(contract_features, contract_targets, pool_features, pool_targets)

    paths = write_risk_modeling_artifacts(result, tmp_path)

    assert paths.model_artifact_path.exists()
    assert paths.contract_test_predictions_path.exists()
    assert paths.all_contract_predictions_path.exists()
    assert paths.pool_risk_scores_path.exists()
    assert paths.anomaly_scores_path.exists()
    with paths.model_artifact_path.open("rb") as artifact_file:
        artifact_payload = pickle.load(artifact_file)
    assert isinstance(artifact_payload["report"], dict)
    assert "feature_preprocessor" in artifact_payload
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    assert report["target_choice_note"] == TARGET_CHOICE_NOTE


def test_run_risk_modeling_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
    contract_features, contract_targets, pool_features, pool_targets = _synthetic_model_data()
    contract_features_path = tmp_path / "contract_features.csv"
    contract_targets_path = tmp_path / "contract_targets.csv"
    pool_features_path = tmp_path / "pool_features.csv"
    pool_targets_path = tmp_path / "pool_targets.csv"
    dataset_version_report_path = tmp_path / "dataset_version_report.json"
    output_dir = tmp_path / "artifacts"
    contract_features.to_csv(contract_features_path, index=False)
    contract_targets.to_csv(contract_targets_path, index=False)
    pool_features.to_csv(pool_features_path, index=False)
    pool_targets.to_csv(pool_targets_path, index=False)
    dataset_version_report_path.write_text(
        json.dumps(_dataset_version_report()),
        encoding="utf-8",
    )

    paths = run_risk_modeling_pipeline(
        contract_features_path=contract_features_path,
        contract_targets_path=contract_targets_path,
        pool_features_path=pool_features_path,
        pool_targets_path=pool_targets_path,
        dataset_version_report_path=dataset_version_report_path,
        output_dir=output_dir,
        enforce_acceptance=False,
    )

    assert paths.report_path == output_dir / RISK_REPORT_FILENAME
    report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    assert report["dvc_output_hash"] == "dataset-hash.dir"


def _synthetic_model_data(
    n_rows: int = 500,
    n_positive: int = 25,
    n_pools: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    ids = np.arange(1, n_rows + 1)
    pool_ids = (ids % n_pools).astype(str)
    veh_power = rng.integers(4, 12, size=n_rows)
    veh_age = rng.integers(0, 16, size=n_rows)
    driv_age = rng.integers(19, 75, size=n_rows)
    bonus_malus = rng.integers(45, 165, size=n_rows)
    exposure = rng.uniform(0.2, 1.0, size=n_rows)
    density = rng.integers(50, 4000, size=n_rows)
    risk_signal = (
        0.045 * bonus_malus
        + 0.08 * veh_power
        - 0.018 * driv_age
        + 0.0004 * density
        + rng.normal(0.0, 0.2, size=n_rows)
    )
    positive_indices = np.argsort(risk_signal)[-n_positive:]
    target_has_claim = np.zeros(n_rows, dtype=int)
    target_has_claim[positive_indices] = 1

    contract_features = pd.DataFrame(
        {
            "IDpol": ids,
            "pool_id": pool_ids,
            "Exposure": exposure,
            "VehPower": veh_power,
            "VehAge": veh_age,
            "DrivAge": driv_age,
            "BonusMalus": bonus_malus,
            "Density": density,
            "Density_log1p": np.log1p(density),
            "VehBrand": np.where(veh_power > 8, "B2", "B1"),
            "VehGas": np.where(ids % 2 == 0, "Diesel", "Regular"),
            "Area": np.where(density > 1200, "E", "C"),
            "Region": np.where(ids % 3 == 0, "R1", "R2"),
            "VehAge_band": np.where(veh_age <= 5, "0-5", "6+"),
            "DrivAge_band": np.where(driv_age <= 35, "0-35", "36+"),
            "BonusMalus_band": np.where(bonus_malus <= 100, "0-100", "101+"),
            "Density_band": np.where(density <= 1500, "0-1500", "1501+"),
        }
    )
    contract_targets = pd.DataFrame(
        {
            "IDpol": ids,
            "target_claim_count": target_has_claim,
            "target_has_claim": target_has_claim,
            "target_claim_frequency": np.where(exposure > 0, target_has_claim / exposure, 0.0),
            "target_total_claim_amount": target_has_claim * 1000.0,
            "target_avg_claim_amount": target_has_claim * 1000.0,
        }
    )
    pool_targets = (
        pd.DataFrame(
            {
                "pool_id": pool_ids,
                "target_has_claim": target_has_claim,
                "Exposure": exposure,
            }
        )
        .groupby("pool_id", as_index=False)
        .agg(
            pool_claim_count=("target_has_claim", "sum"),
            pool_total_exposure=("Exposure", "sum"),
        )
    )
    pool_targets["pool_has_claim"] = (pool_targets["pool_claim_count"] > 0).astype(int)
    pool_targets["pool_claim_rate"] = (
        pool_targets["pool_claim_count"] / pool_targets["pool_total_exposure"]
    )
    pool_targets["pool_total_claim_amount"] = pool_targets["pool_claim_count"] * 1000.0
    pool_targets["pool_avg_claim_amount"] = np.where(
        pool_targets["pool_claim_count"] > 0,
        pool_targets["pool_total_claim_amount"] / pool_targets["pool_claim_count"],
        0.0,
    )
    pool_features = pd.DataFrame(
        {
            "pool_id": sorted(set(pool_ids)),
            "pool_size": [int((pool_ids == pool_id).sum()) for pool_id in sorted(set(pool_ids))],
        }
    )
    return contract_features, contract_targets, pool_features, pool_targets


def _dataset_version_report() -> dict[str, str]:
    return {
        "dvc_output_hash": "dataset-hash.dir",
        "dvc_tracked_path": "data/processed",
        "git_commit_sha": "abc123",
    }
