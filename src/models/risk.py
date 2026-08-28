"""Risk modeling for Phase 7.

The contract model predicts `target_has_claim` because FR-RM-01 asks for claim
probability. `target_claim_frequency` is a frequency/regression target and is
kept out of the v1 classifier inputs to preserve the leakage boundary.
"""
# pyrefly: ignore-errors[missing-import]

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.features import DEFAULT_CONTRACT_FEATURES_FILENAME
from src.data.features import DEFAULT_CONTRACT_TARGETS_FILENAME
from src.data.features import DEFAULT_POOL_FEATURES_FILENAME
from src.data.features import DEFAULT_POOL_TARGETS_FILENAME
from src.data.features import LEAKAGE_EXCLUDED_COLUMNS
from src.data.versioning import DEFAULT_DATASET_VERSION_REPORT_FILENAME

DEFAULT_CONTRACT_FEATURES_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_FEATURES_FILENAME
DEFAULT_CONTRACT_TARGETS_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_CONTRACT_TARGETS_FILENAME
DEFAULT_POOL_FEATURES_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_POOL_FEATURES_FILENAME
DEFAULT_POOL_TARGETS_PATH = DEFAULT_PROCESSED_DIR / DEFAULT_POOL_TARGETS_FILENAME
DEFAULT_DATASET_VERSION_REPORT_PATH = (
    DEFAULT_PROCESSED_DIR / DEFAULT_DATASET_VERSION_REPORT_FILENAME
)
DEFAULT_ARTIFACT_DIR = Path("artifacts/phase7")

MODEL_ARTIFACT_FILENAME = "risk_model.pkl"
CONTRACT_TEST_PREDICTIONS_FILENAME = "contract_test_predictions.csv"
ALL_CONTRACT_PREDICTIONS_FILENAME = "contract_predictions.csv"
POOL_RISK_SCORES_FILENAME = "pool_risk_scores.csv"
ANOMALY_SCORES_FILENAME = "contract_anomaly_scores.csv"
RISK_REPORT_FILENAME = "risk_modeling_report.json"

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CALIBRATION_CV = 5
DEFAULT_ANOMALY_CONTAMINATION = 0.01
CALIBRATION_BINS = 10

ID_COLUMN = "IDpol"
POOL_COLUMN = "pool_id"
TARGET_COLUMN = "target_has_claim"
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "Exposure",
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "Density_log1p",
)
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "pool_id",
    "VehBrand",
    "VehGas",
    "Area",
    "Region",
    "VehAge_band",
    "DrivAge_band",
    "BonusMalus_band",
    "Density_band",
)
MODEL_INPUT_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
REQUIRED_CONTRACT_TARGET_COLUMNS: tuple[str, ...] = (ID_COLUMN, TARGET_COLUMN)
REQUIRED_POOL_TARGET_COLUMNS: tuple[str, ...] = (POOL_COLUMN, "pool_claim_rate")

TARGET_CHOICE_NOTE = (
    "target_has_claim is modeled because FR-RM-01 asks for claim probability; "
    "target_claim_frequency is a regression/frequency target and is deferred."
)
PERFORMANCE_INTERPRETABILITY_NOTE = (
    "Logistic regression is used for the v1 claim-probability model because it "
    "is deterministic, calibratable, and easier to interpret than less linear "
    "model families."
)
ANOMALY_ENCODING_NOTE = (
    "IsolationForest receives numeric transformed inputs: the same leak-free "
    "numeric columns scaled with StandardScaler plus the same categorical "
    "columns one-hot encoded with handle_unknown='ignore'."
)
POOL_SCORE_CAVEAT_NOTE = (
    "Pool risk scores are computed by scoring all contracts, including rows "
    "used for model training, so they are operational aggregate scores rather "
    "than held-out generalization metrics like contract-level test AUC."
)
TRACEABILITY_NOTE = (
    "The Phase 6 DVC hash is copied into this report; Phase 9 should log the "
    "same value as an MLflow tag when tracking and promotion are introduced."
)


class RiskModelingError(ValueError):
    """Raised when risk modeling inputs or acceptance checks fail."""


@dataclass(frozen=True)
class RiskModelingReport:
    contract_input_rows: int
    train_rows: int
    test_rows: int
    positive_rows: int
    train_positive_rows: int
    test_positive_rows: int
    target_column: str
    target_choice_note: str
    model_input_columns: tuple[str, ...]
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    leakage_excluded_columns: tuple[str, ...]
    classifier_name: str
    classifier_class_weight: str | None
    calibration_method: str
    calibration_cv_folds: int | None
    calibration_skipped: bool
    calibration_note: str
    model_auc: float
    baseline_auc: float
    model_normalized_gini: float
    baseline_normalized_gini: float
    brier_score: float
    expected_calibration_error: float
    calibration_bin_count: int
    test_empirical_claim_rate: float
    test_mean_predicted_probability: float
    test_probability_rate_delta: float
    pool_input_rows: int
    pool_score_rows: int
    pool_score_coverage: float
    zero_exposure_pools: int
    anomaly_rows: int
    anomaly_count: int
    anomaly_rate: float
    anomaly_contamination: float
    anomaly_encoded_feature_count: int
    anomaly_encoding_note: str
    pool_score_caveat_note: str
    performance_interpretability_note: str
    dvc_output_hash: str | None
    dvc_tracked_path: str | None
    dataset_git_commit_sha: str | None
    traceability_note: str
    acceptance_passed: bool


@dataclass(frozen=True)
class RiskModelingResult:
    classifier_model: Any
    feature_preprocessor: ColumnTransformer
    anomaly_preprocessor: ColumnTransformer
    anomaly_model: IsolationForest
    contract_test_predictions: pd.DataFrame
    all_contract_predictions: pd.DataFrame
    pool_risk_scores: pd.DataFrame
    anomaly_scores: pd.DataFrame
    report: RiskModelingReport


@dataclass(frozen=True)
class RiskModelingPaths:
    model_artifact_path: Path
    contract_test_predictions_path: Path
    all_contract_predictions_path: Path
    pool_risk_scores_path: Path
    anomaly_scores_path: Path
    report_path: Path


def build_risk_models(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    pool_features: pd.DataFrame,
    pool_targets: pd.DataFrame,
    dataset_version_report: dict[str, Any] | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
    anomaly_contamination: float = DEFAULT_ANOMALY_CONTAMINATION,
) -> RiskModelingResult:
    """Train the Phase 7 probability and anomaly models on leak-free features."""
    _validate_inputs(contract_features, contract_targets, pool_features, pool_targets)
    modeling_data = _prepare_contract_modeling_data(contract_features, contract_targets)
    y = modeling_data[TARGET_COLUMN].astype("int64")
    X = modeling_data.loc[:, MODEL_INPUT_COLUMNS].copy()

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X,
        y,
        modeling_data[ID_COLUMN],
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    classifier_model, calibration_cv_folds, calibration_skipped, calibration_note = (
        _fit_probability_model(X_train, y_train)
    )
    feature_preprocessor = _build_preprocessor()
    feature_preprocessor.fit(X_train)
    test_probabilities = _predict_positive_probability(classifier_model, X_test)
    all_probabilities = _predict_positive_probability(classifier_model, X)

    contract_test_predictions = pd.DataFrame(
        {
            ID_COLUMN: test_ids.to_numpy(),
            "target_has_claim": y_test.to_numpy(),
            "predicted_claim_probability": test_probabilities,
        }
    )
    all_contract_predictions = modeling_data.loc[:, [ID_COLUMN, POOL_COLUMN, "Exposure"]].copy()
    all_contract_predictions["target_has_claim"] = y.to_numpy()
    all_contract_predictions["predicted_claim_probability"] = all_probabilities

    pool_risk_scores, zero_exposure_pools = _build_pool_risk_scores(
        all_contract_predictions, pool_features, pool_targets
    )
    anomaly_preprocessor, anomaly_model, anomaly_scores, anomaly_encoded_feature_count = (
        _fit_anomaly_model(modeling_data, anomaly_contamination, random_state)
    )

    metrics = _compute_metrics(y_test.to_numpy(), test_probabilities, y_train.to_numpy())
    pool_score_coverage = float(pool_risk_scores["pool_risk_score"].notna().mean())
    acceptance_passed = (
        metrics["model_auc"] > metrics["baseline_auc"]
        and metrics["model_normalized_gini"] > metrics["baseline_normalized_gini"]
        and pool_score_coverage == 1.0
    )

    classifier = _base_classifier()
    report = RiskModelingReport(
        contract_input_rows=len(modeling_data),
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rows=int(y.sum()),
        train_positive_rows=int(y_train.sum()),
        test_positive_rows=int(y_test.sum()),
        target_column=TARGET_COLUMN,
        target_choice_note=TARGET_CHOICE_NOTE,
        model_input_columns=MODEL_INPUT_COLUMNS,
        numeric_feature_columns=NUMERIC_FEATURE_COLUMNS,
        categorical_feature_columns=CATEGORICAL_FEATURE_COLUMNS,
        leakage_excluded_columns=LEAKAGE_EXCLUDED_COLUMNS,
        classifier_name=classifier.__class__.__name__,
        classifier_class_weight=classifier.class_weight,
        calibration_method="sigmoid",
        calibration_cv_folds=calibration_cv_folds,
        calibration_skipped=calibration_skipped,
        calibration_note=calibration_note,
        model_auc=metrics["model_auc"],
        baseline_auc=metrics["baseline_auc"],
        model_normalized_gini=metrics["model_normalized_gini"],
        baseline_normalized_gini=metrics["baseline_normalized_gini"],
        brier_score=metrics["brier_score"],
        expected_calibration_error=metrics["expected_calibration_error"],
        calibration_bin_count=CALIBRATION_BINS,
        test_empirical_claim_rate=metrics["test_empirical_claim_rate"],
        test_mean_predicted_probability=metrics["test_mean_predicted_probability"],
        test_probability_rate_delta=metrics["test_probability_rate_delta"],
        pool_input_rows=len(pool_features),
        pool_score_rows=int(pool_risk_scores["pool_risk_score"].notna().sum()),
        pool_score_coverage=pool_score_coverage,
        zero_exposure_pools=zero_exposure_pools,
        anomaly_rows=len(anomaly_scores),
        anomaly_count=int(anomaly_scores["anomaly_flag"].sum()),
        anomaly_rate=float(anomaly_scores["anomaly_flag"].mean()),
        anomaly_contamination=anomaly_contamination,
        anomaly_encoded_feature_count=anomaly_encoded_feature_count,
        anomaly_encoding_note=ANOMALY_ENCODING_NOTE,
        pool_score_caveat_note=POOL_SCORE_CAVEAT_NOTE,
        performance_interpretability_note=PERFORMANCE_INTERPRETABILITY_NOTE,
        dvc_output_hash=_string_or_none(dataset_version_report, "dvc_output_hash"),
        dvc_tracked_path=_string_or_none(dataset_version_report, "dvc_tracked_path"),
        dataset_git_commit_sha=_string_or_none(dataset_version_report, "git_commit_sha"),
        traceability_note=TRACEABILITY_NOTE,
        acceptance_passed=acceptance_passed,
    )

    return RiskModelingResult(
        classifier_model=classifier_model,
        feature_preprocessor=feature_preprocessor,
        anomaly_preprocessor=anomaly_preprocessor,
        anomaly_model=anomaly_model,
        contract_test_predictions=contract_test_predictions,
        all_contract_predictions=all_contract_predictions,
        pool_risk_scores=pool_risk_scores,
        anomaly_scores=anomaly_scores,
        report=report,
    )


def write_risk_modeling_artifacts(
    result: RiskModelingResult,
    output_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> RiskModelingPaths:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = RiskModelingPaths(
        model_artifact_path=artifact_dir / MODEL_ARTIFACT_FILENAME,
        contract_test_predictions_path=artifact_dir / CONTRACT_TEST_PREDICTIONS_FILENAME,
        all_contract_predictions_path=artifact_dir / ALL_CONTRACT_PREDICTIONS_FILENAME,
        pool_risk_scores_path=artifact_dir / POOL_RISK_SCORES_FILENAME,
        anomaly_scores_path=artifact_dir / ANOMALY_SCORES_FILENAME,
        report_path=artifact_dir / RISK_REPORT_FILENAME,
    )
    with paths.model_artifact_path.open("wb") as artifact_file:
        pickle.dump(
            {
                "classifier_model": result.classifier_model,
                "feature_preprocessor": result.feature_preprocessor,
                "anomaly_preprocessor": result.anomaly_preprocessor,
                "anomaly_model": result.anomaly_model,
                "report": asdict(result.report),
            },
            artifact_file,
        )
    result.contract_test_predictions.to_csv(paths.contract_test_predictions_path, index=False)
    result.all_contract_predictions.to_csv(paths.all_contract_predictions_path, index=False)
    result.pool_risk_scores.to_csv(paths.pool_risk_scores_path, index=False)
    result.anomaly_scores.to_csv(paths.anomaly_scores_path, index=False)
    paths.report_path.write_text(
        json.dumps(asdict(result.report), indent=2, default=str),
        encoding="utf-8",
    )
    return paths


def run_risk_modeling_pipeline(
    contract_features_path: Path | str = DEFAULT_CONTRACT_FEATURES_PATH,
    contract_targets_path: Path | str = DEFAULT_CONTRACT_TARGETS_PATH,
    pool_features_path: Path | str = DEFAULT_POOL_FEATURES_PATH,
    pool_targets_path: Path | str = DEFAULT_POOL_TARGETS_PATH,
    dataset_version_report_path: Path | str = DEFAULT_DATASET_VERSION_REPORT_PATH,
    output_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    enforce_acceptance: bool = True,
    test_size: float = DEFAULT_TEST_SIZE,
) -> RiskModelingPaths:
    dataset_version_report = _read_optional_json_report(Path(dataset_version_report_path))
    result = build_risk_models(
        contract_features=pd.read_csv(contract_features_path),
        contract_targets=pd.read_csv(contract_targets_path),
        pool_features=pd.read_csv(pool_features_path),
        pool_targets=pd.read_csv(pool_targets_path),
        dataset_version_report=dataset_version_report,
        test_size=test_size,
    )
    paths = write_risk_modeling_artifacts(result, output_dir)
    if enforce_acceptance and not result.report.acceptance_passed:
        raise RiskModelingError(
            "Phase 7 acceptance failed after writing artifacts: model AUC/Gini must beat "
            "baseline and pool score coverage must be 100%."
        )
    return paths


def _validate_inputs(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
    pool_features: pd.DataFrame,
    pool_targets: pd.DataFrame,
) -> None:
    _validate_required_columns(contract_features, (ID_COLUMN, *MODEL_INPUT_COLUMNS), "contract features")
    _validate_required_columns(contract_targets, REQUIRED_CONTRACT_TARGET_COLUMNS, "contract targets")
    _validate_required_columns(pool_features, (POOL_COLUMN,), "pool features")
    _validate_required_columns(pool_targets, REQUIRED_POOL_TARGET_COLUMNS, "pool targets")
    if contract_features.empty:
        raise RiskModelingError("contract features must contain at least one row")
    if contract_targets.empty:
        raise RiskModelingError("contract targets must contain at least one row")
    if contract_features[ID_COLUMN].duplicated().any():
        raise RiskModelingError("contract features contain duplicate IDpol values")
    if contract_targets[ID_COLUMN].duplicated().any():
        raise RiskModelingError("contract targets contain duplicate IDpol values")
    if pool_features[POOL_COLUMN].duplicated().any():
        raise RiskModelingError("pool features contain duplicate pool_id values")

    target_values = set(contract_targets[TARGET_COLUMN].dropna().astype("int64").unique())
    if not target_values.issubset({0, 1}):
        raise RiskModelingError("target_has_claim must be binary 0/1")
    target_counts = contract_targets[TARGET_COLUMN].value_counts()
    if len(target_counts) != 2 or int(target_counts.min()) < 2:
        raise RiskModelingError("target_has_claim must contain at least two rows from each class")


def _validate_required_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], table_name: str
) -> None:
    missing_columns = tuple(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        raise RiskModelingError(
            f"{table_name} is missing required columns: {', '.join(missing_columns)}"
        )


def _prepare_contract_modeling_data(
    contract_features: pd.DataFrame,
    contract_targets: pd.DataFrame,
) -> pd.DataFrame:
    modeling_data = contract_features.merge(
        contract_targets.loc[:, [ID_COLUMN, TARGET_COLUMN]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(modeling_data) != len(contract_features) or len(modeling_data) != len(contract_targets):
        raise RiskModelingError("contract features and targets must have matching IDpol rows")

    if modeling_data.loc[:, MODEL_INPUT_COLUMNS].isna().any().any():
        raise RiskModelingError("model input features must not contain missing values")

    for column in CATEGORICAL_FEATURE_COLUMNS:
        modeling_data[column] = modeling_data[column].astype("string")
    modeling_data[TARGET_COLUMN] = modeling_data[TARGET_COLUMN].astype("int64")
    return modeling_data


def _fit_probability_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[Any, int | None, bool, str]:
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("classifier", _base_classifier()),
        ]
    )
    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        base_pipeline.fit(X_train, y_train)
        return (
            base_pipeline,
            None,
            True,
            "Calibration skipped because at least one training class has fewer than two rows.",
        )

    cv_folds = min(DEFAULT_CALIBRATION_CV, min_class_count)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=cv_folds,
    )
    calibrated_model.fit(X_train, y_train)
    return (
        calibrated_model,
        cv_folds,
        False,
        f"Sigmoid calibration fitted with {cv_folds} stratified folds.",
    )


def _base_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURE_COLUMNS)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        remainder="drop",
    )


def _predict_positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return model.predict_proba(features)[:, positive_index]


def _compute_metrics(
    y_true: np.ndarray, y_probability: np.ndarray, y_train: np.ndarray
) -> dict[str, float]:
    if len(np.unique(y_true)) != 2:
        raise RiskModelingError("test split must contain both target classes")

    model_auc = float(roc_auc_score(y_true, y_probability))
    baseline_probability = float(np.mean(y_train))
    baseline_predictions = np.full(shape=len(y_true), fill_value=baseline_probability)
    baseline_auc = float(roc_auc_score(y_true, baseline_predictions))
    empirical_claim_rate = float(np.mean(y_true))
    mean_predicted_probability = float(np.mean(y_probability))
    return {
        "model_auc": model_auc,
        "baseline_auc": baseline_auc,
        "model_normalized_gini": float(2 * model_auc - 1),
        "baseline_normalized_gini": float(2 * baseline_auc - 1),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "expected_calibration_error": _expected_calibration_error(
            y_true, y_probability, CALIBRATION_BINS
        ),
        "test_empirical_claim_rate": empirical_claim_rate,
        "test_mean_predicted_probability": mean_predicted_probability,
        "test_probability_rate_delta": float(
            abs(mean_predicted_probability - empirical_claim_rate)
        ),
    }


def _expected_calibration_error(
    y_true: np.ndarray, y_probability: np.ndarray, n_bins: int
) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(y_probability, bin_edges[1:-1], right=True), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if not np.any(in_bin):
            continue
        bin_accuracy = float(np.mean(y_true[in_bin]))
        bin_confidence = float(np.mean(y_probability[in_bin]))
        ece += float(np.mean(in_bin)) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _build_pool_risk_scores(
    all_contract_predictions: pd.DataFrame,
    pool_features: pd.DataFrame,
    pool_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    predictions = all_contract_predictions.copy()
    pool_feature_ids = pool_features.loc[:, [POOL_COLUMN]].copy()
    target_rates = pool_targets.loc[:, [POOL_COLUMN, "pool_claim_rate"]].copy()
    predictions[POOL_COLUMN] = predictions[POOL_COLUMN].astype("string")
    pool_feature_ids[POOL_COLUMN] = pool_feature_ids[POOL_COLUMN].astype("string")
    target_rates[POOL_COLUMN] = target_rates[POOL_COLUMN].astype("string")

    grouped_scores = []
    for pool_id, group in predictions.groupby(POOL_COLUMN, observed=True):
        total_exposure = float(group["Exposure"].sum())
        if total_exposure > 0:
            risk_score = float(
                np.average(group["predicted_claim_probability"], weights=group["Exposure"])
            )
            weighting_method = "exposure_weighted"
        else:
            risk_score = float(group["predicted_claim_probability"].mean())
            weighting_method = "unweighted_zero_exposure_fallback"
        grouped_scores.append(
            {
                POOL_COLUMN: pool_id,
                "pool_contract_count": int(len(group)),
                "pool_scored_exposure": total_exposure,
                "pool_risk_score": risk_score,
                "pool_score_weighting_method": weighting_method,
            }
        )

    score_frame = pd.DataFrame(grouped_scores)
    scores = pool_feature_ids.merge(score_frame, on=POOL_COLUMN, how="left")
    scores = scores.merge(target_rates, on=POOL_COLUMN, how="left")
    zero_exposure_pools = int(
        (scores["pool_score_weighting_method"] == "unweighted_zero_exposure_fallback").sum()
    )
    return scores, zero_exposure_pools


def _fit_anomaly_model(
    modeling_data: pd.DataFrame,
    contamination: float,
    random_state: int,
) -> tuple[ColumnTransformer, IsolationForest, pd.DataFrame, int]:
    features = modeling_data.loc[:, MODEL_INPUT_COLUMNS].copy()
    preprocessor = _build_preprocessor()
    encoded_features = preprocessor.fit_transform(features)
    anomaly_model = IsolationForest(contamination=contamination, random_state=random_state)
    anomaly_model.fit(encoded_features)
    anomaly_predictions = anomaly_model.predict(encoded_features)
    anomaly_scores = -anomaly_model.score_samples(encoded_features)
    anomaly_frame = modeling_data.loc[:, [ID_COLUMN, POOL_COLUMN]].copy()
    anomaly_frame["anomaly_score"] = anomaly_scores
    anomaly_frame["anomaly_flag"] = (anomaly_predictions == -1).astype("int64")
    return preprocessor, anomaly_model, anomaly_frame, int(encoded_features.shape[1])


def _read_optional_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _string_or_none(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 7 risk models.")
    parser.add_argument(
        "--contract-features-path",
        type=Path,
        default=DEFAULT_CONTRACT_FEATURES_PATH,
        help=f"Contract features CSV. Defaults to {DEFAULT_CONTRACT_FEATURES_PATH}.",
    )
    parser.add_argument(
        "--contract-targets-path",
        type=Path,
        default=DEFAULT_CONTRACT_TARGETS_PATH,
        help=f"Contract targets CSV. Defaults to {DEFAULT_CONTRACT_TARGETS_PATH}.",
    )
    parser.add_argument(
        "--pool-features-path",
        type=Path,
        default=DEFAULT_POOL_FEATURES_PATH,
        help=f"Pool features CSV. Defaults to {DEFAULT_POOL_FEATURES_PATH}.",
    )
    parser.add_argument(
        "--pool-targets-path",
        type=Path,
        default=DEFAULT_POOL_TARGETS_PATH,
        help=f"Pool targets CSV. Defaults to {DEFAULT_POOL_TARGETS_PATH}.",
    )
    parser.add_argument(
        "--dataset-version-report-path",
        type=Path,
        default=DEFAULT_DATASET_VERSION_REPORT_PATH,
        help=f"Dataset version report JSON. Defaults to {DEFAULT_DATASET_VERSION_REPORT_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Artifact directory. Defaults to {DEFAULT_ARTIFACT_DIR}.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f"Stratified test fraction. Defaults to {DEFAULT_TEST_SIZE}.",
    )
    parser.add_argument(
        "--no-enforce-acceptance",
        action="store_true",
        help="Write artifacts without raising if Phase 7 acceptance criteria fail.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_risk_modeling_pipeline(
        contract_features_path=args.contract_features_path,
        contract_targets_path=args.contract_targets_path,
        pool_features_path=args.pool_features_path,
        pool_targets_path=args.pool_targets_path,
        dataset_version_report_path=args.dataset_version_report_path,
        output_dir=args.output_dir,
        enforce_acceptance=not args.no_enforce_acceptance,
        test_size=args.test_size,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
