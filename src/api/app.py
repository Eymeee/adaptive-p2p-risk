"""FastAPI deployment service for Phase 10.

The service derives serving features through `src.data.features` instead of
duplicating Phase 4 thresholds locally. It also avoids request/response payload
logging to keep NFR-07 explicit: logs contain operational metadata only.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.data import features as feature_engineering
from src.data.versioning import DEFAULT_DATASET_VERSION_REPORT_FILENAME
from src.mlops.pipeline import DEFAULT_PHASE9_ARTIFACT_DIR
from src.mlops.pipeline import RETRAINED_CANDIDATE_DIRNAME
from src.mlops.pipeline import RETRAINED_CANDIDATE_MODEL_FILENAME
from src.mlops.pipeline import VALIDATION_REPORT_FILENAME
from src.models.risk import DEFAULT_ARTIFACT_DIR as DEFAULT_PHASE7_ARTIFACT_DIR
from src.models.risk import ID_COLUMN
from src.models.risk import MODEL_ARTIFACT_FILENAME
from src.models.risk import MODEL_INPUT_COLUMNS
from src.models.risk import POOL_COLUMN
from src.models.risk import RISK_REPORT_FILENAME
from src.models.risk import _predict_positive_probability

LOGGER = logging.getLogger(__name__)

DEFAULT_PHASE10_ARTIFACT_DIR = Path("artifacts/phase10")
DEPLOYMENT_REPORT_FILENAME = "deployment_report.json"
MODEL_ARTIFACT_ENV_VAR = "MODEL_ARTIFACT_PATH"
SERVICE_NAME = "adaptive-p2p-risk-api"

FEATURE_PARITY_NOTE = (
    "Serving feature construction calls src.data.features.build_serving_contract_features, "
    "which reuses Phase 4 Density_log1p and band definitions to avoid training-serving skew."
)
LOGGING_POLICY_NOTE = (
    "NFR-07: the API does not log full request/response payloads. Logs include only request "
    "ID, method, path, status code, duration, and selected model source; IDpol, raw contract "
    "attributes, predicted probabilities, and pool member lists are not logged."
)
POOL_SCOPE_NOTE = (
    "CdCT Section 9.1 allows pool scoring by pool identifier or member list. Phase 10 implements "
    "member-list scoring only because no deployment database or feature store exists yet for "
    "pool-id membership lookup."
)
LATENCY_NOTE = (
    "No numeric API latency threshold is enforced because NFR-01 latency targets are pending "
    "supervisor confirmation; request duration is measured for observability only."
)
KUBERNETES_SCOPE_NOTE = "Kubernetes orchestration is deferred because FR-DP-03 is Could/optional."
MODEL_RESOLUTION_ORDER: tuple[str, ...] = (
    "MODEL_ARTIFACT_PATH env override",
    "validated Phase 9 retrained candidate",
    "accepted Phase 7 reference model",
)


class DeploymentError(ValueError):
    """Raised when deployment artifacts cannot be served safely."""


@dataclass(frozen=True)
class DeploymentSettings:
    phase7_model_path: Path = DEFAULT_PHASE7_ARTIFACT_DIR / MODEL_ARTIFACT_FILENAME
    phase7_report_path: Path = DEFAULT_PHASE7_ARTIFACT_DIR / RISK_REPORT_FILENAME
    phase9_model_path: Path = (
        DEFAULT_PHASE9_ARTIFACT_DIR
        / RETRAINED_CANDIDATE_DIRNAME
        / RETRAINED_CANDIDATE_MODEL_FILENAME
    )
    phase9_validation_report_path: Path = DEFAULT_PHASE9_ARTIFACT_DIR / VALIDATION_REPORT_FILENAME
    phase9_pipeline_report_path: Path = DEFAULT_PHASE9_ARTIFACT_DIR / "mlops_pipeline_report.json"
    dataset_version_report_path: Path = (
        Path("data/processed") / DEFAULT_DATASET_VERSION_REPORT_FILENAME
    )
    deployment_report_path: Path = DEFAULT_PHASE10_ARTIFACT_DIR / DEPLOYMENT_REPORT_FILENAME


@dataclass(frozen=True)
class ServedModel:
    model: Any | None
    source: str
    artifact_path: Path | None
    model_loaded: bool
    error: str | None
    artifact_report: dict[str, Any] | None
    phase7_report: dict[str, Any] | None
    phase9_validation_report: dict[str, Any] | None
    phase9_pipeline_report: dict[str, Any] | None
    dataset_version_report: dict[str, Any] | None


@dataclass(frozen=True)
class DeploymentReport:
    service_name: str
    model_loaded: bool
    selected_model_source: str
    selected_model_artifact_path: str | None
    model_resolution_order: tuple[str, ...]
    deployment_error: str | None
    endpoints: tuple[str, ...]
    feature_parity_note: str
    logging_policy_note: str
    pool_scope_note: str
    latency_note: str
    kubernetes_scope_note: str
    phase7_acceptance_passed: bool | None
    phase9_validation_passed: bool | None
    dvc_output_hash: str | None
    dataset_git_commit_sha: str | None


class ContractScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    IDpol: int | str | None = None
    pool_id: str
    Exposure: float = Field(ge=0.0)
    VehPower: float = Field(ge=0.0)
    VehAge: float = Field(ge=0.0)
    DrivAge: float = Field(ge=0.0)
    BonusMalus: float = Field(ge=0.0)
    Density: float = Field(ge=0.0)
    VehBrand: str
    VehGas: str
    Area: str
    Region: str


class ContractScoreResponse(BaseModel):
    request_id: str
    model_source: str
    model_artifact_path: str
    predicted_claim_probability: float
    processing_time_ms: float


class PoolScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_id: str
    members: list[ContractScoreRequest] = Field(min_length=1)


class PoolMemberScore(BaseModel):
    member_index: int
    predicted_claim_probability: float


class PoolScoreResponse(BaseModel):
    request_id: str
    pool_id: str
    model_source: str
    model_artifact_path: str
    pool_risk_score: float
    pool_contract_count: int
    pool_scored_exposure: float
    pool_score_weighting_method: str
    member_scores: list[PoolMemberScore]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    service_name: str
    model_loaded: bool
    selected_model_source: str
    selected_model_artifact_path: str | None
    error: str | None


def create_app(settings: DeploymentSettings | None = None) -> FastAPI:
    deployment_settings = settings or DeploymentSettings()
    model_context = resolve_serving_model(deployment_settings)
    try:
        write_deployment_report(deployment_settings.deployment_report_path, model_context)
    except OSError as error:
        LOGGER.warning(
            "deployment_report_write_failed path=%s error_type=%s",
            deployment_settings.deployment_report_path,
            error.__class__.__name__,
        )

    api = FastAPI(
        title="Adaptive P2P Risk API",
        version="0.1.0",
        description="Phase 10 deployment service for contract and pool risk scoring.",
    )
    api.state.model_context = model_context

    @api.middleware("http")
    async def safe_request_logging(request: Request, call_next: Any) -> Any:
        request_id = uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        context: ServedModel = request.app.state.model_context
        LOGGER.info(
            "request_id=%s method=%s path=%s status_code=%s duration_ms=%.3f model_source=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            context.source,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @api.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized_errors = [
            {
                "type": str(error.get("type", "")),
                "loc": tuple(error.get("loc", ())),
                "msg": str(error.get("msg", "")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "detail": sanitized_errors,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        context = _model_context(api)
        return HealthResponse(
            status="ok" if context.model_loaded else "degraded",
            service_name=SERVICE_NAME,
            model_loaded=context.model_loaded,
            selected_model_source=context.source,
            selected_model_artifact_path=(
                None if context.artifact_path is None else str(context.artifact_path)
            ),
            error=context.error,
        )

    @api.get("/model/version")
    def model_version() -> dict[str, Any]:
        context = _model_context(api)
        return _model_version_payload(context)

    @api.post("/score/contract", response_model=ContractScoreResponse)
    def score_contract(contract: ContractScoreRequest, request: Request) -> ContractScoreResponse:
        started_at = time.perf_counter()
        context = _require_model(_model_context(api))
        features = _build_model_features([contract])
        probability = float(_predict_positive_probability(context.model, features)[0])
        return ContractScoreResponse(
            request_id=_request_id(request),
            model_source=context.source,
            model_artifact_path=str(context.artifact_path),
            predicted_claim_probability=probability,
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    @api.post("/score/pool", response_model=PoolScoreResponse)
    def score_pool(pool: PoolScoreRequest, request: Request) -> PoolScoreResponse:
        started_at = time.perf_counter()
        context = _require_model(_model_context(api))
        if any(member.pool_id != pool.pool_id for member in pool.members):
            raise HTTPException(
                status_code=400,
                detail="All pool member rows must use the same pool_id as the pool request.",
            )

        features = _build_model_features(pool.members)
        probabilities = _predict_positive_probability(context.model, features)
        exposures = features["Exposure"].to_numpy(dtype=float)
        total_exposure = float(exposures.sum())
        if total_exposure > 0.0:
            pool_risk_score = float(np.average(probabilities, weights=exposures))
            weighting_method = "exposure_weighted"
        else:
            pool_risk_score = float(np.mean(probabilities))
            weighting_method = "unweighted_zero_exposure_fallback"

        return PoolScoreResponse(
            request_id=_request_id(request),
            pool_id=pool.pool_id,
            model_source=context.source,
            model_artifact_path=str(context.artifact_path),
            pool_risk_score=pool_risk_score,
            pool_contract_count=len(pool.members),
            pool_scored_exposure=total_exposure,
            pool_score_weighting_method=weighting_method,
            member_scores=[
                PoolMemberScore(
                    member_index=index,
                    predicted_claim_probability=float(probability),
                )
                for index, probability in enumerate(probabilities)
            ],
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    return api


def resolve_serving_model(settings: DeploymentSettings) -> ServedModel:
    phase7_report = _read_optional_json(settings.phase7_report_path)
    phase9_validation_report = _read_optional_json(settings.phase9_validation_report_path)
    phase9_pipeline_report = _read_optional_json(settings.phase9_pipeline_report_path)
    dataset_version_report = _read_optional_json(settings.dataset_version_report_path)

    env_artifact_path = os.environ.get(MODEL_ARTIFACT_ENV_VAR)
    if env_artifact_path:
        path = Path(env_artifact_path)
        return _context_from_candidate(
            path=path,
            source="env_override",
            phase7_report=phase7_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
            fail_reason_prefix=f"{MODEL_ARTIFACT_ENV_VAR} is set but invalid",
        )

    if bool((phase9_validation_report or {}).get("validation_passed")):
        return _context_from_candidate(
            path=settings.phase9_model_path,
            source="phase9_retrained_candidate",
            phase7_report=phase7_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
            fail_reason_prefix="Phase 9 validation passed but candidate artifact is invalid",
        )

    if bool((phase7_report or {}).get("acceptance_passed")):
        return _context_from_candidate(
            path=settings.phase7_model_path,
            source="phase7_reference_model",
            phase7_report=phase7_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
            fail_reason_prefix="Phase 7 model is accepted but artifact is invalid",
        )

    return ServedModel(
        model=None,
        source="unavailable",
        artifact_path=None,
        model_loaded=False,
        error="No validated model artifact is available for serving.",
        artifact_report=None,
        phase7_report=phase7_report,
        phase9_validation_report=phase9_validation_report,
        phase9_pipeline_report=phase9_pipeline_report,
        dataset_version_report=dataset_version_report,
    )


def write_deployment_report(path: Path, context: ServedModel) -> None:
    report = DeploymentReport(
        service_name=SERVICE_NAME,
        model_loaded=context.model_loaded,
        selected_model_source=context.source,
        selected_model_artifact_path=None if context.artifact_path is None else str(context.artifact_path),
        model_resolution_order=MODEL_RESOLUTION_ORDER,
        deployment_error=context.error,
        endpoints=("/health", "/model/version", "/score/contract", "/score/pool"),
        feature_parity_note=FEATURE_PARITY_NOTE,
        logging_policy_note=LOGGING_POLICY_NOTE,
        pool_scope_note=POOL_SCOPE_NOTE,
        latency_note=LATENCY_NOTE,
        kubernetes_scope_note=KUBERNETES_SCOPE_NOTE,
        phase7_acceptance_passed=_bool_or_none(context.phase7_report, "acceptance_passed"),
        phase9_validation_passed=_bool_or_none(context.phase9_validation_report, "validation_passed"),
        dvc_output_hash=_string_or_none(context.dataset_version_report, "dvc_output_hash"),
        dataset_git_commit_sha=_string_or_none(context.dataset_version_report, "git_commit_sha"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")


def _context_from_candidate(
    path: Path,
    source: str,
    phase7_report: dict[str, Any] | None,
    phase9_validation_report: dict[str, Any] | None,
    phase9_pipeline_report: dict[str, Any] | None,
    dataset_version_report: dict[str, Any] | None,
    fail_reason_prefix: str,
) -> ServedModel:
    try:
        artifact = _read_model_artifact(path)
    except DeploymentError as error:
        return ServedModel(
            model=None,
            source="unavailable",
            artifact_path=path,
            model_loaded=False,
            error=f"{fail_reason_prefix}: {error}",
            artifact_report=None,
            phase7_report=phase7_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
        )

    return ServedModel(
        model=artifact["classifier_model"],
        source=source,
        artifact_path=path,
        model_loaded=True,
        error=None,
        artifact_report=artifact.get("report") if isinstance(artifact.get("report"), dict) else None,
        phase7_report=phase7_report,
        phase9_validation_report=phase9_validation_report,
        phase9_pipeline_report=phase9_pipeline_report,
        dataset_version_report=dataset_version_report,
    )


def _read_model_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DeploymentError(f"model artifact does not exist: {path}")
    with path.open("rb") as artifact_file:
        payload = pickle.load(artifact_file)
    if not isinstance(payload, dict):
        raise DeploymentError("model artifact payload must be a dictionary")
    model = payload.get("classifier_model")
    if model is None or not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise DeploymentError("model artifact must contain classifier_model with predict_proba/classes_")
    if 1 not in set(model.classes_):
        raise DeploymentError("classifier_model must expose positive class label 1")
    return payload


def _build_model_features(contracts: list[ContractScoreRequest]) -> pd.DataFrame:
    frame = pd.DataFrame([contract.model_dump() for contract in contracts])
    serving_features = feature_engineering.build_serving_contract_features(frame)
    missing_columns = tuple(
        column for column in MODEL_INPUT_COLUMNS if column not in serving_features.columns
    )
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Serving feature builder omitted model columns: {', '.join(missing_columns)}",
        )
    return serving_features.loc[:, MODEL_INPUT_COLUMNS].copy()


def _model_context(api: FastAPI) -> ServedModel:
    return api.state.model_context


def _require_model(context: ServedModel) -> ServedModel:
    if context.model_loaded and context.model is not None and context.artifact_path is not None:
        return context
    raise HTTPException(
        status_code=503,
        detail={
            "error": "model_unavailable",
            "reason": context.error or "No validated model is loaded.",
        },
    )


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _model_version_payload(context: ServedModel) -> dict[str, Any]:
    phase7_report = context.phase7_report or {}
    phase9_validation_report = context.phase9_validation_report or {}
    phase9_pipeline_report = context.phase9_pipeline_report or {}
    dataset_report = context.dataset_version_report or {}
    return {
        "service_name": SERVICE_NAME,
        "model_loaded": context.model_loaded,
        "selected_model_source": context.source,
        "selected_model_artifact_path": (
            None if context.artifact_path is None else str(context.artifact_path)
        ),
        "model_resolution_order": MODEL_RESOLUTION_ORDER,
        "deployment_error": context.error,
        "phase7_acceptance_passed": phase7_report.get("acceptance_passed"),
        "phase7_model_auc": phase7_report.get("model_auc"),
        "phase7_model_normalized_gini": phase7_report.get("model_normalized_gini"),
        "phase7_expected_calibration_error": phase7_report.get("expected_calibration_error"),
        "phase9_candidate_source": phase9_validation_report.get("candidate_source"),
        "phase9_validation_passed": phase9_validation_report.get("validation_passed"),
        "phase9_model_registered": phase9_pipeline_report.get("model_registered"),
        "phase9_registered_model_version": phase9_pipeline_report.get("registered_model_version"),
        "dvc_output_hash": dataset_report.get("dvc_output_hash"),
        "dvc_tracked_path": dataset_report.get("dvc_tracked_path"),
        "dataset_git_commit_sha": dataset_report.get("git_commit_sha"),
        "feature_parity_note": FEATURE_PARITY_NOTE,
        "logging_policy_note": LOGGING_POLICY_NOTE,
        "pool_scope_note": POOL_SCOPE_NOTE,
        "latency_note": LATENCY_NOTE,
        "kubernetes_scope_note": KUBERNETES_SCOPE_NOTE,
    }


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _bool_or_none(payload: dict[str, Any] | None, key: str) -> bool | None:
    if payload is None or key not in payload:
        return None
    return bool(payload[key])


def _string_or_none(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None or payload.get(key) is None:
        return None
    return str(payload[key])


app = create_app()
