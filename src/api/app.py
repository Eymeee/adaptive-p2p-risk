"""FastAPI deployment service for Phases 10 and 11.

The service derives serving features through `src.data.features` instead of
duplicating Phase 4 thresholds locally. It also avoids request/response payload
logging to keep NFR-07 explicit: logs contain operational metadata only, while
Phase 11 trace records store salted fingerprints instead of raw payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import secrets
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.continual.drift import CONTINUAL_REPORT_FILENAME
from src.continual.drift import DEFAULT_PHASE8_ARTIFACT_DIR
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
DEFAULT_PHASE11_ARTIFACT_DIR = Path("artifacts/phase11")
DECISION_LOG_FILENAME = "risk_decisions.jsonl"
MODEL_ARTIFACT_ENV_VAR = "MODEL_ARTIFACT_PATH"
TRACE_HASH_SALT_ENV_VAR = "TRACE_HASH_SALT"
TRACE_DEV_MODE_ENV_VAR = "TRACE_DEV_MODE"
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
TRACE_SALT_POLICY_NOTE = (
    "TRACE_HASH_SALT is required for reproducible secure trace fingerprints. If it is absent "
    "and TRACE_DEV_MODE=true, the API generates an ephemeral per-process salt; if both are absent, "
    "decision trace logging is disabled. No fixed fallback salt is used."
)
TRACEABILITY_JOIN_NOTE = (
    "Model source/path alone is not sufficient for FR-MT-04 because artifact paths can be "
    "overwritten. Decision records therefore carry retraining_cycle_id as the explicit join key "
    "to retraining_cycle_log.csv."
)
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
    phase8_report_path: Path = DEFAULT_PHASE8_ARTIFACT_DIR / CONTINUAL_REPORT_FILENAME
    phase9_validation_report_path: Path = DEFAULT_PHASE9_ARTIFACT_DIR / VALIDATION_REPORT_FILENAME
    phase9_pipeline_report_path: Path = DEFAULT_PHASE9_ARTIFACT_DIR / "mlops_pipeline_report.json"
    dataset_version_report_path: Path = (
        Path("data/processed") / DEFAULT_DATASET_VERSION_REPORT_FILENAME
    )
    deployment_report_path: Path = DEFAULT_PHASE10_ARTIFACT_DIR / DEPLOYMENT_REPORT_FILENAME
    decision_log_path: Path = DEFAULT_PHASE11_ARTIFACT_DIR / DECISION_LOG_FILENAME


@dataclass(frozen=True)
class ServedModel:
    model: Any | None
    source: str
    artifact_path: Path | None
    artifact_sha256: str | None
    retraining_cycle_id: str
    model_loaded: bool
    error: str | None
    artifact_report: dict[str, Any] | None
    phase8_report: dict[str, Any] | None
    phase7_report: dict[str, Any] | None
    phase9_validation_report: dict[str, Any] | None
    phase9_pipeline_report: dict[str, Any] | None
    dataset_version_report: dict[str, Any] | None


@dataclass(frozen=True)
class TraceContext:
    enabled: bool
    salt: bytes | None
    salt_mode: str
    decision_log_path: Path
    disabled_reason: str | None


@dataclass(frozen=True)
class ApiMetrics:
    registry: CollectorRegistry
    requests_total: Counter
    request_latency_seconds: Histogram
    scoring_requests_total: Counter
    contract_score: Histogram
    pool_score: Histogram
    trace_log_failures_total: Counter
    model_loaded: Gauge
    trace_logging_enabled: Gauge
    data_drift_detected: Gauge
    concept_drift_detected: Gauge
    retraining_requested: Gauge
    candidate_validation_passed: Gauge
    model_registered: Gauge


@dataclass(frozen=True)
class DeploymentReport:
    service_name: str
    model_loaded: bool
    selected_model_source: str
    selected_model_artifact_path: str | None
    selected_model_artifact_sha256: str | None
    retraining_cycle_id: str
    model_resolution_order: tuple[str, ...]
    deployment_error: str | None
    endpoints: tuple[str, ...]
    feature_parity_note: str
    logging_policy_note: str
    pool_scope_note: str
    latency_note: str
    kubernetes_scope_note: str
    trace_logging_enabled: bool
    trace_salt_mode: str
    trace_salt_policy_note: str
    traceability_join_note: str
    phase7_acceptance_passed: bool | None
    phase8_acceptance_passed: bool | None
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
    decision_id: str
    request_id: str
    model_source: str
    model_artifact_path: str
    retraining_cycle_id: str
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
    decision_id: str
    request_id: str
    pool_id: str
    model_source: str
    model_artifact_path: str
    retraining_cycle_id: str
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
    trace_context = build_trace_context(deployment_settings)
    metrics = build_api_metrics(model_context, trace_context)
    try:
        write_deployment_report(
            deployment_settings.deployment_report_path,
            model_context,
            trace_context,
        )
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
    api.state.trace_context = trace_context
    api.state.metrics = metrics

    @api.middleware("http")
    async def safe_request_logging(request: Request, call_next: Any) -> Any:
        request_id = uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        context: ServedModel = request.app.state.model_context
        metric_path = _route_path(request)
        metrics.requests_total.labels(
            method=request.method,
            path=metric_path,
            status_code=str(response.status_code),
            model_source=context.source,
        ).inc()
        metrics.request_latency_seconds.labels(
            method=request.method,
            path=metric_path,
            model_source=context.source,
        ).observe(duration_ms / 1000.0)
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
        return _model_version_payload(context, _trace_context(api))

    @api.get("/metrics")
    def metrics_endpoint() -> Response:
        registry = _api_metrics(api).registry
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @api.post("/score/contract", response_model=ContractScoreResponse)
    def score_contract(contract: ContractScoreRequest, request: Request) -> ContractScoreResponse:
        started_at = time.perf_counter()
        context = _require_model(_model_context(api))
        decision_id = uuid4().hex
        features = _build_model_features([contract])
        probability = float(_predict_positive_probability(context.model, features)[0])
        _api_metrics(api).scoring_requests_total.labels(
            endpoint="/score/contract",
            model_source=context.source,
        ).inc()
        _api_metrics(api).contract_score.labels(model_source=context.source).observe(probability)
        _write_decision_record(
            api=api,
            decision_id=decision_id,
            request_id=_request_id(request),
            endpoint="/score/contract",
            model_context=context,
            input_payload=contract.model_dump(mode="json"),
            prediction_summary={"contract_count": 1, "mean_probability": probability},
        )
        return ContractScoreResponse(
            decision_id=decision_id,
            request_id=_request_id(request),
            model_source=context.source,
            model_artifact_path=str(context.artifact_path),
            retraining_cycle_id=context.retraining_cycle_id,
            predicted_claim_probability=probability,
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    @api.post("/score/pool", response_model=PoolScoreResponse)
    def score_pool(pool: PoolScoreRequest, request: Request) -> PoolScoreResponse:
        started_at = time.perf_counter()
        context = _require_model(_model_context(api))
        decision_id = uuid4().hex
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

        _api_metrics(api).scoring_requests_total.labels(
            endpoint="/score/pool",
            model_source=context.source,
        ).inc()
        _api_metrics(api).pool_score.labels(model_source=context.source).observe(pool_risk_score)
        _write_decision_record(
            api=api,
            decision_id=decision_id,
            request_id=_request_id(request),
            endpoint="/score/pool",
            model_context=context,
            input_payload=pool.model_dump(mode="json"),
            prediction_summary={
                "contract_count": len(pool.members),
                "mean_probability": float(np.mean(probabilities)),
                "pool_risk_score": pool_risk_score,
                "pool_scored_exposure": total_exposure,
                "pool_score_weighting_method": weighting_method,
            },
        )
        return PoolScoreResponse(
            decision_id=decision_id,
            request_id=_request_id(request),
            pool_id=pool.pool_id,
            model_source=context.source,
            model_artifact_path=str(context.artifact_path),
            retraining_cycle_id=context.retraining_cycle_id,
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
    phase8_report = _read_optional_json(settings.phase8_report_path)
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
            phase8_report=phase8_report,
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
            phase8_report=phase8_report,
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
            phase8_report=phase8_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
            fail_reason_prefix="Phase 7 model is accepted but artifact is invalid",
        )

    return ServedModel(
        model=None,
        source="unavailable",
        artifact_path=None,
        artifact_sha256=None,
        retraining_cycle_id="unavailable",
        model_loaded=False,
        error="No validated model artifact is available for serving.",
        artifact_report=None,
        phase8_report=phase8_report,
        phase7_report=phase7_report,
        phase9_validation_report=phase9_validation_report,
        phase9_pipeline_report=phase9_pipeline_report,
        dataset_version_report=dataset_version_report,
    )


def build_trace_context(settings: DeploymentSettings) -> TraceContext:
    env_salt = os.environ.get(TRACE_HASH_SALT_ENV_VAR)
    if env_salt:
        return TraceContext(
            enabled=True,
            salt=env_salt.encode("utf-8"),
            salt_mode="env_var",
            decision_log_path=settings.decision_log_path,
            disabled_reason=None,
        )
    if os.environ.get(TRACE_DEV_MODE_ENV_VAR, "").lower() == "true":
        return TraceContext(
            enabled=True,
            salt=secrets.token_bytes(32),
            salt_mode="ephemeral_dev_random",
            decision_log_path=settings.decision_log_path,
            disabled_reason=None,
        )
    return TraceContext(
        enabled=False,
        salt=None,
        salt_mode="disabled_missing_trace_hash_salt",
        decision_log_path=settings.decision_log_path,
        disabled_reason=(
            f"{TRACE_HASH_SALT_ENV_VAR} is unset and {TRACE_DEV_MODE_ENV_VAR}=true was not provided."
        ),
    )


def build_api_metrics(context: ServedModel, trace_context: TraceContext) -> ApiMetrics:
    registry = CollectorRegistry(auto_describe=True)
    metrics = ApiMetrics(
        registry=registry,
        requests_total=Counter(
            "adaptive_p2p_api_requests_total",
            "API requests by method, path, status, and model source.",
            ("method", "path", "status_code", "model_source"),
            registry=registry,
        ),
        request_latency_seconds=Histogram(
            "adaptive_p2p_api_request_latency_seconds",
            "API request latency in seconds.",
            ("method", "path", "model_source"),
            registry=registry,
        ),
        scoring_requests_total=Counter(
            "adaptive_p2p_scoring_requests_total",
            "Scoring requests by endpoint and model source.",
            ("endpoint", "model_source"),
            registry=registry,
        ),
        contract_score=Histogram(
            "adaptive_p2p_contract_score",
            "Contract claim probability distribution.",
            ("model_source",),
            buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=registry,
        ),
        pool_score=Histogram(
            "adaptive_p2p_pool_score",
            "Pool risk score distribution.",
            ("model_source",),
            buckets=(0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
            registry=registry,
        ),
        trace_log_failures_total=Counter(
            "adaptive_p2p_trace_log_failures_total",
            "Decision trace log write failures.",
            registry=registry,
        ),
        model_loaded=Gauge(
            "adaptive_p2p_model_loaded",
            "Whether a validated serving model is loaded.",
            registry=registry,
        ),
        trace_logging_enabled=Gauge(
            "adaptive_p2p_trace_logging_enabled",
            "Whether decision trace logging is enabled.",
            registry=registry,
        ),
        data_drift_detected=Gauge(
            "adaptive_p2p_data_drift_detected",
            "Whether Phase 8 data drift was detected.",
            registry=registry,
        ),
        concept_drift_detected=Gauge(
            "adaptive_p2p_concept_drift_detected",
            "Whether Phase 8 concept drift was detected.",
            registry=registry,
        ),
        retraining_requested=Gauge(
            "adaptive_p2p_retraining_requested",
            "Whether Phase 9 requested retraining.",
            registry=registry,
        ),
        candidate_validation_passed=Gauge(
            "adaptive_p2p_candidate_validation_passed",
            "Whether Phase 9 candidate validation passed.",
            registry=registry,
        ),
        model_registered=Gauge(
            "adaptive_p2p_model_registered",
            "Whether Phase 9 registered a model version.",
            registry=registry,
        ),
    )
    metrics.model_loaded.set(float(context.model_loaded))
    metrics.trace_logging_enabled.set(float(trace_context.enabled))
    metrics.data_drift_detected.set(_drift_detected(context.phase8_report, "data_drift_evaluation"))
    metrics.concept_drift_detected.set(
        _drift_detected(context.phase8_report, "concept_drift_evaluation")
    )
    metrics.retraining_requested.set(_bool_metric(context.phase9_pipeline_report, "retraining_requested"))
    metrics.candidate_validation_passed.set(
        _bool_metric(context.phase9_validation_report, "validation_passed")
    )
    metrics.model_registered.set(_bool_metric(context.phase9_pipeline_report, "model_registered"))
    return metrics


def write_deployment_report(path: Path, context: ServedModel, trace_context: TraceContext) -> None:
    report = DeploymentReport(
        service_name=SERVICE_NAME,
        model_loaded=context.model_loaded,
        selected_model_source=context.source,
        selected_model_artifact_path=None if context.artifact_path is None else str(context.artifact_path),
        selected_model_artifact_sha256=context.artifact_sha256,
        retraining_cycle_id=context.retraining_cycle_id,
        model_resolution_order=MODEL_RESOLUTION_ORDER,
        deployment_error=context.error,
        endpoints=("/health", "/model/version", "/metrics", "/score/contract", "/score/pool"),
        feature_parity_note=FEATURE_PARITY_NOTE,
        logging_policy_note=LOGGING_POLICY_NOTE,
        pool_scope_note=POOL_SCOPE_NOTE,
        latency_note=LATENCY_NOTE,
        kubernetes_scope_note=KUBERNETES_SCOPE_NOTE,
        trace_logging_enabled=trace_context.enabled,
        trace_salt_mode=trace_context.salt_mode,
        trace_salt_policy_note=TRACE_SALT_POLICY_NOTE,
        traceability_join_note=TRACEABILITY_JOIN_NOTE,
        phase7_acceptance_passed=_bool_or_none(context.phase7_report, "acceptance_passed"),
        phase8_acceptance_passed=_bool_or_none(context.phase8_report, "acceptance_passed"),
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
    phase8_report: dict[str, Any] | None,
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
            artifact_sha256=None,
            retraining_cycle_id="unavailable",
            model_loaded=False,
            error=f"{fail_reason_prefix}: {error}",
            artifact_report=None,
            phase8_report=phase8_report,
            phase7_report=phase7_report,
            phase9_validation_report=phase9_validation_report,
            phase9_pipeline_report=phase9_pipeline_report,
            dataset_version_report=dataset_version_report,
        )

    artifact_sha256 = _file_sha256(path)
    return ServedModel(
        model=artifact["classifier_model"],
        source=source,
        artifact_path=path,
        artifact_sha256=artifact_sha256,
        retraining_cycle_id=_build_retraining_cycle_id(
            source=source,
            artifact_sha256=artifact_sha256,
            phase9_pipeline_report=phase9_pipeline_report,
        ),
        model_loaded=True,
        error=None,
        artifact_report=artifact.get("report") if isinstance(artifact.get("report"), dict) else None,
        phase8_report=phase8_report,
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


def _write_decision_record(
    api: FastAPI,
    decision_id: str,
    request_id: str,
    endpoint: str,
    model_context: ServedModel,
    input_payload: dict[str, Any],
    prediction_summary: dict[str, Any],
) -> None:
    trace_context = _trace_context(api)
    if not trace_context.enabled:
        return
    try:
        fingerprint = _fingerprint_payload(input_payload, trace_context)
        record = {
            "decision_id": decision_id,
            "request_id": request_id,
            "endpoint": endpoint,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model_source": model_context.source,
            "model_artifact_path": str(model_context.artifact_path),
            "model_artifact_sha256": model_context.artifact_sha256,
            "retraining_cycle_id": model_context.retraining_cycle_id,
            "dvc_output_hash": _string_or_none(
                model_context.dataset_version_report, "dvc_output_hash"
            ),
            "dataset_git_commit_sha": _string_or_none(
                model_context.dataset_version_report, "git_commit_sha"
            ),
            "prediction_summary": prediction_summary,
            "input_fingerprint": fingerprint,
            "trace_salt_mode": trace_context.salt_mode,
        }
        trace_context.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_context.decision_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError as error:
        _api_metrics(api).trace_log_failures_total.inc()
        LOGGER.warning(
            "decision_trace_write_failed path=%s error_type=%s",
            trace_context.decision_log_path,
            error.__class__.__name__,
        )


def _fingerprint_payload(payload: dict[str, Any], trace_context: TraceContext) -> str:
    if trace_context.salt is None:
        raise DeploymentError("trace fingerprint requested while trace salt is unavailable")
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hmac.new(
        trace_context.salt,
        canonical_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _model_context(api: FastAPI) -> ServedModel:
    return api.state.model_context


def _trace_context(api: FastAPI) -> TraceContext:
    return api.state.trace_context


def _api_metrics(api: FastAPI) -> ApiMetrics:
    return api.state.metrics


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


def _model_version_payload(context: ServedModel, trace_context: TraceContext) -> dict[str, Any]:
    phase7_report = context.phase7_report or {}
    phase8_report = context.phase8_report or {}
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
        "selected_model_artifact_sha256": context.artifact_sha256,
        "retraining_cycle_id": context.retraining_cycle_id,
        "model_resolution_order": MODEL_RESOLUTION_ORDER,
        "deployment_error": context.error,
        "phase7_acceptance_passed": phase7_report.get("acceptance_passed"),
        "phase7_model_auc": phase7_report.get("model_auc"),
        "phase7_model_normalized_gini": phase7_report.get("model_normalized_gini"),
        "phase7_expected_calibration_error": phase7_report.get("expected_calibration_error"),
        "phase8_acceptance_passed": phase8_report.get("acceptance_passed"),
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
        "trace_logging_enabled": trace_context.enabled,
        "trace_salt_mode": trace_context.salt_mode,
        "trace_disabled_reason": trace_context.disabled_reason,
        "trace_salt_policy_note": TRACE_SALT_POLICY_NOTE,
        "traceability_join_note": TRACEABILITY_JOIN_NOTE,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_retraining_cycle_id(
    source: str,
    artifact_sha256: str,
    phase9_pipeline_report: dict[str, Any] | None,
) -> str:
    if source == "phase7_reference_model":
        return f"phase7-reference:{artifact_sha256}"
    if source == "phase9_retrained_candidate":
        trigger_batch = None if phase9_pipeline_report is None else phase9_pipeline_report.get(
            "selected_retraining_trigger_batch"
        )
        if trigger_batch is None:
            return f"phase9:{artifact_sha256}"
        return f"phase9:{trigger_batch}:{artifact_sha256}"
    if source == "env_override":
        return f"env-override:{artifact_sha256}"
    return f"{source}:{artifact_sha256}"


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    return request.url.path


def _bool_metric(payload: dict[str, Any] | None, key: str) -> float:
    return float(bool(payload and payload.get(key)))


def _drift_detected(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    evaluation = payload.get(key)
    if not isinstance(evaluation, dict):
        return 0.0
    return float(evaluation.get("first_detected_batch") is not None)


app = create_app()
