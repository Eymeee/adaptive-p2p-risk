"""Monitoring and traceability reporting for Phase 11.

The explicit join key for FR-MT-04 is `retraining_cycle_id`. Paths are still
reported for human auditability, but they are not trusted as stable identifiers
because artifact files can be overwritten between runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.continual.drift import CONTINUAL_REPORT_FILENAME
from src.continual.drift import DEFAULT_PHASE8_ARTIFACT_DIR
from src.continual.drift import RETRAINING_EVENTS_FILENAME
from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.versioning import DEFAULT_DATASET_VERSION_REPORT_FILENAME
from src.mlops.pipeline import DEFAULT_PHASE9_ARTIFACT_DIR
from src.mlops.pipeline import MLOPS_REPORT_FILENAME
from src.mlops.pipeline import VALIDATION_REPORT_FILENAME

DEFAULT_PHASE10_ARTIFACT_DIR = Path("artifacts/phase10")
DEFAULT_PHASE11_ARTIFACT_DIR = Path("artifacts/phase11")
DEPLOYMENT_REPORT_FILENAME = "deployment_report.json"
DECISION_LOG_FILENAME = "risk_decisions.jsonl"
MONITORING_REPORT_FILENAME = "monitoring_report.json"
RETRAINING_CYCLE_LOG_FILENAME = "retraining_cycle_log.csv"
LOGGING_POLICY_NOTE = (
    "NFR-07: the API does not log full request/response payloads. Logs include only request "
    "ID, method, path, status code, duration, and selected model source; IDpol, raw contract "
    "attributes, predicted probabilities, and pool member lists are not logged."
)
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
TRACEABILITY_CHAIN_NOTE = (
    "Risk decisions join to retraining cycles through retraining_cycle_id, and "
    "each cycle carries model artifact identity plus the Phase 6 DVC dataset hash."
)
ALERTING_SCOPE_NOTE = (
    "Prometheus rules in this repository are local monitoring rules. They reuse "
    "documented status signals and do not introduce supervisor-approved production SLOs."
)
EXTERNAL_ALERTING_SCOPE_NOTE = (
    "External notification channels such as email, Slack, or PagerDuty are deferred."
)


class MonitoringTraceabilityError(ValueError):
    """Raised when monitoring reports cannot be assembled."""


@dataclass(frozen=True)
class MonitoringReport:
    phase11_artifact_dir: str
    decision_log_path: str
    retraining_cycle_log_path: str
    decision_count: int
    decision_retraining_cycle_ids: tuple[str, ...]
    serving_retraining_cycle_id: str | None
    model_source: str | None
    model_artifact_path: str | None
    model_artifact_sha256: str | None
    dvc_output_hash: str | None
    dataset_git_commit_sha: str | None
    trace_logging_enabled: bool | None
    trace_salt_mode: str | None
    trace_salt_policy_note: str
    traceability_join_note: str
    traceability_chain_note: str
    logging_policy_note: str
    alerting_scope_note: str
    external_alerting_scope_note: str
    phase8_data_drift_detected: bool
    phase8_concept_drift_detected: bool
    phase9_retraining_requested: bool
    phase9_validation_passed: bool
    phase9_model_registered: bool
    alert_rules: tuple[str, ...]


@dataclass(frozen=True)
class MonitoringPaths:
    monitoring_report_path: Path
    retraining_cycle_log_path: Path


def build_monitoring_report(
    phase8_report: dict[str, Any] | None,
    retraining_events: list[dict[str, Any]],
    phase9_pipeline_report: dict[str, Any] | None,
    phase9_validation_report: dict[str, Any] | None,
    deployment_report: dict[str, Any] | None,
    dataset_version_report: dict[str, Any] | None,
    decision_records: list[dict[str, Any]],
    output_dir: Path | str = DEFAULT_PHASE11_ARTIFACT_DIR,
) -> tuple[MonitoringReport, list[dict[str, Any]]]:
    output_path = Path(output_dir)
    dvc_hash = _first_present(
        deployment_report,
        dataset_version_report,
        phase9_validation_report,
        key="dvc_output_hash",
    )
    git_sha = _first_present(
        deployment_report,
        dataset_version_report,
        phase9_validation_report,
        key="dataset_git_commit_sha",
    )
    serving_cycle_id = _string_or_none(deployment_report, "retraining_cycle_id")
    cycle_rows = _build_retraining_cycle_rows(
        retraining_events=retraining_events,
        phase9_pipeline_report=phase9_pipeline_report,
        phase9_validation_report=phase9_validation_report,
        deployment_report=deployment_report,
        dvc_output_hash=dvc_hash,
        dataset_git_commit_sha=git_sha,
    )
    decision_cycle_ids = tuple(
        sorted(
            {
                str(record["retraining_cycle_id"])
                for record in decision_records
                if record.get("retraining_cycle_id") is not None
            }
        )
    )
    if serving_cycle_id and all(row["retraining_cycle_id"] != serving_cycle_id for row in cycle_rows):
        cycle_rows.append(
            {
                "retraining_cycle_id": serving_cycle_id,
                "cycle_source": _string_or_none(deployment_report, "selected_model_source") or "",
                "trigger_batch": "",
                "triggering_detector": "",
                "training_window_batches": "",
                "validation_passed": _string_or_none(phase9_validation_report, "validation_passed") or "",
                "model_registered": _string_or_none(phase9_pipeline_report, "model_registered") or "",
                "registered_model_version": _string_or_none(
                    phase9_pipeline_report, "registered_model_version"
                )
                or "",
                "model_artifact_path": _string_or_none(
                    deployment_report, "selected_model_artifact_path"
                )
                or "",
                "model_artifact_sha256": _string_or_none(
                    deployment_report, "selected_model_artifact_sha256"
                )
                or "",
                "dvc_output_hash": dvc_hash or "",
                "dataset_git_commit_sha": git_sha or "",
            }
        )

    report = MonitoringReport(
        phase11_artifact_dir=str(output_path),
        decision_log_path=str(output_path / DECISION_LOG_FILENAME),
        retraining_cycle_log_path=str(output_path / RETRAINING_CYCLE_LOG_FILENAME),
        decision_count=len(decision_records),
        decision_retraining_cycle_ids=decision_cycle_ids,
        serving_retraining_cycle_id=serving_cycle_id,
        model_source=_string_or_none(deployment_report, "selected_model_source"),
        model_artifact_path=_string_or_none(deployment_report, "selected_model_artifact_path"),
        model_artifact_sha256=_string_or_none(deployment_report, "selected_model_artifact_sha256"),
        dvc_output_hash=dvc_hash,
        dataset_git_commit_sha=git_sha,
        trace_logging_enabled=_bool_or_none(deployment_report, "trace_logging_enabled"),
        trace_salt_mode=_string_or_none(deployment_report, "trace_salt_mode"),
        trace_salt_policy_note=TRACE_SALT_POLICY_NOTE,
        traceability_join_note=TRACEABILITY_JOIN_NOTE,
        traceability_chain_note=TRACEABILITY_CHAIN_NOTE,
        logging_policy_note=LOGGING_POLICY_NOTE,
        alerting_scope_note=ALERTING_SCOPE_NOTE,
        external_alerting_scope_note=EXTERNAL_ALERTING_SCOPE_NOTE,
        phase8_data_drift_detected=_drift_detected(phase8_report, "data_drift_evaluation"),
        phase8_concept_drift_detected=_drift_detected(phase8_report, "concept_drift_evaluation"),
        phase9_retraining_requested=bool((phase9_pipeline_report or {}).get("retraining_requested")),
        phase9_validation_passed=bool((phase9_validation_report or {}).get("validation_passed")),
        phase9_model_registered=bool((phase9_pipeline_report or {}).get("model_registered")),
        alert_rules=(
            "AdaptiveP2PModelUnavailable",
            "AdaptiveP2PDataDriftDetected",
            "AdaptiveP2PConceptDriftDetected",
            "AdaptiveP2PRetrainingValidationFailed",
            "AdaptiveP2PTraceLoggingDisabled",
            "AdaptiveP2PTraceLoggingFailures",
        ),
    )
    return report, cycle_rows


def run_monitoring_traceability_pipeline(
    phase8_report_path: Path | str = DEFAULT_PHASE8_ARTIFACT_DIR / CONTINUAL_REPORT_FILENAME,
    retraining_events_path: Path | str = DEFAULT_PHASE8_ARTIFACT_DIR / RETRAINING_EVENTS_FILENAME,
    phase9_pipeline_report_path: Path | str = DEFAULT_PHASE9_ARTIFACT_DIR / MLOPS_REPORT_FILENAME,
    phase9_validation_report_path: Path | str = DEFAULT_PHASE9_ARTIFACT_DIR / VALIDATION_REPORT_FILENAME,
    deployment_report_path: Path | str = DEFAULT_PHASE10_ARTIFACT_DIR / DEPLOYMENT_REPORT_FILENAME,
    dataset_version_report_path: Path | str = (
        DEFAULT_PROCESSED_DIR / DEFAULT_DATASET_VERSION_REPORT_FILENAME
    ),
    decision_log_path: Path | str = DEFAULT_PHASE11_ARTIFACT_DIR / DECISION_LOG_FILENAME,
    output_dir: Path | str = DEFAULT_PHASE11_ARTIFACT_DIR,
) -> MonitoringPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    phase8_report = _read_optional_json(Path(phase8_report_path))
    retraining_events = _read_optional_csv_records(Path(retraining_events_path))
    phase9_pipeline_report = _read_optional_json(Path(phase9_pipeline_report_path))
    phase9_validation_report = _read_optional_json(Path(phase9_validation_report_path))
    deployment_report = _read_optional_json(Path(deployment_report_path))
    dataset_version_report = _read_optional_json(Path(dataset_version_report_path))
    decision_records = _read_optional_jsonl(Path(decision_log_path))

    report, cycle_rows = build_monitoring_report(
        phase8_report=phase8_report,
        retraining_events=retraining_events,
        phase9_pipeline_report=phase9_pipeline_report,
        phase9_validation_report=phase9_validation_report,
        deployment_report=deployment_report,
        dataset_version_report=dataset_version_report,
        decision_records=decision_records,
        output_dir=output_path,
    )
    report_path = output_path / MONITORING_REPORT_FILENAME
    cycle_log_path = output_path / RETRAINING_CYCLE_LOG_FILENAME
    report_path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    _write_cycle_log(cycle_log_path, cycle_rows)
    return MonitoringPaths(
        monitoring_report_path=report_path,
        retraining_cycle_log_path=cycle_log_path,
    )


def _build_retraining_cycle_rows(
    retraining_events: list[dict[str, Any]],
    phase9_pipeline_report: dict[str, Any] | None,
    phase9_validation_report: dict[str, Any] | None,
    deployment_report: dict[str, Any] | None,
    dvc_output_hash: str | None,
    dataset_git_commit_sha: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_trigger_batch = _string_or_none(phase9_pipeline_report, "selected_retraining_trigger_batch")
    serving_source = _string_or_none(deployment_report, "selected_model_source")
    for event in retraining_events:
        trigger_batch = str(event.get("trigger_batch", ""))
        is_selected = selected_trigger_batch is not None and trigger_batch == selected_trigger_batch
        is_served_phase9_cycle = is_selected and serving_source == "phase9_retrained_candidate"
        artifact_sha = (
            _string_or_none(deployment_report, "selected_model_artifact_sha256")
            if is_served_phase9_cycle
            else ""
        )
        rows.append(
            {
                "retraining_cycle_id": (
                    _string_or_none(deployment_report, "retraining_cycle_id")
                    if is_served_phase9_cycle
                    else f"phase8-event:{trigger_batch}"
                ),
                "cycle_source": _cycle_source(is_selected, is_served_phase9_cycle),
                "trigger_batch": trigger_batch,
                "triggering_detector": str(event.get("triggering_detector", "")),
                "training_window_batches": str(event.get("training_window_batches", "")),
                "validation_passed": (
                    _string_or_none(phase9_validation_report, "validation_passed") if is_selected else ""
                )
                or "",
                "model_registered": (
                    _string_or_none(phase9_pipeline_report, "model_registered") if is_selected else ""
                )
                or "",
                "registered_model_version": (
                    _string_or_none(phase9_pipeline_report, "registered_model_version")
                    if is_selected
                    else ""
                )
                or "",
                "model_artifact_path": (
                    _string_or_none(deployment_report, "selected_model_artifact_path")
                    if is_served_phase9_cycle
                    else ""
                )
                or "",
                "model_artifact_sha256": artifact_sha or "",
                "dvc_output_hash": dvc_output_hash or "",
                "dataset_git_commit_sha": dataset_git_commit_sha or "",
            }
        )
    return rows


def _cycle_source(is_selected: bool, is_served_phase9_cycle: bool) -> str:
    if is_served_phase9_cycle:
        return "phase9_served_retraining_cycle"
    if is_selected:
        return "phase9_selected_unserved_retraining_event"
    return "phase8_retraining_event"


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise MonitoringTraceabilityError(
                f"decision log line {line_number} is not a JSON object"
            )
        records.append(value)
    return records


def _read_optional_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_cycle_log(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "retraining_cycle_id",
        "cycle_source",
        "trigger_batch",
        "triggering_detector",
        "training_window_batches",
        "validation_passed",
        "model_registered",
        "registered_model_version",
        "model_artifact_path",
        "model_artifact_sha256",
        "dvc_output_hash",
        "dataset_git_commit_sha",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: "" if row.get(column) is None else row.get(column, "") for column in columns}
            for row in rows
        )


def _first_present(
    *payloads: dict[str, Any] | None,
    key: str,
) -> str | None:
    for payload in payloads:
        value = _string_or_none(payload, key)
        if value:
            return value
    return None


def _string_or_none(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None or payload.get(key) is None:
        return None
    return str(payload[key])


def _bool_or_none(payload: dict[str, Any] | None, key: str) -> bool | None:
    if payload is None or key not in payload:
        return None
    return bool(payload[key])


def _drift_detected(payload: dict[str, Any] | None, key: str) -> bool:
    if not payload:
        return False
    evaluation = payload.get(key)
    if not isinstance(evaluation, dict):
        return False
    return evaluation.get("first_detected_batch") is not None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 11 monitoring traceability reports.")
    parser.add_argument("--phase8-report-path", type=Path, default=DEFAULT_PHASE8_ARTIFACT_DIR / CONTINUAL_REPORT_FILENAME)
    parser.add_argument("--retraining-events-path", type=Path, default=DEFAULT_PHASE8_ARTIFACT_DIR / RETRAINING_EVENTS_FILENAME)
    parser.add_argument("--phase9-pipeline-report-path", type=Path, default=DEFAULT_PHASE9_ARTIFACT_DIR / MLOPS_REPORT_FILENAME)
    parser.add_argument("--phase9-validation-report-path", type=Path, default=DEFAULT_PHASE9_ARTIFACT_DIR / VALIDATION_REPORT_FILENAME)
    parser.add_argument("--deployment-report-path", type=Path, default=DEFAULT_PHASE10_ARTIFACT_DIR / DEPLOYMENT_REPORT_FILENAME)
    parser.add_argument("--dataset-version-report-path", type=Path, default=DEFAULT_PROCESSED_DIR / DEFAULT_DATASET_VERSION_REPORT_FILENAME)
    parser.add_argument("--decision-log-path", type=Path, default=DEFAULT_PHASE11_ARTIFACT_DIR / DECISION_LOG_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE11_ARTIFACT_DIR)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_monitoring_traceability_pipeline(
        phase8_report_path=args.phase8_report_path,
        retraining_events_path=args.retraining_events_path,
        phase9_pipeline_report_path=args.phase9_pipeline_report_path,
        phase9_validation_report_path=args.phase9_validation_report_path,
        deployment_report_path=args.deployment_report_path,
        dataset_version_report_path=args.dataset_version_report_path,
        decision_log_path=args.decision_log_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
