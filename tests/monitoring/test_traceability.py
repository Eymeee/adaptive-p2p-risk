from __future__ import annotations

import csv
import json
from pathlib import Path

from src.api.app import LOGGING_POLICY_NOTE
from src.api.app import TRACE_SALT_POLICY_NOTE
from src.api.app import TRACEABILITY_JOIN_NOTE
from src.monitoring.traceability import MONITORING_REPORT_FILENAME
from src.monitoring.traceability import RETRAINING_CYCLE_LOG_FILENAME
from src.monitoring.traceability import run_monitoring_traceability_pipeline


def test_monitoring_traceability_pipeline_writes_report_and_cycle_log(tmp_path: Path) -> None:
    paths = _write_monitoring_inputs(tmp_path)

    result = run_monitoring_traceability_pipeline(
        phase8_report_path=paths["phase8_report"],
        retraining_events_path=paths["retraining_events"],
        phase9_pipeline_report_path=paths["phase9_pipeline"],
        phase9_validation_report_path=paths["phase9_validation"],
        deployment_report_path=paths["deployment_report"],
        dataset_version_report_path=paths["dataset_version"],
        decision_log_path=paths["decision_log"],
        output_dir=tmp_path / "phase11",
    )

    assert result.monitoring_report_path.name == MONITORING_REPORT_FILENAME
    assert result.retraining_cycle_log_path.name == RETRAINING_CYCLE_LOG_FILENAME
    report = json.loads(result.monitoring_report_path.read_text(encoding="utf-8"))
    cycle_rows = _read_csv(result.retraining_cycle_log_path)
    assert report["decision_count"] == 2
    assert report["serving_retraining_cycle_id"] == "phase9:19:abc123"
    assert report["decision_retraining_cycle_ids"] == ["phase9:19:abc123"]
    assert report["trace_salt_policy_note"] == TRACE_SALT_POLICY_NOTE
    assert report["traceability_join_note"] == TRACEABILITY_JOIN_NOTE
    assert report["logging_policy_note"] == LOGGING_POLICY_NOTE
    assert report["phase8_data_drift_detected"] is True
    assert report["phase8_concept_drift_detected"] is True
    assert report["phase9_retraining_requested"] is True
    assert report["phase9_validation_passed"] is False
    assert report["phase9_model_registered"] is False
    assert any(row["retraining_cycle_id"] == "phase9:19:abc123" for row in cycle_rows)


def test_monitoring_cycle_log_has_serving_row_even_without_phase8_events(tmp_path: Path) -> None:
    paths = _write_monitoring_inputs(tmp_path, with_events=False)

    result = run_monitoring_traceability_pipeline(
        phase8_report_path=paths["phase8_report"],
        retraining_events_path=paths["retraining_events"],
        phase9_pipeline_report_path=paths["phase9_pipeline"],
        phase9_validation_report_path=paths["phase9_validation"],
        deployment_report_path=paths["deployment_report"],
        dataset_version_report_path=paths["dataset_version"],
        decision_log_path=paths["decision_log"],
        output_dir=tmp_path / "phase11",
    )

    cycle_rows = _read_csv(result.retraining_cycle_log_path)
    assert len(cycle_rows) == 1
    assert cycle_rows[0]["retraining_cycle_id"] == "phase9:19:abc123"
    assert cycle_rows[0]["dvc_output_hash"] == "hash.dir"


def test_failed_phase9_candidate_does_not_relabel_phase7_serving_cycle(tmp_path: Path) -> None:
    paths = _write_monitoring_inputs(
        tmp_path,
        deployment_source="phase7_reference_model",
        retraining_cycle_id="phase7-reference:def456",
    )

    result = run_monitoring_traceability_pipeline(
        phase8_report_path=paths["phase8_report"],
        retraining_events_path=paths["retraining_events"],
        phase9_pipeline_report_path=paths["phase9_pipeline"],
        phase9_validation_report_path=paths["phase9_validation"],
        deployment_report_path=paths["deployment_report"],
        dataset_version_report_path=paths["dataset_version"],
        decision_log_path=paths["decision_log"],
        output_dir=tmp_path / "phase11",
    )

    cycle_rows = _read_csv(result.retraining_cycle_log_path)
    assert any(row["retraining_cycle_id"] == "phase7-reference:def456" for row in cycle_rows)
    selected_failed_event = next(row for row in cycle_rows if row["trigger_batch"] == "19")
    assert selected_failed_event["retraining_cycle_id"] == "phase8-event:19"
    assert selected_failed_event["cycle_source"] == "phase9_selected_unserved_retraining_event"
    assert selected_failed_event["model_artifact_path"] == ""
    assert selected_failed_event["model_artifact_sha256"] == ""


def _write_monitoring_inputs(
    tmp_path: Path,
    with_events: bool = True,
    deployment_source: str = "phase9_retrained_candidate",
    retraining_cycle_id: str = "phase9:19:abc123",
) -> dict[str, Path]:
    phase8_dir = tmp_path / "phase8"
    phase9_dir = tmp_path / "phase9"
    phase10_dir = tmp_path / "phase10"
    phase11_dir = tmp_path / "phase11-input"
    processed_dir = tmp_path / "processed"
    for directory in (phase8_dir, phase9_dir, phase10_dir, phase11_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phase8_report = phase8_dir / "continual_learning_report.json"
    phase8_report.write_text(
        json.dumps(
            {
                "acceptance_passed": True,
                "data_drift_evaluation": {"first_detected_batch": 12},
                "concept_drift_evaluation": {"first_detected_batch": 15},
            }
        ),
        encoding="utf-8",
    )
    retraining_events = phase8_dir / "retraining_events.csv"
    if with_events:
        retraining_events.write_text(
            "trigger_batch,triggering_detector,training_window_batches\n"
            '12,data,"8,9,10,11,12"\n'
            '19,data_and_concept,"15,16,17,18,19"\n',
            encoding="utf-8",
        )
    else:
        retraining_events.write_text(
            "trigger_batch,triggering_detector,training_window_batches\n",
            encoding="utf-8",
        )
    phase9_pipeline = phase9_dir / "mlops_pipeline_report.json"
    phase9_pipeline.write_text(
        json.dumps(
            {
                "retraining_requested": True,
                "selected_retraining_trigger_batch": 19,
                "model_registered": False,
                "registered_model_version": None,
            }
        ),
        encoding="utf-8",
    )
    phase9_validation = phase9_dir / "candidate_validation_report.json"
    phase9_validation.write_text(
        json.dumps({"validation_passed": False, "dvc_output_hash": "hash.dir"}),
        encoding="utf-8",
    )
    deployment_report = phase10_dir / "deployment_report.json"
    deployment_report.write_text(
        json.dumps(
            {
                "selected_model_source": deployment_source,
                "selected_model_artifact_path": "artifacts/phase9/retrained_candidate/risk_model.pkl",
                "selected_model_artifact_sha256": "abc123",
                "retraining_cycle_id": retraining_cycle_id,
                "dvc_output_hash": "hash.dir",
                "dataset_git_commit_sha": "git123",
                "trace_logging_enabled": True,
                "trace_salt_mode": "env_var",
            }
        ),
        encoding="utf-8",
    )
    dataset_version = processed_dir / "dataset_version_report.json"
    dataset_version.write_text(
        json.dumps({"dvc_output_hash": "hash.dir", "git_commit_sha": "git123"}),
        encoding="utf-8",
    )
    decision_log = phase11_dir / "risk_decisions.jsonl"
    decision_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "decision_id": "decision-1",
                        "retraining_cycle_id": retraining_cycle_id,
                        "input_fingerprint": "fp1",
                    }
                ),
                json.dumps(
                    {
                        "decision_id": "decision-2",
                        "retraining_cycle_id": retraining_cycle_id,
                        "input_fingerprint": "fp2",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "phase8_report": phase8_report,
        "retraining_events": retraining_events,
        "phase9_pipeline": phase9_pipeline,
        "phase9_validation": phase9_validation,
        "deployment_report": deployment_report,
        "dataset_version": dataset_version,
        "decision_log": decision_log,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
