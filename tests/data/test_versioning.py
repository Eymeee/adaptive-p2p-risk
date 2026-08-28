from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.versioning import DOWNSTREAM_CONSUMPTION_NOTE
from src.data.versioning import SNAPSHOT_GRANULARITY_NOTE
from src.data.versioning import DatasetVersioningError
from src.data.versioning import build_dataset_version_report
from src.data.versioning import expected_processed_artifacts
from src.data.versioning import run_dataset_versioning_pipeline
from src.data.versioning import write_dataset_version_report


def test_expected_processed_artifacts_excludes_raw_files() -> None:
    artifacts = expected_processed_artifacts()

    assert "freMTPL2freq_cleaned.csv" in artifacts
    assert "stream/batch_019.csv" in artifacts
    assert "dataset_version_report.json" not in artifacts
    assert all(not artifact.startswith("../raw") for artifact in artifacts)
    assert all("data/raw" not in artifact for artifact in artifacts)


def test_build_dataset_version_report_extracts_dvc_hash_and_notes(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    dvc_metadata_path = tmp_path / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    _write_dvc_metadata(dvc_metadata_path)

    report = build_dataset_version_report(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=tmp_path,
        generated_at_utc="2026-08-28T00:00:00+00:00",
    )

    assert report.dvc_tracked_path == "data/processed"
    assert report.dvc_metadata_file == str(dvc_metadata_path)
    assert report.dvc_output_hash == "abc123.dir"
    assert report.dvc_output_hash_name == "md5"
    assert report.generated_at_utc == "2026-08-28T00:00:00+00:00"
    assert report.missing_artifacts == ()
    assert report.git_commit_sha is None
    assert report.remote_configured is False
    assert report.snapshot_granularity_note == SNAPSHOT_GRANULARITY_NOTE
    assert report.downstream_consumption_note == DOWNSTREAM_CONSUMPTION_NOTE
    assert "MLflow run tag" in report.downstream_consumption_note


def test_build_dataset_version_report_normalizes_dvc_path_under_data_dir(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "data" / "processed"
    dvc_metadata_path = tmp_path / "data" / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    dvc_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    dvc_metadata_path.write_text(
        "\n".join(
            [
                "outs:",
                "- md5: abc123.dir",
                "  path: processed",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = build_dataset_version_report(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=tmp_path,
    )

    assert report.dvc_tracked_path.endswith("data/processed")


def test_build_dataset_version_report_raises_when_expected_artifact_is_missing(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    dvc_metadata_path = tmp_path / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    (processed_dir / "stream" / "batch_019.csv").unlink()
    _write_dvc_metadata(dvc_metadata_path)

    with pytest.raises(DatasetVersioningError, match="stream/batch_019.csv"):
        build_dataset_version_report(
            processed_dir=processed_dir,
            dvc_metadata_path=dvc_metadata_path,
            repo_dir=tmp_path,
        )


def test_build_dataset_version_report_can_allow_missing_artifacts(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    dvc_metadata_path = tmp_path / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    (processed_dir / "stream" / "batch_018.csv").unlink()
    _write_dvc_metadata(dvc_metadata_path)

    report = build_dataset_version_report(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=tmp_path,
        allow_missing=True,
    )

    assert report.missing_artifacts == ("stream/batch_018.csv",)


def test_build_dataset_version_report_raises_for_missing_dvc_metadata(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    _create_expected_artifacts(processed_dir)

    with pytest.raises(DatasetVersioningError, match="DVC metadata file does not exist"):
        build_dataset_version_report(
            processed_dir=processed_dir,
            dvc_metadata_path=tmp_path / "missing.dvc",
            repo_dir=tmp_path,
        )


def test_write_dataset_version_report_serializes_json(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    dvc_metadata_path = tmp_path / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    _write_dvc_metadata(dvc_metadata_path)
    report = build_dataset_version_report(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=tmp_path,
        generated_at_utc="2026-08-28T00:00:00+00:00",
    )

    paths = write_dataset_version_report(report, processed_dir)

    written_report = json.loads(paths.report_path.read_text(encoding="utf-8"))
    assert paths.report_path == processed_dir / "dataset_version_report.json"
    assert written_report["dvc_output_hash"] == "abc123.dir"
    assert written_report["snapshot_granularity_note"] == SNAPSHOT_GRANULARITY_NOTE


def test_run_dataset_versioning_pipeline_writes_report(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    dvc_metadata_path = tmp_path / "processed.dvc"
    _create_expected_artifacts(processed_dir)
    _write_dvc_metadata(dvc_metadata_path)

    paths = run_dataset_versioning_pipeline(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=tmp_path,
    )

    assert paths.report_path.exists()


def _create_expected_artifacts(processed_dir: Path) -> None:
    for relative_path in expected_processed_artifacts():
        path = processed_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")


def _write_dvc_metadata(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "outs:",
                "- md5: abc123.dir",
                "  size: 123",
                "  nfiles: 42",
                "  hash: md5",
                "  path: data/processed",
                "",
            ]
        ),
        encoding="utf-8",
    )
