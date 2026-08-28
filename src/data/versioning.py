"""Dataset version reporting for Phase 6.

`data/processed` is versioned as one DVC snapshot for simplicity. That means
re-running any one processed artifact, such as Phase 5 stream batches, changes
the version for the whole processed dataset, including unchanged Phase 1-4
outputs. Phase 7 training should read this report and log the DVC hash as an
MLflow run tag; Phase 9 should preserve that link through promotion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.data.cleaning import DEFAULT_CLEANING_REPORT_FILENAME
from src.data.cleaning import DEFAULT_CLEANED_FREQUENCY_FILENAME
from src.data.cleaning import DEFAULT_CLEANED_SEVERITY_FILENAME
from src.data.cleaning import DEFAULT_INGESTION_REPORT_FILENAME
from src.data.cleaning import DEFAULT_PROCESSED_DIR
from src.data.features import DEFAULT_CONTRACT_FEATURES_FILENAME
from src.data.features import DEFAULT_CONTRACT_TARGETS_FILENAME
from src.data.features import DEFAULT_FEATURE_REPORT_FILENAME
from src.data.features import DEFAULT_POOL_FEATURES_FILENAME
from src.data.features import DEFAULT_POOL_TARGETS_FILENAME
from src.data.pools import DEFAULT_POOL_REPORT_FILENAME
from src.data.pools import DEFAULT_POOLED_FREQUENCY_FILENAME
from src.data.streaming import DEFAULT_DRIFT_GROUND_TRUTH_FILENAME
from src.data.streaming import DEFAULT_N_BATCHES
from src.data.streaming import DEFAULT_STREAM_REPORT_FILENAME

DEFAULT_DVC_METADATA_PATH = Path("data/processed.dvc")
DEFAULT_DATASET_VERSION_REPORT_FILENAME = "dataset_version_report.json"
SNAPSHOT_GRANULARITY_NOTE = (
    "data/processed is versioned as one DVC snapshot for Phase 6 simplicity. "
    "Any single artifact change, such as re-running Phase 5, bumps the version "
    "for the full processed dataset, including unchanged Phase 1-4 outputs. "
    "Finer artifact-level DVC granularity can be introduced later if needed."
)
DOWNSTREAM_CONSUMPTION_NOTE = (
    "Phase 7 training scripts should read dataset_version_report.json and log "
    "dvc_output_hash as an MLflow run tag. Phase 9 promotion should preserve "
    "the model version to dataset DVC hash to scoring decision link for NFR-06 "
    "traceability."
)
NO_REMOTE_NOTE = "No DVC remote is configured in Phase 6; remote storage is deferred to Phase 9."


class DatasetVersioningError(ValueError):
    """Raised when processed artifacts cannot be version-reported safely."""


@dataclass(frozen=True)
class DatasetVersionReport:
    dvc_tracked_path: str
    dvc_metadata_file: str
    dvc_output_hash: str
    dvc_output_hash_name: str | None
    generated_at_utc: str
    expected_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    git_commit_sha: str | None
    remote_configured: bool
    snapshot_granularity_note: str
    downstream_consumption_note: str
    remote_note: str


@dataclass(frozen=True)
class DatasetVersionPaths:
    report_path: Path


def expected_processed_artifacts(n_batches: int = DEFAULT_N_BATCHES) -> tuple[str, ...]:
    stream_batches = tuple(f"stream/batch_{batch_id:03d}.csv" for batch_id in range(n_batches))
    return (
        DEFAULT_INGESTION_REPORT_FILENAME,
        DEFAULT_CLEANING_REPORT_FILENAME,
        DEFAULT_CLEANED_FREQUENCY_FILENAME,
        DEFAULT_CLEANED_SEVERITY_FILENAME,
        DEFAULT_POOLED_FREQUENCY_FILENAME,
        DEFAULT_POOL_REPORT_FILENAME,
        DEFAULT_CONTRACT_FEATURES_FILENAME,
        DEFAULT_CONTRACT_TARGETS_FILENAME,
        DEFAULT_POOL_FEATURES_FILENAME,
        DEFAULT_POOL_TARGETS_FILENAME,
        DEFAULT_FEATURE_REPORT_FILENAME,
        *stream_batches,
        f"stream/{DEFAULT_STREAM_REPORT_FILENAME}",
        f"stream/{DEFAULT_DRIFT_GROUND_TRUTH_FILENAME}",
    )


def build_dataset_version_report(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    dvc_metadata_path: Path | str = DEFAULT_DVC_METADATA_PATH,
    repo_dir: Path | str = Path("."),
    allow_missing: bool = False,
    generated_at_utc: str | None = None,
) -> DatasetVersionReport:
    processed_path = Path(processed_dir)
    dvc_metadata_file = Path(dvc_metadata_path)
    expected_artifacts = expected_processed_artifacts()
    missing_artifacts = tuple(
        artifact for artifact in expected_artifacts if not (processed_path / artifact).exists()
    )
    if missing_artifacts and not allow_missing:
        raise DatasetVersioningError(
            "processed dataset is missing expected artifacts: "
            f"{', '.join(missing_artifacts)}"
        )

    dvc_output = _read_dvc_output(dvc_metadata_file)
    return DatasetVersionReport(
        dvc_tracked_path=str(dvc_output.get("path", processed_path)),
        dvc_metadata_file=str(dvc_metadata_file),
        dvc_output_hash=str(dvc_output["hash"]),
        dvc_output_hash_name=dvc_output.get("hash_name"),
        generated_at_utc=generated_at_utc or datetime.now(UTC).isoformat(),
        expected_artifacts=expected_artifacts,
        missing_artifacts=missing_artifacts,
        git_commit_sha=_current_git_sha(Path(repo_dir)),
        remote_configured=_remote_is_configured(Path(repo_dir) / ".dvc" / "config"),
        snapshot_granularity_note=SNAPSHOT_GRANULARITY_NOTE,
        downstream_consumption_note=DOWNSTREAM_CONSUMPTION_NOTE,
        remote_note=NO_REMOTE_NOTE,
    )


def write_dataset_version_report(
    report: DatasetVersionReport,
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> DatasetVersionPaths:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    report_path = processed_path / DEFAULT_DATASET_VERSION_REPORT_FILENAME
    report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return DatasetVersionPaths(report_path=report_path)


def run_dataset_versioning_pipeline(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    dvc_metadata_path: Path | str = DEFAULT_DVC_METADATA_PATH,
    repo_dir: Path | str = Path("."),
    allow_missing: bool = False,
) -> DatasetVersionPaths:
    report = build_dataset_version_report(
        processed_dir=processed_dir,
        dvc_metadata_path=dvc_metadata_path,
        repo_dir=repo_dir,
        allow_missing=allow_missing,
    )
    return write_dataset_version_report(report, processed_dir)


def _read_dvc_output(dvc_metadata_path: Path) -> dict[str, str | None]:
    if not dvc_metadata_path.exists():
        raise DatasetVersioningError(f"DVC metadata file does not exist: {dvc_metadata_path}")

    metadata = yaml.safe_load(dvc_metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise DatasetVersioningError(f"DVC metadata file is not a mapping: {dvc_metadata_path}")

    outs = metadata.get("outs")
    if not isinstance(outs, list) or not outs:
        raise DatasetVersioningError(f"DVC metadata file has no outs entry: {dvc_metadata_path}")

    first_output = outs[0]
    if not isinstance(first_output, dict):
        raise DatasetVersioningError(f"DVC metadata output is invalid: {dvc_metadata_path}")

    output_hash = first_output.get("md5") or first_output.get("etag") or first_output.get("checksum")
    if not output_hash:
        raise DatasetVersioningError(f"DVC metadata output has no hash: {dvc_metadata_path}")

    return {
        "path": _normalize_dvc_output_path(dvc_metadata_path, first_output),
        "hash": str(output_hash),
        "hash_name": _detect_hash_name(first_output),
    }


def _normalize_dvc_output_path(dvc_metadata_path: Path, output: dict[str, Any]) -> str:
    output_path = Path(str(output.get("path", DEFAULT_PROCESSED_DIR)))
    if output_path.is_absolute():
        return output_path.as_posix()
    if dvc_metadata_path.parent.name == "data":
        return (dvc_metadata_path.parent / output_path).as_posix()
    return output_path.as_posix()


def _detect_hash_name(output: dict[str, Any]) -> str | None:
    for candidate in ("md5", "etag", "checksum"):
        if candidate in output:
            return candidate
    return None


def _current_git_sha(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _remote_is_configured(dvc_config_path: Path) -> bool:
    if not dvc_config_path.exists():
        return False
    config_text = dvc_config_path.read_text(encoding="utf-8")
    return "[remote " in config_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a dataset version report for the local DVC processed snapshot."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed artifact directory. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--dvc-metadata-path",
        type=Path,
        default=DEFAULT_DVC_METADATA_PATH,
        help=f"DVC metadata path. Defaults to {DEFAULT_DVC_METADATA_PATH}.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("."),
        help="Repository directory used to read the current git commit SHA.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the report even if expected processed artifacts are missing.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = run_dataset_versioning_pipeline(
        processed_dir=args.processed_dir,
        dvc_metadata_path=args.dvc_metadata_path,
        repo_dir=args.repo_dir,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(asdict(paths), indent=2, default=str))


if __name__ == "__main__":
    main()
