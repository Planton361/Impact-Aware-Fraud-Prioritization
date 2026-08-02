"""Semantic contract for successfully completed experiment runs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from fraud_detection.artifacts import _write_json

from ..config import EXPERIMENT_PROFILE_NAMES, EffectiveExperimentConfig

_SCHEMA = "fraud_detection.run_manifest.v1"
_MANIFEST_NAME = "RUN_MANIFEST.json"
_COMPLETED_PHASES = (
    "preflight",
    "inner_selection",
    "selection_freeze",
    "final_outer",
    "aggregation",
    "qa",
)
_ARTIFACT_GROUPS = frozenset(
    {
        "preflight",
        "inner_selection",
        "selection_freeze",
        "final_outer",
        "aggregation",
        "qa",
        "integrity",
    }
)
_SELECTION_COMPARISON_FILES = frozenset(
    {
        "inner_gain_comparison.csv",
        "selected_vs_nonselected_gain_inner_only.csv",
        "selection_status_summary.csv",
    }
)
_INTEGRITY_COMPARISON_FILES = frozenset(
    {
        "checksums.sha256",
        "run_identity_checks.csv",
    }
)


def _effective_config_payload(
    effective_config: EffectiveExperimentConfig,
) -> dict[str, object]:
    return effective_config.as_dict()


def _data_summary(
    data_metadata: Mapping[str, object],
    effective_config: EffectiveExperimentConfig,
) -> dict[str, object]:
    source_count_kind = (
        "generated"
        if effective_config.data_source_kind == "synthetic"
        else "raw"
    )
    summary: dict[str, object] = {
        "source_kind": effective_config.data_source_kind,
        "data_identity": data_metadata["deduplicated_dataframe_sha256"],
        "source_counts": {
            "kind": source_count_kind,
            "rows": int(data_metadata["rows_before_deduplication"]),
            "fraud": int(data_metadata["class_1_before"]),
            "legitimate": int(data_metadata["class_0_before"]),
        },
        "deduplicated_counts": {
            "rows": int(data_metadata["rows_after_deduplication"]),
            "fraud": int(data_metadata["class_1_after"]),
            "legitimate": int(data_metadata["class_0_after"]),
        },
        "removed_duplicate_count": int(
            data_metadata["rows_removed_as_duplicate_followups"]
        ),
    }
    if effective_config.data_source_kind == "synthetic":
        summary["synthetic"] = {
            "generator_schema": data_metadata["synthetic_generator_schema"],
            "generation_seed": data_metadata["synthetic_generation_seed"],
            "requested_row_count": data_metadata[
                "synthetic_requested_row_count"
            ],
        }
    return summary


def _artifact_group(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    root = path.parts[0]
    if root == "preflight":
        return "preflight"
    if root == "inner_validation":
        return "inner_selection"
    if root == "selection_freeze":
        return "selection_freeze"
    if root == "final_outer_run":
        return "final_outer"
    if root in {"tables", "diagnostics", "figure_data"}:
        return "aggregation"
    if root == "comparison":
        if path.name == "final_qa.json":
            return "qa"
        if path.name in _INTEGRITY_COMPARISON_FILES:
            return "integrity"
        if path.name in _SELECTION_COMPARISON_FILES:
            return "selection_freeze"
        return "aggregation"
    if root == "logs":
        return "integrity"
    raise RuntimeError(f"No semantic artifact group for {relative_path!r}.")


def _artifact_format(relative_path: str) -> str:
    name = PurePosixPath(relative_path).name
    if name.endswith(".csv.gz"):
        return "csv.gz"
    suffix = PurePosixPath(name).suffix.removeprefix(".")
    if not suffix:
        raise RuntimeError(f"Artifact has no file format: {relative_path!r}.")
    return suffix


def _excluded_artifact(path: PurePosixPath) -> bool:
    return (
        path.name == _MANIFEST_NAME
        or any(part.startswith(".") for part in path.parts)
        or "__pycache__" in path.parts
        or path.suffix in {".tmp", ".lock", ".pyc"}
        or path.name.endswith("~")
    )


def _produced_artifacts(output_root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(output_root).as_posix())
        if _excluded_artifact(relative):
            continue
        relative_text = relative.as_posix()
        entries.append(
            {
                "path": relative_text,
                "group": _artifact_group(relative_text),
                "format": _artifact_format(relative_text),
            }
        )
    return sorted(entries, key=lambda entry: entry["path"])


def _build_run_manifest(
    *,
    effective_config: EffectiveExperimentConfig,
    data_metadata: Mapping[str, object],
    produced_artifacts: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "status": "COMPLETE",
        "profile": effective_config.profile_name,
        "evidence_classification": effective_config.evidence_classification,
        "completed_phases": list(_COMPLETED_PHASES),
        "effective_config": _effective_config_payload(effective_config),
        "data_summary": _data_summary(data_metadata, effective_config),
        "produced_artifacts": produced_artifacts,
    }


def _finite_manifest_numbers(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_manifest_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_manifest_numbers(item) for item in value)
    return False


def _nonnegative_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"RUN_MANIFEST.json has invalid {name}.")
    return value


def _validate_data_summary(
    summary: object,
    effective_config: EffectiveExperimentConfig,
) -> None:
    if not isinstance(summary, Mapping):
        raise RuntimeError("RUN_MANIFEST.json data_summary is invalid.")
    required_keys = {
        "source_kind",
        "data_identity",
        "source_counts",
        "deduplicated_counts",
        "removed_duplicate_count",
    }
    expected_keys = required_keys | (
        {"synthetic"}
        if effective_config.data_source_kind == "synthetic"
        else set()
    )
    if set(summary) != expected_keys:
        raise RuntimeError("RUN_MANIFEST.json data_summary fields are invalid.")
    if summary["source_kind"] != effective_config.data_source_kind:
        raise RuntimeError("RUN_MANIFEST.json data source kind is incorrect.")
    identity = summary["data_identity"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise RuntimeError("RUN_MANIFEST.json data identity is invalid.")

    source = summary["source_counts"]
    deduplicated = summary["deduplicated_counts"]
    if not isinstance(source, Mapping) or set(source) != {
        "kind",
        "rows",
        "fraud",
        "legitimate",
    }:
        raise RuntimeError("RUN_MANIFEST.json source counts are invalid.")
    if not isinstance(deduplicated, Mapping) or set(deduplicated) != {
        "rows",
        "fraud",
        "legitimate",
    }:
        raise RuntimeError("RUN_MANIFEST.json deduplicated counts are invalid.")
    expected_count_kind = (
        "generated"
        if effective_config.data_source_kind == "synthetic"
        else "raw"
    )
    if source["kind"] != expected_count_kind:
        raise RuntimeError("RUN_MANIFEST.json source count kind is invalid.")
    source_rows = _nonnegative_count(source["rows"], "source row count")
    source_fraud = _nonnegative_count(source["fraud"], "source Fraud count")
    source_legitimate = _nonnegative_count(
        source["legitimate"], "source legitimate count"
    )
    deduplicated_rows = _nonnegative_count(
        deduplicated["rows"], "deduplicated row count"
    )
    deduplicated_fraud = _nonnegative_count(
        deduplicated["fraud"], "deduplicated Fraud count"
    )
    deduplicated_legitimate = _nonnegative_count(
        deduplicated["legitimate"], "deduplicated legitimate count"
    )
    removed = _nonnegative_count(
        summary["removed_duplicate_count"], "removed duplicate count"
    )
    if (
        source_fraud + source_legitimate != source_rows
        or deduplicated_fraud + deduplicated_legitimate != deduplicated_rows
        or source_rows - removed != deduplicated_rows
        or source_fraud < deduplicated_fraud
        or source_legitimate < deduplicated_legitimate
        or (source_fraud - deduplicated_fraud)
        + (source_legitimate - deduplicated_legitimate)
        != removed
    ):
        raise RuntimeError("RUN_MANIFEST.json data-count arithmetic is invalid.")

    if effective_config.data_source_kind == "synthetic":
        synthetic = summary["synthetic"]
        if not isinstance(synthetic, Mapping) or set(synthetic) != {
            "generator_schema",
            "generation_seed",
            "requested_row_count",
        }:
            raise RuntimeError("RUN_MANIFEST.json synthetic metadata is invalid.")
        if (
            not isinstance(synthetic["generator_schema"], str)
            or not synthetic["generator_schema"]
            or synthetic["generation_seed"]
            != effective_config.synthetic_generation_seed
            or synthetic["requested_row_count"]
            != effective_config.synthetic_row_target
        ):
            raise RuntimeError("RUN_MANIFEST.json synthetic metadata is incorrect.")


def _validate_artifact_inventory(
    artifacts: object,
    available_artifact_paths: frozenset[str],
) -> None:
    if not isinstance(artifacts, list):
        raise RuntimeError("RUN_MANIFEST.json artifact inventory is invalid.")
    paths: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "group",
            "format",
        }:
            raise RuntimeError("RUN_MANIFEST.json artifact entry is invalid.")
        path = entry["path"]
        if not isinstance(path, str):
            raise RuntimeError("RUN_MANIFEST.json artifact path is invalid.")
        pure_path = PurePosixPath(path)
        if (
            not path
            or pure_path.is_absolute()
            or PureWindowsPath(path).is_absolute()
            or ".." in pure_path.parts
            or pure_path.as_posix() != path
            or _excluded_artifact(pure_path)
        ):
            raise RuntimeError(f"RUN_MANIFEST.json has unsafe artifact path {path!r}.")
        if entry["group"] not in _ARTIFACT_GROUPS:
            raise RuntimeError("RUN_MANIFEST.json artifact group is invalid.")
        if not isinstance(entry["format"], str) or not entry["format"]:
            raise RuntimeError("RUN_MANIFEST.json artifact format is invalid.")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError(
            "RUN_MANIFEST.json artifact paths must be sorted and unique."
        )
    if frozenset(paths) != available_artifact_paths:
        raise RuntimeError(
            "RUN_MANIFEST.json artifact inventory does not match produced files."
        )


def _validate_run_manifest(
    manifest: Mapping[str, object],
    *,
    effective_config: EffectiveExperimentConfig,
    available_artifact_paths: frozenset[str],
) -> None:
    if set(manifest) != {
        "schema",
        "status",
        "profile",
        "evidence_classification",
        "completed_phases",
        "effective_config",
        "data_summary",
        "produced_artifacts",
    }:
        raise RuntimeError("RUN_MANIFEST.json top-level fields are invalid.")
    if not _finite_manifest_numbers(manifest):
        raise RuntimeError("RUN_MANIFEST.json contains non-finite numbers.")
    if manifest["schema"] != _SCHEMA:
        raise RuntimeError("RUN_MANIFEST.json schema is invalid.")
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("RUN_MANIFEST.json status is not COMPLETE.")
    if manifest["profile"] not in EXPERIMENT_PROFILE_NAMES:
        raise RuntimeError("RUN_MANIFEST.json profile is unknown.")
    if manifest["profile"] != effective_config.profile_name:
        raise RuntimeError("RUN_MANIFEST.json profile is incorrect.")
    if (
        manifest["evidence_classification"]
        != effective_config.evidence_classification
    ):
        raise RuntimeError(
            "RUN_MANIFEST.json evidence classification is incorrect."
        )
    if manifest["completed_phases"] != list(_COMPLETED_PHASES):
        raise RuntimeError("RUN_MANIFEST.json completed phases are incomplete.")
    if manifest["effective_config"] != _effective_config_payload(
        effective_config
    ):
        raise RuntimeError("RUN_MANIFEST.json effective configuration is incorrect.")
    _validate_data_summary(manifest["data_summary"], effective_config)
    _validate_artifact_inventory(
        manifest["produced_artifacts"], available_artifact_paths
    )


def write_completed_run_manifest(
    *,
    output_root: Path,
    effective_config: EffectiveExperimentConfig,
    data_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate and atomically write the completed-run semantic manifest."""

    artifacts = _produced_artifacts(output_root)
    manifest = _build_run_manifest(
        effective_config=effective_config,
        data_metadata=data_metadata,
        produced_artifacts=artifacts,
    )
    available_paths = frozenset(entry["path"] for entry in artifacts)
    _validate_run_manifest(
        manifest,
        effective_config=effective_config,
        available_artifact_paths=available_paths,
    )

    manifest_path = output_root / _MANIFEST_NAME
    temporary_path = output_root / f".{_MANIFEST_NAME}.tmp"
    if manifest_path.exists() or temporary_path.exists():
        raise FileExistsError("Refusing to overwrite RUN_MANIFEST.json.")
    try:
        _write_json(temporary_path, manifest)
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest
