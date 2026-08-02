"""Build final Chapter-5 presentation data from one completed experiment run.

This is the middle layer of the presentation architecture. It checks the
registered experiment files, validates their semantic schemas, and performs
presentation-only aggregation.  It never imports a runner or model, creates a
score, selects a parameter, or changes a ranking.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from fraud_detection.experiment.config import (
    EXPECTED_DEDUPLICATED_SHA256,
    EXPECTED_RAW_SHA256,
    EXPERIMENT_PROFILE_NAMES,
    DataSourceKind,
    EffectiveExperimentConfig,
    EvidenceClassification,
    ExperimentProfileName,
    resolve_experiment_profile,
)

from .. import (
    METHOD_BCE,
    METHOD_FIXED,
    METHOD_ORDER,
    file_inventory,
    prepare_output_directory,
    require_generated_path,
    sha256_file,
    write_csv,
    write_json,
)
from ..catalog import (
    CANONICAL_ARTIFACT_IDS,
    ENGINEERING_ARTIFACT_IDS,
    build_profile_selection_registry,
)
from .derivations import (
    CANONICAL_OUTPUT_PATHS,
    ENGINEERING_OUTPUT_PATHS,
    derive_all,
    derive_engineering,
)

_EventSink = Callable[[str, Mapping[str, object]], None]
RUN_MANIFEST_SCHEMA = "fraud_detection.run_manifest.v1"
RUN_MANIFEST_NAME = "RUN_MANIFEST.json"
CHECKSUM_MANIFEST_PATH = "comparison/checksums.sha256"
COMPLETED_PHASES = (
    "preflight",
    "inner_selection",
    "selection_freeze",
    "final_outer",
    "aggregation",
    "qa",
)
ARTIFACT_GROUPS = frozenset(
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
EXPECTED_PREFLIGHT_SCHEMA = "ranker_gain_validation.preflight.v2"
EXPECTED_FINAL_SCHEMA = "ranker_gain_validation.final_outer.v1"
EXPECTED_PRESENTATION_FRAMES = 29
EXPECTED_ENGINEERING_FRAMES = 2
_REAL_SOURCE_COUNTS = (284_807, 492, 284_315)
_REAL_DEDUPLICATED_COUNTS = (283_726, 473, 283_253)
_REAL_REMOVED_DUPLICATE_COUNT = 1_081
_REAL_PREFLIGHT_FIELDS = frozenset(
    {
        "raw_path",
        "expected_raw_sha256",
        "actual_raw_sha256",
        "rows",
        "class_0",
        "class_1",
        "expected_deduplicated_dataframe_sha256",
        "actual_deduplicated_dataframe_sha256",
    }
)
_SYNTHETIC_PREFLIGHT_FIELDS = frozenset(
    {
        "data_source_kind",
        "synthetic_generator_schema",
        "synthetic_generation_seed",
        "synthetic_requested_row_count",
        "synthetic_generated_row_count",
        "synthetic_generated_fraud_count",
        "synthetic_generated_legitimate_count",
        "deterministic_data_identity",
        "evidence_classification",
    }
)
PresentationRole = Literal["canonical", "engineering"]


def _presentation_role(
    profile: ExperimentProfileName,
    evidence_classification: EvidenceClassification,
) -> PresentationRole:
    if profile == "canonical":
        if evidence_classification != "thesis-evidentiary":
            raise RuntimeError("Canonical presentation evidence metadata is missing.")
        return "canonical"
    if profile in {"mini-real", "smoke-synthetic"}:
        if evidence_classification == "thesis-evidentiary":
            raise RuntimeError("Engineering presentation evidence metadata is missing.")
        return "engineering"
    raise RuntimeError(f"Unsupported presentation profile: {profile!r}.")


@dataclass(frozen=True, slots=True)
class PresentationInputContext:
    """Immutable semantic and file-integrity contract for one completed run."""

    profile: ExperimentProfileName
    presentation_role: PresentationRole
    evidence_classification: EvidenceClassification
    data_source_kind: DataSourceKind
    seeds: tuple[int, ...]
    target_budgets: tuple[int, ...]
    primary_budgets: tuple[int, ...]
    candidate_pool_size: int
    inner_folds: int
    bce_oof_folds: int
    data_summary: Mapping[str, object]
    registered_artifact_paths: frozenset[str]
    artifact_checksums: Mapping[str, str]
    experiment_root: Path
    effective_config: EffectiveExperimentConfig


@dataclass(frozen=True, slots=True)
class _PreflightDataSummary:
    source_kind: DataSourceKind
    source_rows: int
    source_fraud: int
    source_legitimate: int
    deduplicated_rows: int
    deduplicated_fraud: int
    deduplicated_legitimate: int
    removed_duplicate_count: int
    data_identity: str
    synthetic_generator_schema: str | None = None
    synthetic_generation_seed: int | None = None
    synthetic_requested_row_count: int | None = None


def _status_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _emit_status(
    event_sink: _EventSink | None,
    level: str,
    event: str,
    **fields: object,
) -> None:
    if event_sink is not None:
        event_sink(
            "status",
            {
                "level": level,
                "event": event,
                "fields": fields,
            },
        )












































def _finite_json_numbers(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_json_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json_numbers(item) for item in value)
    return False


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    return value


def _nonnegative_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"RUN_MANIFEST.json has invalid {name}.")
    return value


def _validate_data_summary(
    summary: object,
    effective: EffectiveExperimentConfig,
) -> Mapping[str, object]:
    if not isinstance(summary, Mapping):
        raise RuntimeError("RUN_MANIFEST.json data summary is invalid.")
    required = {
        "source_kind",
        "data_identity",
        "source_counts",
        "deduplicated_counts",
        "removed_duplicate_count",
    }
    expected = required | (
        {"synthetic"} if effective.data_source_kind == "synthetic" else set()
    )
    if set(summary) != expected:
        raise RuntimeError("RUN_MANIFEST.json data summary fields are invalid.")
    if summary["source_kind"] != effective.data_source_kind:
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
    expected_kind = (
        "generated" if effective.data_source_kind == "synthetic" else "raw"
    )
    if source["kind"] != expected_kind:
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

    if effective.data_source_kind == "synthetic":
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
            != effective.synthetic_generation_seed
            or synthetic["requested_row_count"]
            != effective.synthetic_row_target
        ):
            raise RuntimeError("RUN_MANIFEST.json synthetic metadata is incorrect.")
    return summary


def _excluded_run_artifact(path: PurePosixPath) -> bool:
    return (
        path.name == RUN_MANIFEST_NAME
        or any(part.startswith(".") for part in path.parts)
        or "__pycache__" in path.parts
        or path.suffix in {".tmp", ".lock", ".pyc"}
        or path.name.endswith("~")
    )


def _safe_manifest_artifact_path(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("RUN_MANIFEST.json artifact path is invalid.")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
        or path.as_posix() != value
        or _excluded_run_artifact(path)
    ):
        raise RuntimeError(
            f"RUN_MANIFEST.json has unsafe artifact path {value!r}."
        )
    return value


def _artifact_target(root: Path, relative: str) -> Path:
    target = (root / PurePosixPath(relative)).resolve()
    if root not in target.parents:
        raise RuntimeError(f"Experiment artifact escapes root: {relative}")
    return target


def _actual_artifact_paths(root: Path) -> frozenset[str]:
    paths: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _excluded_run_artifact(relative):
            continue
        if root not in path.resolve().parents:
            raise RuntimeError(
                f"Experiment artifact escapes root: {relative.as_posix()}"
            )
        paths.add(relative.as_posix())
    return frozenset(paths)


def _validate_artifact_inventory(
    root: Path,
    entries: object,
) -> frozenset[str]:
    if not isinstance(entries, list):
        raise RuntimeError("RUN_MANIFEST.json artifact inventory is invalid.")
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "group",
            "format",
        }:
            raise RuntimeError("RUN_MANIFEST.json artifact entry is invalid.")
        relative = _safe_manifest_artifact_path(entry["path"])
        group = entry["group"]
        if not isinstance(group, str) or group not in ARTIFACT_GROUPS:
            raise RuntimeError("RUN_MANIFEST.json artifact group is invalid.")
        if not isinstance(entry["format"], str) or not entry["format"]:
            raise RuntimeError("RUN_MANIFEST.json artifact format is invalid.")
        target = _artifact_target(root, relative)
        if not target.is_file():
            raise FileNotFoundError(
                f"Registered experiment file is missing: {relative}"
            )
        paths.append(relative)
    if len(paths) != len(set(paths)):
        raise RuntimeError("RUN_MANIFEST.json contains duplicate artifact paths.")
    if paths != sorted(paths):
        raise RuntimeError("RUN_MANIFEST.json artifact paths must be sorted.")
    registered = frozenset(paths)
    if registered != _actual_artifact_paths(root):
        raise RuntimeError(
            "RUN_MANIFEST.json artifact inventory does not match produced files."
        )
    return registered


def _read_run_manifest(root: Path) -> dict[str, object]:
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing RUN_MANIFEST.json: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (JSONDecodeError, UnicodeError) as exc:
        raise RuntimeError("RUN_MANIFEST.json is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("RUN_MANIFEST.json must contain one JSON object.")
    return value


def _validate_run_manifest(
    root: Path,
    manifest: Mapping[str, object],
) -> tuple[
    EffectiveExperimentConfig,
    Mapping[str, object],
    frozenset[str],
]:
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
    if not _finite_json_numbers(manifest):
        raise RuntimeError("RUN_MANIFEST.json contains non-finite numbers.")
    if manifest["schema"] != RUN_MANIFEST_SCHEMA:
        raise RuntimeError("RUN_MANIFEST.json schema is invalid.")
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("RUN_MANIFEST.json status is not COMPLETE.")
    if manifest["completed_phases"] != list(COMPLETED_PHASES):
        raise RuntimeError("RUN_MANIFEST.json completed phases are incomplete.")
    profile = manifest["profile"]
    if not isinstance(profile, str) or profile not in EXPERIMENT_PROFILE_NAMES:
        raise RuntimeError("RUN_MANIFEST.json profile is unknown.")
    effective = resolve_experiment_profile(profile)
    if manifest["evidence_classification"] != effective.evidence_classification:
        raise RuntimeError(
            "RUN_MANIFEST.json evidence classification does not match profile."
        )
    if manifest["effective_config"] != effective.as_dict():
        raise RuntimeError(
            "RUN_MANIFEST.json effective configuration does not match profile."
        )
    summary = _validate_data_summary(manifest["data_summary"], effective)
    registered = _validate_artifact_inventory(
        root, manifest["produced_artifacts"]
    )
    return effective, summary, registered


def _read_checksum_manifest(
    root: Path,
    registered_artifacts: frozenset[str],
) -> dict[str, str]:
    if CHECKSUM_MANIFEST_PATH not in registered_artifacts:
        raise RuntimeError(
            "RUN_MANIFEST.json does not register comparison/checksums.sha256."
        )
    manifest = _artifact_target(root, CHECKSUM_MANIFEST_PATH)
    registered: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise RuntimeError(
            "comparison/checksums.sha256 is not valid UTF-8 text."
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        digest = parts[0].lower() if parts else ""
        if (
            len(parts) != 2
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"Malformed checksum line {line_number}: {manifest}"
            )
        relative = parts[1].lstrip("*").replace("\\", "/")
        _safe_manifest_artifact_path(relative)
        if relative not in registered_artifacts:
            raise RuntimeError(
                "Checksum manifest references unregistered experiment file: "
                f"{relative}"
            )
        if relative in registered:
            raise RuntimeError(
                f"Duplicate checksum registration for experiment file: {relative}"
            )
        target = _artifact_target(root, relative)
        if not target.is_file():
            raise FileNotFoundError(
                f"Registered experiment file is missing: {relative}"
            )
        observed = sha256_file(target)
        if observed != digest:
            raise RuntimeError(f"Experiment checksum mismatch: {relative}")
        registered[relative] = digest
    if not registered:
        raise RuntimeError("Completed-run checksum manifest is empty.")
    return registered


def _required_presentation_inputs(
    effective: EffectiveExperimentConfig,
    registered_artifacts: frozenset[str],
) -> tuple[str, ...]:
    required = [
        CHECKSUM_MANIFEST_PATH,
        "preflight/preflight_validation.json",
        "final_outer_run/final_outer_manifest.json",
        "figure_data/all_budget_matched_results.csv",
        "diagnostics/global_metrics_seedwise.csv",
    ]
    for seed in effective.seeds:
        parquet = f"final_outer_run/seed_{seed}/ranking_dump.parquet"
        csv = f"final_outer_run/seed_{seed}/ranking_dump.csv"
        if parquet in registered_artifacts:
            required.append(parquet)
        elif csv in registered_artifacts:
            required.append(csv)
        else:
            raise RuntimeError(
                "RUN_MANIFEST.json lacks a complete-ranking artifact for "
                f"configured seed {seed}."
            )
    for relative in required:
        if relative not in registered_artifacts:
            raise RuntimeError(
                "RUN_MANIFEST.json lacks required presentation input: "
                f"{relative}"
            )
    return tuple(required)


def load_presentation_input_context(
    experiment_root: Path,
) -> PresentationInputContext:
    """Validate one completed run without creating presentation output."""

    root = experiment_root.resolve()
    manifest = _read_run_manifest(root)
    effective, data_summary, registered_artifacts = _validate_run_manifest(
        root, manifest
    )
    required = _required_presentation_inputs(effective, registered_artifacts)
    checksums = _read_checksum_manifest(root, registered_artifacts)
    for relative in required:
        if relative != CHECKSUM_MANIFEST_PATH and relative not in checksums:
            raise RuntimeError(
                "Required presentation input lacks checksum registration: "
                f"{relative}"
            )
    frozen_summary = _freeze_json(data_summary)
    if not isinstance(frozen_summary, Mapping):
        raise RuntimeError("RUN_MANIFEST.json data summary is invalid.")
    return PresentationInputContext(
        profile=effective.profile_name,
        presentation_role=_presentation_role(
            effective.profile_name,
            effective.evidence_classification,
        ),
        evidence_classification=effective.evidence_classification,
        data_source_kind=effective.data_source_kind,
        seeds=effective.seeds,
        target_budgets=effective.target_budgets,
        primary_budgets=effective.primary_budgets,
        candidate_pool_size=effective.candidate_pool_size,
        inner_folds=effective.inner_folds,
        bce_oof_folds=effective.bce_oof_folds,
        data_summary=frozen_summary,
        registered_artifact_paths=registered_artifacts,
        artifact_checksums=MappingProxyType(dict(checksums)),
        experiment_root=root,
        effective_config=effective,
    )


class ExperimentStore:
    """Manifest-registered and checksum-gated completed-run reader."""

    def __init__(self, context: PresentationInputContext) -> None:
        self.root = context.experiment_root
        self.artifact_paths = context.registered_artifact_paths
        self.registered = context.artifact_checksums
        self.read_paths: set[str] = {CHECKSUM_MANIFEST_PATH}

    def path(self, relative: str) -> Path:
        normalized = relative.replace("\\", "/")
        _safe_manifest_artifact_path(normalized)
        if normalized not in self.artifact_paths:
            raise RuntimeError(
                f"Unregistered experiment input rejected: {normalized}"
            )
        path = _artifact_target(self.root, normalized)
        if not path.is_file():
            raise FileNotFoundError(
                f"Registered experiment file is missing: {normalized}"
            )
        if normalized != CHECKSUM_MANIFEST_PATH:
            expected = self.registered.get(normalized)
            if expected is None:
                raise RuntimeError(
                    "Registered experiment input lacks a checksum: "
                    f"{normalized}"
                )
            if sha256_file(path) != expected:
                raise RuntimeError(f"Experiment checksum mismatch: {normalized}")
        self.read_paths.add(normalized)
        return path

    def csv(self, relative: str) -> pd.DataFrame:
        return pd.read_csv(self.path(relative))

    def json(self, relative: str) -> dict[str, Any]:
        try:
            value = json.loads(
                self.path(relative).read_text(encoding="utf-8-sig")
            )
        except (JSONDecodeError, UnicodeError) as exc:
            raise RuntimeError(
                f"Registered experiment JSON is invalid: {relative}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected JSON object: {relative}")
        return value

    def ranking(self, seed: int) -> pd.DataFrame:
        parquet = f"final_outer_run/seed_{seed}/ranking_dump.parquet"
        csv = f"final_outer_run/seed_{seed}/ranking_dump.csv"
        columns = [
            "seed",
            "target_budget",
            "method_family",
            "score_path",
            "row_index",
            "original_position",
            "candidate_flag",
            "candidate_pool_size",
            "candidate_pool_sha256",
            "p_fraud",
            "raw_ranker_score",
            "final_rank_position",
            "bce_rank_position",
            "priority_order_score",
            "Class",
            "Amount",
            "selected_gain",
            "selection_status",
            "truncation_level",
            "final_n_estimators",
            "score_type",
        ]
        if parquet in self.artifact_paths:
            return pd.read_parquet(self.path(parquet), columns=columns)
        if csv in self.artifact_paths:
            return self.csv(csv)
        raise FileNotFoundError(
            f"No registered full-ranking dump for seed {seed}"
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def _require_finite(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} contains non-finite numerical values.")


def _validate_ranking_score_semantics(
    frame: pd.DataFrame,
    *,
    seed: int,
    budget: int,
    method: str,
    expected_pool_size: int,
) -> int:
    identity = f"seed={seed}, budget={budget}, path={method}"
    global_numeric = (
        "seed",
        "target_budget",
        "row_index",
        "original_position",
        "candidate_pool_size",
        "p_fraud",
        "final_rank_position",
        "bce_rank_position",
        "priority_order_score",
        "Class",
        "Amount",
        "truncation_level",
        "final_n_estimators",
    )
    required = {*global_numeric, "candidate_flag", "raw_ranker_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Ranking dump schema is incomplete for {identity}: missing={missing}."
        )
    flags_column = frame["candidate_flag"]
    if not pd.api.types.is_bool_dtype(flags_column.dtype) or flags_column.isna().any():
        raise RuntimeError(
            f"Ranking candidate mask is invalid for {identity}, "
            f"column=candidate_flag."
        )
    for column in global_numeric:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            raise RuntimeError(
                f"Ranking globally required numeric column is invalid for "
                f"{identity}, column={column}."
            )
        values = series.to_numpy(dtype=float, na_value=np.nan)
        invalid_count = int((~np.isfinite(values)).sum())
        if invalid_count:
            raise RuntimeError(
                f"Ranking globally required numeric value is non-finite for "
                f"{identity}, column={column}, count={invalid_count}."
            )

    flags = flags_column.to_numpy(dtype=bool)
    candidate_count = int(flags.sum())
    non_candidate_count = int((~flags).sum())
    expected_non_candidates = len(frame) - expected_pool_size
    if (
        candidate_count != expected_pool_size
        or non_candidate_count != expected_non_candidates
    ):
        raise RuntimeError(
            f"Ranking candidate count differs for {identity}, "
            f"candidate_count={candidate_count}, expected={expected_pool_size}, "
            f"non_candidate_count={non_candidate_count}, "
            f"expected_non_candidates={expected_non_candidates}."
        )
    if not frame["candidate_pool_size"].astype(int).eq(expected_pool_size).all():
        raise RuntimeError(
            f"Ranking declared candidate-pool size differs for {identity}, "
            f"column=candidate_pool_size."
        )

    raw_column = frame["raw_ranker_score"]
    if not pd.api.types.is_numeric_dtype(raw_column.dtype):
        raise RuntimeError(
            f"Ranking raw score column is not numeric for {identity}, "
            "column=raw_ranker_score."
        )
    raw = raw_column.to_numpy(dtype=float, na_value=np.nan)
    infinity_count = int(np.isinf(raw).sum())
    if infinity_count:
        raise RuntimeError(
            f"Ranking raw score contains infinity for {identity}, "
            f"column=raw_ranker_score, count={infinity_count}."
        )
    if method == METHOD_BCE:
        missing_count = int(np.isnan(raw).sum())
        if missing_count:
            raise RuntimeError(
                f"Ranking path has an unsupported raw-score missingness pattern "
                f"for {identity}, column=raw_ranker_score, "
                f"count={missing_count}."
            )
        return 0
    if method not in METHOD_ORDER:
        raise RuntimeError(
            f"Ranking path has unsupported raw-score semantics for {identity}."
        )
    missing_candidate_count = int(np.isnan(raw[flags]).sum())
    if missing_candidate_count:
        raise RuntimeError(
            f"Ranking candidate raw score is missing for {identity}, "
            f"column=raw_ranker_score, count={missing_candidate_count}."
        )
    finite_non_candidate_count = int(np.isfinite(raw[~flags]).sum())
    if finite_non_candidate_count:
        raise RuntimeError(
            f"Ranking non-candidate raw score is unexpectedly finite for "
            f"{identity}, column=raw_ranker_score, "
            f"count={finite_non_candidate_count}."
        )
    return int(np.isnan(raw[~flags]).sum())


def _ordered_seed_metrics(
    frame: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    seed_order = {seed: index for index, seed in enumerate(seeds)}
    budget_order = {budget: index for index, budget in enumerate(budgets)}
    method_order = {
        method: index for index, method in enumerate(METHOD_ORDER)
    }
    result = frame.copy()
    result["__seed_order"] = result["seed"].astype(int).map(seed_order)
    result["__budget_order"] = (
        result["target_budget"].astype(int).map(budget_order)
    )
    result["__method_order"] = result["method_family"].map(method_order)
    if result[
        ["__seed_order", "__budget_order", "__method_order"]
    ].isna().any().any():
        raise RuntimeError(
            "Unexpected seed, budget, or path in experiment metrics."
        )
    return (
        result.sort_values(
            ["__budget_order", "__method_order", "__seed_order"],
            kind="mergesort",
        )
        .drop(
            columns=["__seed_order", "__budget_order", "__method_order"]
        )
        .reset_index(drop=True)
    )


def _validate_seed_metrics(
    frame: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    _require_columns(
        frame,
        [
            "seed",
            "target_budget",
            "method_family",
            "score_path",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "precision_at_k",
            "recall_at_k",
            "fraud_amount_sum_at_k",
            "amount_ndcg_at_k",
            "q90_captured_fraud_count",
            "q90_captured_ratio_at_k",
            "selected_gain",
            "selection_status",
        ],
        "all-budget matched results",
    )
    _require_finite(
        frame,
        [
            "seed",
            "target_budget",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "precision_at_k",
            "recall_at_k",
            "fraud_amount_sum_at_k",
            "amount_ndcg_at_k",
            "q90_captured_fraud_count",
            "q90_captured_ratio_at_k",
        ],
        "all-budget matched results",
    )
    selected = frame.loc[
        frame["seed"].astype(int).isin(seeds)
        & frame["target_budget"].astype(int).isin(budgets)
    ].copy()
    expected = {
        (seed, budget, method)
        for seed in seeds
        for budget in budgets
        for method in METHOD_ORDER
    }
    observed = set(
        selected[["seed", "target_budget", "method_family"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected or selected.duplicated(
        ["seed", "target_budget", "method_family"]
    ).any():
        raise RuntimeError(
            "Seed/budget/path coverage differs from the complete requested grid."
        )
    for row in selected.itertuples(index=False):
        method = str(row.method_family)
        budget = int(row.target_budget)
        expected_path = (
            method if method == METHOD_BCE else f"{method}_k{budget}"
        )
        if str(row.score_path) != expected_path:
            raise RuntimeError(
                f"Score path does not identify its budget model: "
                f"{row.score_path}"
            )
    return _ordered_seed_metrics(selected, seeds, budgets)


def _load_ranking_groups(
    store: ExperimentStore,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    expected_pool_size: int,
) -> dict[tuple[int, int, str], pd.DataFrame]:
    groups: dict[tuple[int, int, str], pd.DataFrame] = {}
    required = [
        "seed",
        "target_budget",
        "method_family",
        "score_path",
        "row_index",
        "candidate_flag",
        "candidate_pool_size",
        "candidate_pool_sha256",
        "p_fraud",
        "raw_ranker_score",
        "final_rank_position",
        "bce_rank_position",
        "priority_order_score",
        "Class",
        "Amount",
    ]
    for seed in seeds:
        dump = store.ranking(seed)
        _require_columns(dump, required, f"ranking dump seed {seed}")
        _require_finite(
            dump,
            [
                "seed",
                "target_budget",
            ],
            f"ranking dump seed {seed}",
        )
        dump = dump.loc[
            dump["target_budget"].astype(int).isin(budgets)
        ].copy()
        for (budget, method), group in dump.groupby(
            ["target_budget", "method_family"],
            sort=False,
        ):
            key = (seed, int(budget), str(method))
            if key in groups:
                raise RuntimeError(f"Duplicate ranking group: {key}")
            _validate_ranking_score_semantics(
                group,
                seed=seed,
                budget=int(budget),
                method=str(method),
                expected_pool_size=expected_pool_size,
            )
            ordered = group.sort_values(
                "final_rank_position", kind="mergesort"
            ).reset_index(drop=True)
            positions = ordered["final_rank_position"].to_numpy(int)
            if not np.array_equal(
                positions, np.arange(1, len(ordered) + 1)
            ):
                raise RuntimeError(f"Invalid full ranking positions: {key}")
            if ordered["row_index"].duplicated().any():
                raise RuntimeError(f"Duplicate row_index in ranking: {key}")
            groups[key] = ordered
    expected = {
        (seed, budget, method)
        for seed in seeds
        for budget in budgets
        for method in METHOD_ORDER
    }
    if set(groups) != expected:
        raise RuntimeError(
            "Full-ranking coverage mismatch: "
            f"missing={sorted(expected - set(groups))}, "
            f"extra={sorted(set(groups) - expected)}"
        )
    _validate_ranking_groups(groups, seeds, budgets)
    return groups


def _validate_ranking_groups(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> None:
    """Validate common case identity and registered path identity per seed."""

    for seed in seeds:
        reference = groups[(seed, budgets[0], METHOD_BCE)].sort_values(
            "row_index", kind="mergesort"
        )
        reference_ids = reference["row_index"].to_numpy(int)
        reference_cases = reference[
            ["Class", "Amount", "p_fraud"]
        ].reset_index(drop=True)
        for budget in budgets:
            for method in METHOD_ORDER:
                key = (seed, budget, method)
                ranking = groups[key]
                if not ranking["seed"].astype(int).eq(seed).all():
                    raise RuntimeError(
                        f"Ranking seed identity mismatch: {key}"
                    )
                if not ranking["target_budget"].astype(int).eq(budget).all():
                    raise RuntimeError(
                        f"Ranking budget identity mismatch: {key}"
                    )
                if not ranking["method_family"].astype(str).eq(method).all():
                    raise RuntimeError(
                        f"Ranking method identity mismatch: {key}"
                    )
                expected_path = (
                    method if method == METHOD_BCE else f"{method}_k{budget}"
                )
                observed_paths = ranking["score_path"].dropna().astype(str).unique()
                if (
                    len(observed_paths) != 1
                    or str(observed_paths[0]) != expected_path
                ):
                    raise RuntimeError(
                        "Ranking score path does not identify its budget model: "
                        f"{key}, observed={observed_paths.tolist()}"
                    )
                current = ranking.sort_values(
                    "row_index", kind="mergesort"
                ).reset_index(drop=True)
                if not np.array_equal(
                    current["row_index"].to_numpy(int), reference_ids
                ):
                    raise RuntimeError(
                        f"Full-ranking row-index universe differs: {key}"
                    )
                for column in ("Class", "Amount", "p_fraud"):
                    if not current[column].equals(reference_cases[column]):
                        raise RuntimeError(
                            "Immutable case value differs across ranking paths: "
                            f"{key}, column={column}"
                        )

        bce_orders = {
            tuple(
                groups[(seed, budget, METHOD_BCE)]["row_index"].astype(int)
            )
            for budget in budgets
        }
        fixed_orders = {
            tuple(
                groups[(seed, budget, METHOD_FIXED)]["row_index"].astype(int)
            )
            for budget in budgets
        }
        if len(bce_orders) != 1:
            raise RuntimeError(
                f"BCE full order differs across budgets for seed {seed}"
            )
        if len(fixed_orders) != 1:
            raise RuntimeError(
                f"Fixed-reference full order differs across budgets for "
                f"seed {seed}"
            )


def _preflight_count(
    data: Mapping[str, object],
    field: str,
    profile: ExperimentProfileName,
) -> int:
    if field not in data:
        raise RuntimeError(
            f"{profile} preflight data is missing required field {field}."
        )
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            f"{profile} preflight field {field} must be a non-negative integer."
        )
    return value


def _validate_preflight_summary_agreement(
    summary: _PreflightDataSummary,
    context: PresentationInputContext,
) -> None:
    source = context.data_summary.get("source_counts")
    deduplicated = context.data_summary.get("deduplicated_counts")
    if not isinstance(source, Mapping) or not isinstance(deduplicated, Mapping):
        raise RuntimeError("Validated run data summary is incomplete.")
    expected_source_kind = "generated" if summary.source_kind == "synthetic" else "raw"
    if (
        context.data_summary.get("source_kind") != summary.source_kind
        or source.get("kind") != expected_source_kind
        or (source.get("rows"), source.get("fraud"), source.get("legitimate"))
        != (summary.source_rows, summary.source_fraud, summary.source_legitimate)
        or (
            deduplicated.get("rows"),
            deduplicated.get("fraud"),
            deduplicated.get("legitimate"),
        )
        != (
            summary.deduplicated_rows,
            summary.deduplicated_fraud,
            summary.deduplicated_legitimate,
        )
        or context.data_summary.get("removed_duplicate_count")
        != summary.removed_duplicate_count
        or context.data_summary.get("data_identity") != summary.data_identity
    ):
        raise RuntimeError(
            f"{context.profile} preflight counts or deduplication semantics, "
            "or identity, disagree with RUN_MANIFEST.json."
        )
    if summary.source_kind == "synthetic":
        synthetic = context.data_summary.get("synthetic")
        if not isinstance(synthetic, Mapping) or (
            synthetic.get("generator_schema"),
            synthetic.get("generation_seed"),
            synthetic.get("requested_row_count"),
        ) != (
            summary.synthetic_generator_schema,
            summary.synthetic_generation_seed,
            summary.synthetic_requested_row_count,
        ):
            raise RuntimeError(
                "smoke-synthetic preflight generator metadata disagrees with "
                "RUN_MANIFEST.json."
            )


def _normalize_preflight_data(
    data: Mapping[str, object],
    context: PresentationInputContext,
) -> _PreflightDataSummary:
    if (
        data.get("deduplication_keep") != "first"
        or data.get("deduplication_before_split") is not True
    ):
        raise RuntimeError(
            f"{context.profile} preflight deduplication semantics are invalid."
        )
    real_profile = context.profile in {"canonical", "mini-real"}
    if real_profile:
        if context.data_source_kind != "real":
            raise RuntimeError(
                f"{context.profile} presentation context has an invalid source."
            )
        if any(field in data for field in _SYNTHETIC_PREFLIGHT_FIELDS):
            raise RuntimeError(
                f"{context.profile} real-data preflight contains synthetic fields."
            )
        missing = sorted(_REAL_PREFLIGHT_FIELDS - set(data))
        if missing:
            raise RuntimeError(
                f"{context.profile} real-data preflight is missing required field "
                f"{missing[0]}."
            )
        deduplicated = (
            _preflight_count(data, "rows", context.profile),
            _preflight_count(data, "class_1", context.profile),
            _preflight_count(data, "class_0", context.profile),
        )
        hashes = (
            data.get("expected_raw_sha256"),
            data.get("actual_raw_sha256"),
            data.get("expected_deduplicated_dataframe_sha256"),
            data.get("actual_deduplicated_dataframe_sha256"),
        )
        if deduplicated != _REAL_DEDUPLICATED_COUNTS:
            raise RuntimeError(
                f"{context.profile} real-data preflight counts differ from the "
                "canonical dataset contract."
            )
        if hashes != (
            EXPECTED_RAW_SHA256,
            EXPECTED_RAW_SHA256,
            EXPECTED_DEDUPLICATED_SHA256,
            EXPECTED_DEDUPLICATED_SHA256,
        ):
            raise RuntimeError(
                f"{context.profile} real-data preflight identity is invalid."
            )
        summary = _PreflightDataSummary(
            "real",
            *_REAL_SOURCE_COUNTS,
            *_REAL_DEDUPLICATED_COUNTS,
            _REAL_REMOVED_DUPLICATE_COUNT,
            EXPECTED_DEDUPLICATED_SHA256,
        )
    elif context.profile == "smoke-synthetic":
        if context.data_source_kind != "synthetic":
            raise RuntimeError("smoke-synthetic context has an invalid source.")
        if any(field in data for field in _REAL_PREFLIGHT_FIELDS):
            raise RuntimeError(
                "smoke-synthetic preflight contains real-data fields."
            )
        missing = sorted(_SYNTHETIC_PREFLIGHT_FIELDS - set(data))
        if missing:
            raise RuntimeError(
                "smoke-synthetic preflight is missing required field "
                f"{missing[0]}."
            )
        generated = tuple(
            _preflight_count(data, field, context.profile)
            for field in (
                "synthetic_generated_row_count",
                "synthetic_generated_fraud_count",
                "synthetic_generated_legitimate_count",
            )
        )
        requested = _preflight_count(
            data, "synthetic_requested_row_count", context.profile
        )
        generation_seed = _preflight_count(
            data, "synthetic_generation_seed", context.profile
        )
        generator_schema = data.get("synthetic_generator_schema")
        identity = data.get("deterministic_data_identity")
        if generated[1] + generated[2] != generated[0]:
            raise RuntimeError(
                "smoke-synthetic preflight class-count arithmetic is invalid."
            )
        if generated[0] != requested:
            raise RuntimeError(
                "smoke-synthetic generated and requested row counts differ."
            )
        if (
            data.get("data_source_kind") != "synthetic"
            or not isinstance(generator_schema, str)
            or not generator_schema
            or not isinstance(identity, str)
            or data.get("evidence_classification")
            != context.evidence_classification
            or generation_seed
            != context.effective_config.synthetic_generation_seed
            or requested != context.effective_config.synthetic_row_target
        ):
            raise RuntimeError(
                "smoke-synthetic preflight semantic metadata is invalid."
            )
        summary = _PreflightDataSummary(
            "synthetic",
            *generated,
            *generated,
            0,
            identity,
            generator_schema,
            generation_seed,
            requested,
        )
    else:
        raise RuntimeError(
            f"Unsupported presentation profile: {context.profile!r}."
        )
    _validate_preflight_summary_agreement(summary, context)
    return summary


def _validate_run_manifests(
    store: ExperimentStore,
    context: PresentationInputContext,
) -> dict[str, Any]:
    preflight = store.json("preflight/preflight_validation.json")
    if (
        preflight.get("schema") != EXPECTED_PREFLIGHT_SCHEMA
        or preflight.get("status") != "PASS"
    ):
        raise RuntimeError("Preflight schema/status is not approved.")
    data = preflight.get("data")
    if not isinstance(data, Mapping):
        raise RuntimeError("Preflight data metadata is invalid.")
    _normalize_preflight_data(data, context)
    locked = preflight.get("locked_definitions")
    if not isinstance(locked, Mapping):
        raise RuntimeError("Preflight locked definitions are invalid.")
    outer_seeds = locked.get("outer_seeds")
    target_budgets = locked.get("target_budgets")
    if not isinstance(outer_seeds, list) or tuple(outer_seeds) != context.seeds:
        raise RuntimeError("Requested seeds differ from the experiment seeds.")
    if (
        not isinstance(target_budgets, list)
        or tuple(target_budgets) != context.target_budgets
    ):
        raise RuntimeError("Requested budgets differ from the experiment budgets.")
    if locked.get("ranker_scope") != "candidate_rerank":
        raise RuntimeError("Experiment ranker scope is not candidate_rerank.")
    if locked.get("candidate_pool_size") != context.candidate_pool_size:
        raise RuntimeError(
            "Experiment candidate-pool size differs from the run manifest."
        )

    final_manifest = store.json(
        "final_outer_run/final_outer_manifest.json"
    )
    if (
        final_manifest.get("schema") != EXPECTED_FINAL_SCHEMA
        or final_manifest.get("status") != "PASS"
    ):
        raise RuntimeError("Final-outer schema/status is not approved.")
    expected_final = {
        "outer_seed_count": len(context.seeds),
        "target_budget_count": len(context.target_budgets),
        "ranking_dump_count": len(context.seeds),
        "ranker_scope": "candidate_rerank",
    }
    for key, expected in expected_final.items():
        if final_manifest.get(key) != expected:
            raise RuntimeError(
                f"Final-outer field {key} differs: "
                f"{final_manifest.get(key)!r} != {expected!r}"
            )
    return preflight


def _validate_candidate_pools(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    expected_pool_size: int,
) -> None:
    for seed in seeds:
        reference_ids: set[int] | None = None
        reference_hash: str | None = None
        for budget in budgets:
            for method in METHOD_ORDER:
                ranking = groups[(seed, budget, method)]
                candidate = ranking.loc[
                    ranking["candidate_flag"].astype(bool)
                ]
                declared_sizes = (
                    ranking["candidate_pool_size"].dropna().astype(int).unique()
                )
                if (
                    len(declared_sizes) != 1
                    or int(declared_sizes[0]) != expected_pool_size
                ):
                    raise RuntimeError(
                        "Stored candidate-pool size mismatch: "
                        f"{(seed, budget, method)}"
                    )
                if len(candidate) != expected_pool_size:
                    raise RuntimeError(
                        f"Candidate-pool size mismatch: "
                        f"{(seed, budget, method)}"
                    )
                ids = set(candidate["row_index"].astype(int))
                hashes = (
                    candidate["candidate_pool_sha256"].dropna().unique()
                )
                if len(hashes) != 1:
                    raise RuntimeError(
                        f"Candidate-pool hash is not unique: "
                        f"{(seed, budget, method)}"
                    )
                observed_hash = str(hashes[0])
                if reference_ids is None:
                    reference_ids = ids
                    reference_hash = observed_hash
                elif ids != reference_ids or observed_hash != reference_hash:
                    raise RuntimeError(
                        f"Candidate-pool identity differs across paths: "
                        f"{(seed, budget, method)}"
                    )


def _validated_catalog_artifact_ids(
    catalog: Mapping[str, object],
) -> tuple[str, ...]:
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Presentation catalog artifacts are invalid.")
    identifiers: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise RuntimeError("Presentation catalog artifact is invalid.")
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise RuntimeError("Presentation catalog artifact ID is invalid.")
        identifiers.append(artifact_id)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Presentation catalog contains duplicate artifact IDs.")
    return tuple(identifiers)


def _validate_presentation_role(
    context: PresentationInputContext,
    requested: str,
) -> PresentationRole:
    derived = _presentation_role(
        context.profile,
        context.evidence_classification,
    )
    if context.presentation_role != derived:
        raise RuntimeError("Validated presentation role is inconsistent.")
    if requested not in {"canonical", "engineering"}:
        raise RuntimeError(f"Unsupported presentation role: {requested!r}.")
    if requested != context.presentation_role:
        if requested == "canonical":
            raise RuntimeError(
                "Canonical presentation catalog requested for noncanonical profile."
            )
        raise RuntimeError(
            "Engineering presentation catalog requested for canonical profile."
        )
    return context.presentation_role


def build(
    repository_root: Path,
    experiment_root: Path,
    output_dir: Path,
    *,
    input_context: PresentationInputContext,
    presentation_role: PresentationRole,
    force: bool = False,
    event_sink: _EventSink | None = None,
) -> dict[str, Any]:
    """Build R6 presentation data below one new ignored artifact root."""

    started = time.perf_counter()
    _emit_status(
        event_sink,
        "INFO",
        "build-data-start",
        experiment_root=experiment_root,
        output_dir=output_dir,
        utc=_status_utc_now(),
    )
    root = repository_root.resolve()
    experiment = experiment_root.resolve()
    if not isinstance(input_context, PresentationInputContext):
        raise TypeError("input_context must be a PresentationInputContext.")
    context = input_context
    if context.experiment_root != experiment:
        raise ValueError(
            "Presentation input context belongs to another experiment root."
        )
    role = _validate_presentation_role(context, presentation_role)
    seeds = context.seeds
    budgets = context.target_budgets
    requested_output = require_generated_path(root, output_dir)
    if (
        requested_output == experiment
        or requested_output in experiment.parents
        or experiment in requested_output.parents
    ):
        raise ValueError(
            "Presentation output must be disjoint from the completed "
            "experiment root."
        )
    store = ExperimentStore(context)
    catalog = build_profile_selection_registry(
        presentation_role=role,
        profile=context.profile,
        evidence_classification=context.evidence_classification,
        data_source_kind=context.data_source_kind,
    )
    catalog_artifact_ids = _validated_catalog_artifact_ids(catalog)
    expected_catalog_ids = (
        CANONICAL_ARTIFACT_IDS
        if role == "canonical"
        else ENGINEERING_ARTIFACT_IDS
    )
    if catalog_artifact_ids != expected_catalog_ids:
        raise RuntimeError(
            f"{role.capitalize()} presentation catalog inventory drift."
        )
    engineering_manifest_metadata: dict[str, str] = {}
    if role == "engineering":
        evidence_statement = catalog.get("evidence_statement")
        comparability_boundary = catalog.get("comparability_boundary")
        if (
            not isinstance(evidence_statement, str)
            or not evidence_statement
            or not isinstance(comparability_boundary, str)
            or not comparability_boundary
        ):
            raise RuntimeError("Engineering presentation evidence metadata is missing.")
        engineering_manifest_metadata = {
            "evidence_statement": evidence_statement,
            "comparability_boundary": comparability_boundary,
        }

    _validate_run_manifests(store, context)
    seed_metrics = _validate_seed_metrics(
        store.csv("figure_data/all_budget_matched_results.csv"),
        seeds,
        budgets,
    )
    diagnostics = store.csv("diagnostics/global_metrics_seedwise.csv")
    groups = _load_ranking_groups(
        store,
        seeds,
        budgets,
        context.candidate_pool_size,
    )
    _validate_candidate_pools(
        groups, seeds, budgets, context.candidate_pool_size
    )
    _emit_status(
        event_sink,
        "PASS",
        "build-data-input-validation",
        frames_expected=(
            EXPECTED_PRESENTATION_FRAMES
            if role == "canonical"
            else EXPECTED_ENGINEERING_FRAMES
        ),
    )

    if role == "canonical":
        outputs = derive_all(
            groups, seed_metrics, diagnostics, seeds, budgets
        )
        expected_output_paths = CANONICAL_OUTPUT_PATHS
        derivation_descriptions = [
            "paired trade-off",
            "discrete budget-policy profile",
            "within-model cumulative depth",
            "full-order and candidate-pool ROC/PR",
            "hard-impact seed profile",
            "row-index replacement events",
            "seedwise k=50 diagnostic",
            "seed-budget heatmap grid",
            "exact tie permutation intervals",
            "candidate-pool coverage and ceiling utilization",
        ]
    else:
        outputs = derive_engineering(
            seed_metrics,
            seeds,
            budgets,
            profile=context.profile,
            evidence_classification=context.evidence_classification,
            data_source_kind=context.data_source_kind,
        )
        expected_output_paths = ENGINEERING_OUTPUT_PATHS
        derivation_descriptions = [
            "engineering seed-budget comparison-path deltas",
            "engineering central Top-k mean and sample SD",
        ]
    if tuple(outputs) != expected_output_paths:
        raise RuntimeError(
            f"{role.capitalize()} presentation-data inventory drift."
        )
    artifact_root = prepare_output_directory(root, output_dir, force=force)
    data_root = artifact_root / "data"
    data_root.mkdir(parents=True, exist_ok=False)
    generated_files: list[Path] = []
    completed = 0
    for relative, frame in sorted(outputs.items()):
        target = data_root / relative
        write_csv(target, frame)
        generated_files.append(target)
        completed += 1
        _emit_status(
            event_sink,
            "INFO",
            "build-data-frame",
            completed=completed,
            total=len(expected_output_paths),
            name=relative,
        )

    manifest = {
        "schema": "fraud_detection.chapter5_presentation_data.r6.v1",
        "status": "PASS",
        "profile": context.profile,
        "presentation_role": role,
        "evidence_classification": context.evidence_classification,
        "data_source_kind": context.data_source_kind,
        "experiment_root_kind": "checksum-registered completed run",
        "experiment_checksum_manifest_sha256": sha256_file(
            store.path(CHECKSUM_MANIFEST_PATH)
        ),
        "seeds": list(seeds),
        "budgets": list(budgets),
        "primary_budgets": list(context.primary_budgets),
        "candidate_pool_size": context.candidate_pool_size,
        "data_summary": _json_compatible(context.data_summary),
        "selected_catalog_artifact_ids": list(catalog_artifact_ids),
        "sources": [
            {"path": path, "sha256": store.registered[path]}
            for path in sorted(store.read_paths)
            if path != CHECKSUM_MANIFEST_PATH
        ],
        "outputs": file_inventory(data_root, generated_files),
        "derivations": derivation_descriptions,
        "model_fit_performed": False,
        "model_scoring_performed": False,
        "parameter_selection_performed": False,
        "ranking_modified": False,
        "rendering_performed": False,
        "latex_generation_performed": False,
        "cross_budget_rank_segments_generated": False,
        "historical_near_tie_generated": False,
        "historical_presentation_values_used": False,
        **engineering_manifest_metadata,
    }
    write_json(data_root / "PRESENTATION_DATA_MANIFEST.json", manifest)
    write_json(
        artifact_root / "PRESENTATION_SELECTION.json",
        catalog,
    )
    _emit_status(
        event_sink,
        "PASS",
        "build-data-complete",
        frames=len(expected_output_paths),
        manifest=data_root / "PRESENTATION_DATA_MANIFEST.json",
        elapsed_seconds=time.perf_counter() - started,
    )
    return manifest
