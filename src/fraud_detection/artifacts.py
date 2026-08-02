"""Safe generated paths and read-only artifact inventory."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import IO, Any, Final, Literal

import numpy as np
import pandas as pd

from fraud_detection.errors import ProductError
from fraud_detection.setup.environment import (
    find_repository_root as find_repository_root,
)

GENERATED_ROOT_NAMES: Final[frozenset[str]] = frozenset(
    {"generated", "outputs", "thesis_build"}
)


def require_generated_path(repository_root: Path, output: Path) -> Path:
    """Resolve *output* and require it below an ignored generated root."""

    root = repository_root.resolve()
    target = output if output.is_absolute() else root / output
    target = target.resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Output must be inside this repository under generated/, "
            "outputs/, or thesis_build/."
        ) from exc
    if not relative.parts or relative.parts[0] not in GENERATED_ROOT_NAMES:
        raise ValueError(
            "Output must be below generated/, outputs/, or thesis_build/."
        )
    if len(relative.parts) == 1:
        raise ValueError("Refusing to use an entire generated root as output.")
    return target

def _physical_text(path: Path | str) -> str:
    """Return an absolute extended path for Windows physical operations."""

    value = os.path.abspath(os.fspath(path))
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return value
    if value.startswith("\\\\"):
        return "\\\\?\\UNC\\" + value.removeprefix("\\\\")
    return "\\\\?\\" + value


def _open_file(
    path: Path | str,
    mode: str,
    *,
    encoding: str | None = None,
    newline: str | None = None,
) -> IO[Any]:
    return open(  # noqa: PTH123 - builtin open accepts extended Windows paths.
        _physical_text(path),
        mode,
        encoding=encoding,
        newline=newline,
    )


def _path_exists(path: Path | str) -> bool:
    return os.path.exists(_physical_text(path))


def _path_is_file(path: Path | str) -> bool:
    return os.path.isfile(_physical_text(path))


def _path_is_dir(path: Path | str) -> bool:
    return os.path.isdir(_physical_text(path))


def _path_mkdir(
    path: Path | str,
    *,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    physical = _physical_text(path)
    if parents:
        os.makedirs(physical, exist_ok=exist_ok)
        return
    try:
        os.mkdir(physical)
    except FileExistsError:
        if not exist_ok:
            raise


def _path_read_text(path: Path | str, *, encoding: str) -> str:
    with _open_file(path, "r", encoding=encoding) as handle:
        return str(handle.read())


def _path_write_text(
    path: Path | str,
    value: str,
    *,
    encoding: str,
    newline: str | None = None,
) -> None:
    with _open_file(path, "w", encoding=encoding, newline=newline) as handle:
        handle.write(value)


def _ensure_run_directories(root: Path) -> None:
    required = (
        "preflight",
        "inner_validation",
        "selection_freeze",
        "final_outer_run",
        "diagnostics",
        "tables",
        "figure_data",
        "comparison",
        "logs",
    )
    if not _path_is_dir(root):
        raise FileNotFoundError("The generated run root has not been initialized.")
    for relative in required:
        path = root / relative
        if not _path_is_dir(path):
            _path_mkdir(path)


def _require_new_file(path: Path) -> None:
    if _path_exists(path):
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    _path_mkdir(path.parent, parents=True, exist_ok=True)


def _write_json(path: Path, payload: object) -> None:
    _require_new_file(path)
    _path_write_text(
        path,
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
def _write_csv(path: Path, frame: pd.DataFrame, *, compressed: bool = False) -> None:
    _require_new_file(path)
    compression: object = (
        {"method": "gzip", "mtime": 0} if compressed else None
    )
    frame.to_csv(path, index=False, compression=compression)
def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    _require_new_file(path)
    try:
        frame.to_parquet(
            path,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for the mandated Parquet artifacts."
        ) from exc
def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON payload contains a non-finite float.")
        return value
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_file(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_index_sha256(index: object) -> str:
    values = pd.util.hash_pandas_object(
        pd.Series(np.asarray(index)),
        index=False,
        categorize=True,
    ).to_numpy(dtype="<u8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.ordered_index_sha256.v1\0")
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _index_score_mapping_sha256(
    index: object,
    scores: object,
    *,
    score_type: str,
) -> str:
    index_values = np.asarray(index, dtype="<i8")
    score_values = np.asarray(scores, dtype="<f8")
    if (
        index_values.ndim != 1
        or score_values.ndim != 1
        or index_values.shape[0] != score_values.shape[0]
        or not np.isfinite(score_values).all()
    ):
        raise ValueError("Invalid index/score mapping.")
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.index_score_mapping_sha256.v1\0")
    digest.update(score_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(index_values).tobytes())
    digest.update(np.ascontiguousarray(score_values).tobytes())
    return digest.hexdigest()


def _integer_vector_sha256(values: object, *, vector_type: str) -> str:
    arr = np.asarray(values, dtype="<i8")
    if arr.ndim != 1:
        raise ValueError("Integer vector must be one-dimensional.")
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.integer_vector_sha256.v1\0")
    digest.update(vector_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _write_checksum_manifest(
    root: Path,
    paths: Iterable[Path],
    output_path: Path,
) -> None:
    lines = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)} *{relative}")
    value = "\n".join(lines) + "\n"
    _require_new_file(output_path)
    _path_write_text(output_path, value, encoding="ascii")


def _verify_checksum_manifest(root: Path, manifest_path: Path) -> int:
    failures: list[str] = []
    count = 0
    for line in _path_read_text(
        manifest_path,
        encoding="ascii",
    ).splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = root / relative
        count += 1
        if not _path_is_file(path) or _sha256_file(path) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(
            f"Checksum verification failed for: {failures[:5]}"
        )
    return count


def _finite_score_vector(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one element.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_vector_sha256(values: object, *, score_type: str) -> str:
    arr = _finite_score_vector(values, score_type).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.score_vector_sha256.v1\0")
    digest.update(score_type.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


_EXPECTED_REFERENCE_FILENAMES = frozenset(
    {
        "central_topk_results.csv",
        "selected_configuration_summary.csv",
        "data_identity.json",
    }
)
_RUN_MANIFEST_NAME = "RUN_MANIFEST.json"
_PRESENTATION_SENTINELS = (
    "PRESENTATION_SELECTION.json",
    "data/PRESENTATION_DATA_MANIFEST.json",
    "figures/FIGURE_RENDER_MANIFEST.json",
    "tables/TABLE_RENDER_MANIFEST.json",
)
_PHASE_CONTRACTS = (
    (
        "preflight",
        "preflight/preflight_validation.json",
        "ranker_gain_validation.preflight.v2",
    ),
    (
        "inner_selection",
        "inner_validation/inner_validation_manifest.json",
        None,
    ),
    (
        "selection_freeze",
        "selection_freeze/selection_manifest.json",
        "ranker_gain_validation.selection_manifest.v1",
    ),
    (
        "final_outer",
        "final_outer_run/final_outer_manifest.json",
        "ranker_gain_validation.final_outer.v1",
    ),
    ("qa", "comparison/final_qa.json", None),
)
_SEMANTIC_PHASE_ORDER = (
    "preflight",
    "inner_selection",
    "selection_freeze",
    "final_outer",
    "aggregation",
    "qa",
)
_PUBLIC_COMPLETED_PHASES = ("inner", "final", "qa")
_PUBLIC_COMMANDS = (
    "setup",
    "check",
    "run",
    "build",
    "inspect",
)


@dataclass(frozen=True, slots=True)
class InspectionResult:
    """Immutable semantic summary of one supported read-only path."""

    inspected_path: Path
    path_type: Literal[
        "repository",
        "experiment",
        "experiment-partial",
        "presentation",
    ]
    status: str
    profile: str | None = None
    presentation_role: str | None = None
    evidence_classification: str | None = None
    source_kind: str | None = None
    completed_phases: tuple[str, ...] = ()
    missing_phases: tuple[str, ...] = ()
    artifact_count: int | None = None
    checksum_status: str | None = None
    prepared_data_count: int | None = None
    figure_count: int | None = None
    figure_file_count: int | None = None
    table_count: int | None = None
    table_file_count: int | None = None
    presentation_compatible: bool | None = None
    key_paths: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    suggested_command: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key_paths",
            MappingProxyType(dict(self.key_paths)),
        )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(dict(self.details)),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the stable CLI-facing semantic fields."""

        return {
            "inspected_path": str(self.inspected_path),
            "path_type": self.path_type,
            "status": self.status,
            "profile": self.profile,
            "presentation_role": self.presentation_role,
            "evidence_classification": self.evidence_classification,
            "source_kind": self.source_kind,
            "completed_phases": list(self.completed_phases),
            "missing_phases": list(self.missing_phases),
            "artifact_count": self.artifact_count,
            "checksum_status": self.checksum_status,
            "prepared_data_count": self.prepared_data_count,
            "figure_count": self.figure_count,
            "figure_file_count": self.figure_file_count,
            "table_count": self.table_count,
            "table_file_count": self.table_file_count,
            "presentation_compatible": self.presentation_compatible,
            "key_paths": dict(self.key_paths),
            "warnings": list(self.warnings),
            "suggested_command": self.suggested_command,
            "details": dict(self.details),
        }


def _read_inspection_json(path: Path, label: str) -> dict[str, object]:
    if not _path_is_file(path):
        raise ProductError(
            "FD-INSPECT-ARTIFACT",
            f"{label} is missing or is not a file.",
            ("Supply an intact output root and retry.",),
            {"path": str(path)},
        )
    try:
        value = json.loads(_path_read_text(path, encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductError(
            "FD-INSPECT-MANIFEST",
            f"{label} is not valid UTF-8 JSON.",
            ("Supply an intact output root and retry.",),
            {"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise ProductError(
            "FD-INSPECT-MANIFEST",
            f"{label} must contain one JSON object.",
            ("Supply an intact output root and retry.",),
            {"path": str(path)},
        )
    return value


def _inspection_manifest_error(
    label: str,
    summary: str,
    *,
    path: Path,
) -> ProductError:
    return ProductError(
        "FD-INSPECT-MANIFEST",
        f"{label} {summary}",
        ("Supply an intact output root produced by this version.",),
        {"path": str(path)},
    )


def _repository_signature(root: Path) -> bool:
    return (
        _path_is_file(root / "pyproject.toml")
        and _path_is_file(root / "src" / "fraud_detection" / "__init__.py")
    )


def _presentation_signature(root: Path) -> bool:
    return any(_path_exists(root / relative) for relative in _PRESENTATION_SENTINELS)


def _partial_experiment_signature(root: Path) -> bool:
    return any(
        _path_exists(root / relative)
        for _phase, relative, _schema in _PHASE_CONTRACTS
    )


def _inspect_repository(root: Path) -> InspectionResult:
    pyproject_path = root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(
            _path_read_text(pyproject_path, encoding="utf-8-sig")
        )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ProductError(
            "FD-INSPECT-REPOSITORY",
            "The repository pyproject.toml is unreadable or malformed.",
            ("Supply the root of an intact repository checkout.",),
            {"path": str(pyproject_path)},
        ) from exc
    project = pyproject.get("project")
    scripts = project.get("scripts") if isinstance(project, dict) else None
    if not isinstance(project, dict) or not isinstance(scripts, dict):
        raise ProductError(
            "FD-INSPECT-REPOSITORY",
            "The repository does not declare the expected project metadata.",
            ("Supply the root of an intact repository checkout.",),
            {"path": str(root)},
        )
    entry_point = scripts.get("fraud-detection")
    if not isinstance(entry_point, str) or not entry_point:
        raise ProductError(
            "FD-INSPECT-REPOSITORY",
            "The fraud-detection console entry point is missing.",
            ("Supply the root of an intact repository checkout.",),
            {"path": str(pyproject_path)},
        )
    config = import_module("fraud_detection.experiment.config")
    reference_root = root / "reference_results"
    reference_paths = {
        name: reference_root / name for name in sorted(_EXPECTED_REFERENCE_FILENAMES)
    }
    present_references = tuple(
        name for name, path in reference_paths.items() if _path_is_file(path)
    )
    return InspectionResult(
        inspected_path=root,
        path_type="repository",
        status="VALID",
        artifact_count=len(present_references),
        checksum_status="NOT_APPLICABLE",
        key_paths={
            "pyproject": str(pyproject_path),
            "package": str(root / "src" / "fraud_detection"),
            "reference_results": str(reference_root),
            "local_data": str(root / "data" / "creditcard.csv"),
        },
        suggested_command=(
            "fraud-detection run --profile smoke-synthetic --dry-run"
        ),
        details={
            "distribution_name": project.get("name"),
            "package_name": "fraud_detection",
            "console_entry_point": "fraud-detection",
            "entry_point_target": entry_point,
            "supported_profiles": list(config.EXPERIMENT_PROFILE_NAMES),
            "public_commands": list(_PUBLIC_COMMANDS),
            "reference_results_present": (
                len(present_references) == len(reference_paths)
            ),
            "reference_result_files_present": list(present_references),
            "local_data_path_exists": _path_is_file(
                root / "data" / "creditcard.csv"
            ),
            "authorized_output_roots": sorted(GENERATED_ROOT_NAMES),
        },
    )


def _validated_partial_phases(
    root: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    completed: set[str] = set()
    key_paths: dict[str, str] = {}
    for phase, relative, expected_schema in _PHASE_CONTRACTS:
        path = root / relative
        if not _path_exists(path):
            continue
        manifest = _read_inspection_json(path, f"{phase} manifest")
        if expected_schema is not None and manifest.get("schema") != expected_schema:
            raise ProductError(
                "FD-INSPECT-SCHEMA",
                f"The {phase} manifest uses an unsupported schema.",
                ("Supply an output root produced by this version.",),
                {"path": str(path)},
            )
        if phase == "selection_freeze":
            if (
                manifest.get("outer_test_selection_locked") is not True
                or manifest.get("outer_test_labels_used_for_selection") is not False
            ):
                raise _inspection_manifest_error(
                    "The selection-freeze manifest",
                    "does not record a completed locked selection.",
                    path=path,
                )
        elif manifest.get("status") != "PASS":
            raise _inspection_manifest_error(
                f"The {phase} manifest",
                "does not have status PASS.",
                path=path,
            )
        completed.add(phase)
        if phase == "final_outer":
            completed.add("aggregation")
        key_paths[f"{phase}_manifest"] = str(path)
    ordered = tuple(phase for phase in _SEMANTIC_PHASE_ORDER if phase in completed)
    return ordered, key_paths


def _inspect_partial_experiment(root: Path) -> InspectionResult:
    completed, key_paths = _validated_partial_phases(root)
    if not completed:
        raise ProductError(
            "FD-INSPECT-UNSUPPORTED",
            "The directory does not contain a valid current experiment phase.",
            ("Supply a repository, experiment, partial-run, or presentation root.",),
            {"path": str(root)},
        )
    missing = tuple(
        phase for phase in _SEMANTIC_PHASE_ORDER if phase not in completed
    )
    latest = completed[-1] if completed else None
    return InspectionResult(
        inspected_path=root,
        path_type="experiment-partial",
        status="INCOMPLETE",
        completed_phases=completed,
        missing_phases=missing,
        artifact_count=len(key_paths),
        checksum_status="NOT_APPLICABLE",
        presentation_compatible=False,
        key_paths=key_paths,
        warnings=(
            "No COMPLETE RUN_MANIFEST.json is present.",
            "This partial run is not valid presentation input.",
            "Fit-level resume is unsupported.",
        ),
        suggested_command="Start a new run using a new output directory.",
        details={
            "latest_complete_semantic_artifact": latest,
            "fit_level_resume": "unsupported",
        },
    )


def _completed_run_error(root: Path, exc: Exception) -> ProductError:
    message = str(exc)
    lowered = message.lower()
    if "checksum mismatch" in lowered:
        code = "FD-INSPECT-CHECKSUM"
        summary = "The completed experiment has a checksum mismatch."
    elif "unsafe artifact path" in lowered or "escapes root" in lowered:
        code = "FD-INSPECT-ARTIFACT"
        summary = "The completed experiment registers an unsafe artifact path."
    elif "missing" in lowered or "does not match produced files" in lowered:
        code = "FD-INSPECT-ARTIFACT"
        summary = "The completed experiment has a missing or unregistered artifact."
    elif "schema" in lowered:
        code = "FD-INSPECT-SCHEMA"
        summary = "The completed experiment uses an unsupported manifest schema."
    else:
        code = "FD-INSPECT-MANIFEST"
        summary = "The completed experiment manifest is invalid."
    return ProductError(
        code,
        summary,
        ("Supply an intact completed experiment-run root.",),
        {"path": str(root), "reason": message},
    )


def _inspect_completed_experiment(root: Path) -> InspectionResult:
    data_stage = import_module(
        "fraud_detection.presentation.preparation.data"
    )
    try:
        context = data_stage.load_presentation_input_context(root)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _completed_run_error(root, exc) from exc

    completed, phase_paths = _validated_partial_phases(root)
    if completed != _SEMANTIC_PHASE_ORDER:
        missing = [
            phase for phase in _SEMANTIC_PHASE_ORDER if phase not in completed
        ]
        raise ProductError(
            "FD-INSPECT-MANIFEST",
            "The COMPLETE run lacks completed semantic phase contracts.",
            ("Supply an intact completed experiment-run root.",),
            {"path": str(root), "missing_phases": missing},
        )
    for manifest_path in phase_paths.values():
        relative = Path(manifest_path).relative_to(root).as_posix()
        if relative not in context.artifact_checksums:
            raise ProductError(
                "FD-INSPECT-CHECKSUM",
                "A completed semantic phase manifest lacks checksum coverage.",
                ("Supply an intact completed experiment-run root.",),
                {"path": manifest_path},
            )

    summary = context.data_summary
    source_counts = summary.get("source_counts")
    deduplicated_counts = summary.get("deduplicated_counts")
    if not isinstance(source_counts, Mapping) or not isinstance(
        deduplicated_counts, Mapping
    ):
        raise ProductError(
            "FD-INSPECT-MANIFEST",
            "The completed experiment data summary is invalid.",
            ("Supply an intact completed experiment-run root.",),
            {"path": str(root / _RUN_MANIFEST_NAME)},
        )
    seed_count = len(context.seeds)
    budget_count = len(context.target_budgets)
    try:
        ranking_groups = data_stage._load_ranking_groups(
            data_stage.ExperimentStore(context),
            context.seeds,
            context.target_budgets,
            context.candidate_pool_size,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _completed_run_error(root, exc) from exc
    method_count = len({identity[2] for identity in ranking_groups})
    ranking_group_count = len(ranking_groups)
    key_paths = {
        "run_manifest": str(root / _RUN_MANIFEST_NAME),
        "checksum_manifest": str(root / "comparison" / "checksums.sha256"),
        "preflight_manifest": str(
            root / "preflight" / "preflight_validation.json"
        ),
        "final_outer_manifest": str(
            root / "final_outer_run" / "final_outer_manifest.json"
        ),
        "qa_manifest": str(root / "comparison" / "final_qa.json"),
    }
    return InspectionResult(
        inspected_path=root,
        path_type="experiment",
        status="COMPLETE",
        profile=context.profile,
        presentation_role=context.presentation_role,
        evidence_classification=context.evidence_classification,
        source_kind=context.data_source_kind,
        completed_phases=_SEMANTIC_PHASE_ORDER,
        artifact_count=len(context.registered_artifact_paths),
        checksum_status="VERIFIED",
        presentation_compatible=True,
        key_paths=key_paths,
        suggested_command=f"fraud-detection build {root}",
        details={
            "public_completed_phases": list(_PUBLIC_COMPLETED_PHASES),
            "seeds": list(context.seeds),
            "budgets": list(context.target_budgets),
            "candidate_pool_size": context.candidate_pool_size,
            "checksummed_artifact_count": len(context.artifact_checksums),
            "result_grid_dimensions": {
                "seeds": seed_count,
                "budgets": budget_count,
                "methods": method_count,
                "rows": ranking_group_count,
            },
            "ranking_group_count": ranking_group_count,
            "data_summary_counts": {
                "source": dict(source_counts),
                "deduplicated": dict(deduplicated_counts),
                "removed_duplicate_count": summary.get(
                    "removed_duplicate_count"
                ),
            },
        },
    )


def _safe_inventory_target(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise _inspection_manifest_error(
            label,
            "contains an invalid artifact path.",
            path=root,
        )
    pure = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\x00" in value
        or pure.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in pure.parts
        or pure.as_posix() != value
    ):
        raise ProductError(
            "FD-INSPECT-ARTIFACT",
            f"{label} contains an unsafe artifact path.",
            ("Supply an intact presentation root.",),
            {"path": value},
        )
    target = (root / pure).resolve()
    if root not in target.parents:
        raise ProductError(
            "FD-INSPECT-ARTIFACT",
            f"{label} contains an artifact path outside its output root.",
            ("Supply an intact presentation root.",),
            {"path": value},
        )
    return target


def _validate_presentation_inventory(
    root: Path,
    entries: object,
    label: str,
) -> int:
    if not isinstance(entries, list):
        raise _inspection_manifest_error(
            label,
            "has an invalid output inventory.",
            path=root,
        )
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise _inspection_manifest_error(
                label,
                "has an invalid output inventory entry.",
                path=root,
            )
        relative = entry["path"]
        target = _safe_inventory_target(root, relative, label)
        if not isinstance(relative, str) or relative in seen:
            raise _inspection_manifest_error(
                label,
                "has duplicate output paths.",
                path=root,
            )
        seen.add(relative)
        if not _path_is_file(target):
            raise ProductError(
                "FD-INSPECT-ARTIFACT",
                f"{label} registers a missing output file.",
                ("Supply an intact presentation root.",),
                {"path": str(target)},
            )
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise _inspection_manifest_error(
                label,
                "contains an invalid checksum.",
                path=root,
            )
        try:
            size = target.stat().st_size
            observed = _sha256_file(target)
        except OSError as exc:
            raise ProductError(
                "FD-INSPECT-ARTIFACT",
                f"{label} contains an unreadable output file.",
                ("Check file permissions and retry.",),
                {"path": str(target)},
            ) from exc
        if entry["size_bytes"] != size or digest != observed:
            raise ProductError(
                "FD-INSPECT-CHECKSUM",
                f"{label} output checksum or size does not match.",
                ("Supply an intact presentation root.",),
                {"path": str(target)},
            )
    return len(seen)


def _require_presentation_manifest(
    manifest: Mapping[str, object],
    *,
    label: str,
    path: Path,
    schemas: frozenset[str],
) -> None:
    if manifest.get("schema") not in schemas:
        raise ProductError(
            "FD-INSPECT-SCHEMA",
            f"{label} uses an unsupported schema.",
            ("Supply a presentation root produced by this version.",),
            {"path": str(path)},
        )
    if manifest.get("status") != "PASS":
        raise _inspection_manifest_error(
            label,
            "does not have status PASS.",
            path=path,
        )


def _presentation_warnings(
    profile: str,
    role: str,
    missing_stages: tuple[str, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if missing_stages:
        warnings.append(
            "Presentation output is inspectable but incomplete; missing: "
            + ", ".join(missing_stages)
            + "."
        )
    if role == "engineering":
        if profile == "smoke-synthetic":
            warnings.append("Deterministic synthetic engineering data.")
        else:
            warnings.append("Engineering mini profile using the real canonical dataset.")
        warnings.extend(
            (
                "Not thesis evidence.",
                "Not comparable with canonical empirical results.",
            )
        )
    return tuple(warnings)


def _inspect_presentation(root: Path) -> InspectionResult:
    selection_path = root / "PRESENTATION_SELECTION.json"
    data_path = root / "data" / "PRESENTATION_DATA_MANIFEST.json"
    selection = _read_inspection_json(
        selection_path, "PRESENTATION_SELECTION.json"
    )
    data_manifest = _read_inspection_json(
        data_path, "PRESENTATION_DATA_MANIFEST.json"
    )
    _require_presentation_manifest(
        data_manifest,
        label="The prepared-data manifest",
        path=data_path,
        schemas=frozenset(
            {"fraud_detection.chapter5_presentation_data.r6.v1"}
        ),
    )
    profile = data_manifest.get("profile")
    if not isinstance(profile, str):
        raise _inspection_manifest_error(
            "The prepared-data manifest",
            "has no supported source profile.",
            path=data_path,
        )
    config = import_module("fraud_detection.experiment.config")
    try:
        effective = config.resolve_experiment_profile(profile)
    except ValueError as exc:
        raise _inspection_manifest_error(
            "The prepared-data manifest",
            "has an unknown source profile.",
            path=data_path,
        ) from exc
    role = "canonical" if profile == "canonical" else "engineering"
    if (
        data_manifest.get("presentation_role") != role
        or data_manifest.get("evidence_classification")
        != effective.evidence_classification
        or data_manifest.get("data_source_kind") != effective.data_source_kind
        or data_manifest.get("seeds") != list(effective.seeds)
        or data_manifest.get("budgets") != list(effective.target_budgets)
        or data_manifest.get("primary_budgets")
        != list(effective.primary_budgets)
        or data_manifest.get("candidate_pool_size")
        != effective.candidate_pool_size
    ):
        raise _inspection_manifest_error(
            "The prepared-data manifest",
            "does not match its source profile.",
            path=data_path,
        )
    catalog = import_module("fraud_detection.presentation.catalog")
    expected_selection = catalog.build_profile_selection_registry(
        presentation_role=role,
        profile=profile,
        evidence_classification=effective.evidence_classification,
        data_source_kind=effective.data_source_kind,
    )
    if selection != expected_selection:
        raise _inspection_manifest_error(
            "The presentation selection manifest",
            "does not match the current profile catalog.",
            path=selection_path,
        )
    expected_ids = (
        catalog.CANONICAL_ARTIFACT_IDS
        if role == "canonical"
        else catalog.ENGINEERING_ARTIFACT_IDS
    )
    if data_manifest.get("selected_catalog_artifact_ids") != list(expected_ids):
        raise _inspection_manifest_error(
            "The prepared-data manifest",
            "does not match the selected artifact catalog.",
            path=data_path,
        )
    if role == "engineering":
        evidence_statement = data_manifest.get("evidence_statement")
        comparability = data_manifest.get("comparability_boundary")
        if (
            not isinstance(evidence_statement, str)
            or "not thesis evidence" not in evidence_statement
            or not isinstance(comparability, str)
            or comparability not in evidence_statement
        ):
            raise _inspection_manifest_error(
                "The prepared-data manifest",
                "is missing the engineering evidence boundary.",
                path=data_path,
            )
    prepared_count = _validate_presentation_inventory(
        data_path.parent,
        data_manifest.get("outputs"),
        "The prepared-data manifest",
    )
    derivations = import_module(
        "fraud_detection.presentation.preparation.derivations"
    )
    expected_prepared_paths = (
        derivations.CANONICAL_OUTPUT_PATHS
        if role == "canonical"
        else derivations.ENGINEERING_OUTPUT_PATHS
    )
    data_outputs = data_manifest["outputs"]
    if not isinstance(data_outputs, list) or [
        entry["path"] for entry in data_outputs
    ] != sorted(expected_prepared_paths):
        raise _inspection_manifest_error(
            "The prepared-data manifest",
            "does not contain the complete profile output scope.",
            path=data_path,
        )

    figure_path = root / "figures" / "FIGURE_RENDER_MANIFEST.json"
    table_path = root / "tables" / "TABLE_RENDER_MANIFEST.json"
    figure_count: int | None = None
    figure_file_count: int | None = None
    table_count: int | None = None
    table_file_count: int | None = None
    latex_status: str | None = None
    missing_stages: list[str] = []
    key_paths = {
        "selection_manifest": str(selection_path),
        "prepared_data_manifest": str(data_path),
        "figure_directory": str(root / "figures"),
        "table_directory": str(root / "tables"),
    }

    if _path_exists(figure_path):
        figure_manifest = _read_inspection_json(
            figure_path, "FIGURE_RENDER_MANIFEST.json"
        )
        _require_presentation_manifest(
            figure_manifest,
            label="The figure manifest",
            path=figure_path,
            schemas=frozenset(
                {
                    "fraud_detection.chapter5_figure_render.r7b.v1",
                    "fraud_detection.engineering_figure_render.v1",
                }
            ),
        )
        expected_figure_schema = (
            "fraud_detection.chapter5_figure_render.r7b.v1"
            if role == "canonical"
            else "fraud_detection.engineering_figure_render.v1"
        )
        if figure_manifest.get("schema") != expected_figure_schema:
            raise _inspection_manifest_error(
                "The figure manifest",
                "does not match the presentation role.",
                path=figure_path,
            )
        stems = figure_manifest.get("rendered_stems")
        if not isinstance(stems, list) or not all(
            isinstance(stem, str) and stem for stem in stems
        ):
            raise _inspection_manifest_error(
                "The figure manifest",
                "has an invalid logical figure inventory.",
                path=figure_path,
            )
        figure_count = len(stems)
        figure_file_count = _validate_presentation_inventory(
            figure_path.parent,
            figure_manifest.get("outputs"),
            "The figure manifest",
        )
        expected_figure_count = 9 if role == "canonical" else 1
        if (
            figure_count != expected_figure_count
            or figure_file_count != expected_figure_count * 3
        ):
            raise _inspection_manifest_error(
                "The figure manifest",
                "does not contain the complete logical and physical output scope.",
                path=figure_path,
            )
        if role == "engineering" and (
            figure_manifest.get("profile") != profile
            or figure_manifest.get("presentation_role") != role
            or figure_manifest.get("evidence_classification")
            != effective.evidence_classification
            or figure_manifest.get("logical_figure_count") != figure_count
            or figure_manifest.get("rendered_file_count") != figure_file_count
        ):
            raise _inspection_manifest_error(
                "The figure manifest",
                "does not match the engineering presentation contract.",
                path=figure_path,
            )
        key_paths["figure_manifest"] = str(figure_path)
    else:
        missing_stages.append("figures")

    if _path_exists(table_path):
        table_manifest = _read_inspection_json(
            table_path, "TABLE_RENDER_MANIFEST.json"
        )
        _require_presentation_manifest(
            table_manifest,
            label="The table manifest",
            path=table_path,
            schemas=frozenset(
                {
                    "fraud_detection.chapter5_table_render.r6.v1",
                    "fraud_detection.engineering_table_render.v1",
                }
            ),
        )
        expected_table_schema = (
            "fraud_detection.chapter5_table_render.r6.v1"
            if role == "canonical"
            else "fraud_detection.engineering_table_render.v1"
        )
        if table_manifest.get("schema") != expected_table_schema:
            raise _inspection_manifest_error(
                "The table manifest",
                "does not match the presentation role.",
                path=table_path,
            )
        tables = table_manifest.get("tables")
        if not isinstance(tables, list):
            raise _inspection_manifest_error(
                "The table manifest",
                "has an invalid logical table inventory.",
                path=table_path,
            )
        table_count = len(tables)
        table_file_count = _validate_presentation_inventory(
            table_path.parent,
            table_manifest.get("outputs"),
            "The table manifest",
        )
        expected_table_count = 9 if role == "canonical" else 1
        if (
            table_count != expected_table_count
            or table_file_count != expected_table_count * 2
        ):
            raise _inspection_manifest_error(
                "The table manifest",
                "does not contain the complete logical and physical output scope.",
                path=table_path,
            )
        preview = table_manifest.get("latex_preview")
        if not isinstance(preview, Mapping) or preview.get("status") not in {
            "PASS",
            "SKIPPED_NO_ENGINE",
        }:
            raise _inspection_manifest_error(
                "The table manifest",
                "has an invalid LaTeX preview status.",
                path=table_path,
            )
        latex_status = str(preview["status"])
        if role == "engineering" and (
            table_manifest.get("profile") != profile
            or table_manifest.get("presentation_role") != role
            or table_manifest.get("evidence_classification")
            != effective.evidence_classification
            or table_manifest.get("logical_table_count") != table_count
            or table_manifest.get("rendered_file_count") != table_file_count
        ):
            raise _inspection_manifest_error(
                "The table manifest",
                "does not match the engineering presentation contract.",
                path=table_path,
            )
        key_paths["table_manifest"] = str(table_path)
    else:
        missing_stages.append("tables")

    missing = tuple(missing_stages)
    return InspectionResult(
        inspected_path=root,
        path_type="presentation",
        status="COMPLETE" if not missing else "INSPECTABLE",
        profile=profile,
        presentation_role=role,
        evidence_classification=effective.evidence_classification,
        source_kind=effective.data_source_kind,
        missing_phases=missing,
        artifact_count=(
            prepared_count
            + (figure_file_count or 0)
            + (table_file_count or 0)
        ),
        checksum_status="VERIFIED",
        prepared_data_count=prepared_count,
        figure_count=figure_count,
        figure_file_count=figure_file_count,
        table_count=table_count,
        table_file_count=table_file_count,
        key_paths=key_paths,
        warnings=_presentation_warnings(profile, role, missing),
        suggested_command=(
            f"Review figures in {root / 'figures'} and tables in {root / 'tables'}."
        ),
        details={
            "manifest_validation": "VALID",
            "latex_preview_status": latex_status,
        },
    )


def inspect_path(path: Path) -> InspectionResult:
    """Identify and semantically validate one supplied root without writes."""

    root = path.resolve()
    if not _path_exists(root):
        raise ProductError(
            "FD-INSPECT-NOT-FOUND",
            "The inspected path does not exist.",
            ("Supply an existing repository or artifact root.",),
            {"path": str(root)},
        )
    if not _path_is_dir(root):
        raise ProductError(
            "FD-INSPECT-NOT-DIRECTORY",
            "The inspected path is not a directory.",
            ("Supply a repository, experiment, partial-run, or presentation root.",),
            {"path": str(root)},
        )

    signatures: list[str] = []
    if _repository_signature(root):
        signatures.append("repository")
    if _path_exists(root / _RUN_MANIFEST_NAME):
        signatures.append("experiment")
    elif _partial_experiment_signature(root):
        signatures.append("experiment-partial")
    if _presentation_signature(root):
        signatures.append("presentation")
    if len(signatures) > 1:
        raise ProductError(
            "FD-INSPECT-CONFLICT",
            "The path has conflicting semantic root signatures: "
            + ", ".join(signatures)
            + ".",
            ("Supply the exact root of only one supported path type.",),
            {"path": str(root), "detected_types": signatures},
        )
    if not signatures:
        raise ProductError(
            "FD-INSPECT-UNSUPPORTED",
            "The directory is not a supported inspection root.",
            (
                "Repository: expected pyproject.toml and src/fraud_detection/.",
                "Completed experiment: expected root-level RUN_MANIFEST.json.",
                "Partial experiment: expected a current phase manifest at its exact path.",
                "Presentation: expected root selection and stage manifests.",
            ),
            {"path": str(root)},
        )

    path_type = signatures[0]
    if path_type == "repository":
        return _inspect_repository(root)
    if path_type == "experiment":
        return _inspect_completed_experiment(root)
    if path_type == "experiment-partial":
        return _inspect_partial_experiment(root)
    return _inspect_presentation(root)


def display_path(root: Path, path: Path) -> str:
    """Display a repository-relative path when possible."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def safe_output_path(
    repository_root: Path,
    requested: Path,
    generated_root: Literal["outputs", "generated"],
) -> Path:
    """Resolve a new artifact path below an approved generated root."""

    repository_root = repository_root.resolve()
    target = (
        requested
        if requested.is_absolute()
        else repository_root / requested
    ).resolve()
    try:
        relative = target.relative_to(repository_root)
    except ValueError as exc:
        raise ProductError(
            "FD-OUTPUT-UNSAFE",
            f"Output must remain inside this repository below {generated_root}/.",
            (f"Choose a new path below {generated_root}/.",),
            {"path": str(target)},
        ) from exc
    if len(relative.parts) < 2 or relative.parts[0] != generated_root:
        raise ProductError(
            "FD-OUTPUT-UNSAFE",
            f"Output must be a child path below {generated_root}/.",
            (f"Choose a path such as {generated_root}/my-run.",),
            {"path": relative.as_posix()},
        )
    return target
