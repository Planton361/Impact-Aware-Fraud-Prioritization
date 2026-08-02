"""Computation-free planning for the complete frozen experiment."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from fraud_detection.artifacts import (
    display_path,
    find_repository_root,
    safe_output_path,
)
from fraud_detection.errors import ProductError
from fraud_detection.setup.environment import (
    PYTHON_CONTRACT,
    parse_pinned_requirements,
)

from ..config import (
    EXPECTED_RAW_SHA256,
    EffectiveExperimentConfig,
    ExperimentConfig,
    ExperimentProfileName,
    resolve_experiment_profile,
)

_EXECUTION = "fraud_detection.experiment.run_experiment"


def _task_counts(
    effective_config: EffectiveExperimentConfig,
) -> dict[str, int]:
    seed_count = len(effective_config.seeds)
    budget_count = len(effective_config.target_budgets)
    return {
        "inner_bce_fits": (
            seed_count
            * effective_config.inner_folds
            * (effective_config.bce_oof_folds + 1)
        ),
        "inner_ranker_fits": (
            seed_count
            * effective_config.inner_folds
            * budget_count
            * len(effective_config.enabled_gain_profiles)
        ),
        "selection_freeze_configurations": seed_count * budget_count,
        "final_bce_fits": (
            seed_count * (effective_config.bce_oof_folds + 1)
        ),
        "final_ranker_fits": seed_count * budget_count * 2,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_display_path(repository_root: Path, data_path: Path) -> str:
    try:
        return data_path.relative_to(repository_root).as_posix()
    except ValueError:
        return str(data_path)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Validated, computation-free description of one complete run."""

    repository_root: Path
    config: ExperimentConfig
    data_status: str
    actual_data_sha256: str | None
    prerequisite_errors: tuple[ProductError, ...]

    @property
    def data_path(self) -> Path:
        return self.config.data_path

    @property
    def output_path(self) -> Path:
        return self.config.output_root

    @property
    def profile(self) -> ExperimentProfileName:
        return self.config.profile

    @property
    def effective_config(self) -> EffectiveExperimentConfig:
        return self.config.effective_config

    def as_dict(self) -> dict[str, object]:
        effective_config = self.effective_config
        task_counts = _task_counts(effective_config)
        data_path = (
            None
            if effective_config.data_source_kind == "synthetic"
            else _data_display_path(
                self.repository_root,
                self.data_path,
            )
        )
        return {
            "phase": self.config.phase,
            "profile": self.profile,
            "effective_config": effective_config.as_dict(),
            "evidence_notice": _evidence_notice(effective_config),
            "seeds": list(effective_config.seeds),
            "budgets": list(effective_config.target_budgets),
            "task_counts": task_counts,
            "known_task_total": sum(task_counts.values()),
            "qa_task_total": None,
            "qa_task_note": (
                "The runner exposes integrity gates, not a discrete QA task total."
            ),
            "data": {
                "path": data_path,
                "status": self.data_status,
                "actual_sha256": self.actual_data_sha256,
                "expected_sha256": EXPECTED_RAW_SHA256,
            },
            "output_path": display_path(
                self.repository_root,
                self.output_path,
            ),
            "execution": _EXECUTION,
            "execution_notes": (
                [
                    "Runtime depends on the local environment; no duration is guaranteed.",
                    "Individual model fits are not resumable.",
                ]
                if effective_config.data_source_kind == "real"
                else []
            ),
            "prerequisites": [
                {
                    "code": error.code,
                    "summary": error.summary,
                    "recovery": list(error.recovery),
                    "details": error.details,
                }
                for error in self.prerequisite_errors
            ],
        }


def _evidence_notice(
    effective_config: EffectiveExperimentConfig,
) -> list[str]:
    if effective_config.data_source_kind == "synthetic":
        return [
            "Deterministic synthetic engineering data.",
            "Not thesis evidence.",
            "Not comparable with canonical empirical results.",
        ]
    if effective_config.evidence_classification == "thesis-evidentiary":
        return ["Canonical thesis-evidentiary profile."]
    return [
        "Engineering mini profile.",
        "Real canonical dataset.",
        "Not thesis evidence.",
        "Not comparable with canonical empirical results.",
    ]


def _environment_errors(repository_root: Path) -> list[ProductError]:
    errors: list[ProductError] = []
    for relative in (
        Path("environment/final_experiment_requirements.txt"),
        Path("environment/bootstrap_requirements.txt"),
    ):
        path = repository_root / relative
        if not path.is_file():
            errors.append(
                ProductError(
                    "FD-ENV-REQUIREMENTS",
                    "A pinned requirements file is missing.",
                    ("Restore the repository checkout before planning a run.",),
                    {"path": relative.as_posix()},
                )
            )
            continue
        try:
            pins = parse_pinned_requirements(path)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(
                ProductError(
                    "FD-ENV-REQUIREMENTS",
                    "A pinned requirements file is invalid.",
                    (
                        "Restore the registered requirements file.",
                        "Run fraud-detection setup before planning.",
                    ),
                    {"path": relative.as_posix(), "reason": str(exc)},
                )
            )
            continue
        for distribution, expected in pins:
            try:
                actual = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                errors.append(
                    ProductError(
                        "FD-ENV-MISSING",
                        f"{distribution} is not installed.",
                        ("Run fraud-detection setup before planning.",),
                        {
                            "distribution": distribution,
                            "expected": expected,
                        },
                    )
                )
            except Exception as exc:
                errors.append(
                    ProductError(
                        "FD-ENV-INSPECT",
                        f"{distribution} could not be inspected.",
                        (
                            "Run fraud-detection check --full and correct the "
                            "environment.",
                        ),
                        {"reason": str(exc)},
                    )
                )
            else:
                if actual != expected:
                    errors.append(
                        ProductError(
                            "FD-ENV-VERSION",
                            (
                                f"{distribution} has the wrong installed "
                                "version."
                            ),
                            ("Run fraud-detection setup before planning.",),
                            {
                                "distribution": distribution,
                                "actual": actual,
                                "expected": expected,
                            },
                        )
                    )
    return errors


def _unavailable_plan(
    *,
    data_path: Path,
    output_dir: Path,
    profile: ExperimentProfileName,
    repository_root: Path | None = None,
) -> ExperimentPlan:
    resolved_root = (repository_root or Path.cwd()).resolve()
    return ExperimentPlan(
        repository_root=resolved_root,
        config=ExperimentConfig(
            data_path=(
                data_path
                if data_path.is_absolute()
                else resolved_root / data_path
            ).resolve(),
            output_root=(
                output_dir
                if output_dir.is_absolute()
                else resolved_root / output_dir
            ).resolve(),
            phase="all",
            profile=profile,
        ),
        data_status="unavailable",
        actual_data_sha256=None,
        prerequisite_errors=(
            ProductError(
                "FD-ROOT-NOT-FOUND",
                "Repository root was not found.",
                ("Run the command from the repository checkout.",),
            ),
        ),
    )


def build_experiment_plan(
    *,
    data_path: Path,
    output_dir: Path,
    profile: ExperimentProfileName = "canonical",
    repository_root: Path | None = None,
    inspect_data: bool = True,
) -> ExperimentPlan:
    """Validate prerequisites and describe the direct complete-run call."""

    effective_config = resolve_experiment_profile(profile)
    if repository_root is None:
        resolved_root = find_repository_root()
    else:
        supplied_root = repository_root.resolve()
        discovered_root = find_repository_root(supplied_root)
        resolved_root = (
            supplied_root
            if discovered_root is not None and discovered_root == supplied_root
            else None
        )
    if resolved_root is None:
        return _unavailable_plan(
            data_path=data_path,
            output_dir=output_dir,
            profile=profile,
            repository_root=repository_root,
        )
    resolved_root = resolved_root.resolve()
    selected_data = (
        data_path
        if data_path.is_absolute()
        else resolved_root / data_path
    )
    resolved_data = (
        selected_data.absolute()
        if effective_config.data_source_kind == "synthetic" or not inspect_data
        else selected_data.resolve()
    )
    errors = _environment_errors(resolved_root)

    actual_hash: str | None = None
    if effective_config.data_source_kind == "synthetic":
        data_status = "synthetic"
    elif not inspect_data:
        data_status = "not_inspected"
    elif not resolved_data.is_file():
        data_status = "missing"
        errors.append(
            ProductError(
                "FD-DATA-MISSING",
                "The canonical experiment dataset is missing.",
                (
                    "Run fraud-detection setup.",
                    "Re-run fraud-detection run --profile canonical --dry-run "
                    "after setup succeeds.",
                ),
                {"path": display_path(resolved_root, resolved_data)},
            )
        )
    else:
        try:
            actual_hash = _sha256_file(resolved_data)
        except OSError as exc:
            data_status = "unreadable"
            errors.append(
                ProductError(
                    "FD-DATA-READ",
                    "The experiment dataset could not be hashed.",
                    (
                        "Check file permissions and local storage.",
                        "Run fraud-detection check --require-data before "
                        "retrying the run.",
                    ),
                    {
                        "path": display_path(resolved_root, resolved_data),
                        "reason": str(exc),
                    },
                )
            )
        else:
            if actual_hash == EXPECTED_RAW_SHA256:
                data_status = "verified"
            else:
                data_status = "hash_mismatch"
                errors.append(
                    ProductError(
                        "FD-DATA-HASH",
                        "The experiment dataset has the wrong SHA-256.",
                        (
                            "Move the mismatching CSV to a safe location "
                            "outside data/.",
                            "Run fraud-detection setup to obtain the canonical "
                            "CSV.",
                        ),
                        {
                            "path": display_path(
                                resolved_root,
                                resolved_data,
                            ),
                            "actual_sha256": actual_hash,
                            "expected_sha256": EXPECTED_RAW_SHA256,
                        },
                    )
                )
    try:
        generated_root = (
            "outputs"
            if effective_config.evidence_classification == "thesis-evidentiary"
            else "generated"
        )
        resolved_output = safe_output_path(
            resolved_root,
            output_dir,
            generated_root,
        )
    except ProductError as exc:
        errors.append(exc)
        resolved_output = (
            output_dir
            if output_dir.is_absolute()
            else resolved_root / output_dir
        ).resolve()
    else:
        if resolved_output.exists():
            errors.append(
                ProductError(
                    "FD-OUTPUT-EXISTS",
                    "A complete run requires a new output directory.",
                    (
                        "Choose --output with a new child path below "
                        f"{generated_root}/.",
                    ),
                    {
                        "path": display_path(
                            resolved_root,
                            resolved_output,
                        )
                    },
                )
            )
    if sys.version_info[:2] != (3, 12):
        errors.append(
            ProductError(
                "FD-PYTHON-VERSION",
                (
                    f"Python {sys.version_info.major}."
                    f"{sys.version_info.minor} does not satisfy "
                    f"{PYTHON_CONTRACT}."
                ),
                (
                    "Run the command from the bootstrap-created Python 3.12 "
                    "environment.",
                ),
            )
        )
    return ExperimentPlan(
        repository_root=resolved_root,
        config=ExperimentConfig(
            data_path=resolved_data,
            output_root=resolved_output,
            phase="all",
            profile=profile,
            repository_root=resolved_root,
        ),
        data_status=data_status,
        actual_data_sha256=actual_hash,
        prerequisite_errors=tuple(errors),
    )
