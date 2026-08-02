"""Public configuration for the deterministic serial experiment."""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ExperimentPhase = Literal["inner", "final", "qa", "all"]
ExperimentEventSink = Callable[[str, Mapping[str, object]], None]
ExperimentProfileName = Literal[
    "canonical",
    "smoke-synthetic",
    "mini-real",
]
DataSourceKind = Literal["real", "synthetic"]
EvidenceClassification = Literal[
    "thesis-evidentiary",
    "non-evidentiary",
    "engineering mini profile — not thesis evidence",
]

EXPERIMENT_PROFILE_NAMES: tuple[ExperimentProfileName, ...] = (
    "canonical",
    "smoke-synthetic",
    "mini-real",
)


RANKER_SCOPE = "candidate_rerank"
CANDIDATE_POOL_SIZE = 1000
TARGET_BUDGETS = (5, 10, 20, 50, 100, 200, 500)
PRIMARY_BUDGETS = (20, 50, 100)
SUPPLEMENTARY_BUDGETS = (5, 10, 200, 500)
GAIN_PROFILES: dict[str, tuple[int, int, int, int, int]] = {
    "exponential": (0, 1, 3, 7, 15),
    "linear": (0, 1, 2, 3, 4),
}

RANKER_MAX_ESTIMATORS = 500
RANKER_LEARNING_RATE = 0.05
RANKER_NUM_LEAVES = 7
RANKER_MIN_CHILD_SAMPLES = 20
RANKER_MIN_CHILD_WEIGHT = 1e-3
RANKER_REG_LAMBDA = 0.0
RANKER_N_JOBS = 1
RANKER_VERBOSITY = -1
RANKER_EARLY_STOPPING_ROUNDS = 50

EXPECTED_RAW_SHA256 = (
    "76274b691b16a6c49d3f159c883398e03"
    "ccd6d1ee12d9d8ee38f4b4b98551a89"
)

EXPECTED_DEDUPLICATED_SHA256 = (
    "525bfe7a3155e7a5b01cf52ffdacec38a"
    "09725667b307cb6c553047c28120875"
)

OUTER_SEEDS = (42, 7, 13, 123, 202)

INNER_FOLDS = 3

BCE_FEATURES = tuple(f"V{i}" for i in range(1, 29))

BCE_LEARNING_RATE = 0.1

BCE_TOL = 1e-6

BCE_MAX_ITER = 10_000

BCE_L2_ALPHA = 0.0

BCE_OOF_FOLDS = 5


@dataclass(frozen=True, slots=True)
class EffectiveExperimentConfig:
    """Immutable scientific settings selected by an experiment profile."""

    profile_name: ExperimentProfileName
    evidence_classification: EvidenceClassification
    data_source_kind: DataSourceKind
    synthetic_row_target: int | None
    synthetic_generation_seed: int | None
    seeds: tuple[int, ...]
    target_budgets: tuple[int, ...]
    primary_budgets: tuple[int, ...]
    supplementary_budgets: tuple[int, ...]
    bce_oof_folds: int
    inner_folds: int
    candidate_pool_size: int
    enabled_gain_profiles: tuple[str, ...]
    ranker_max_estimators: int
    ranker_early_stopping_rounds: int

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_name": self.profile_name,
            "evidence_classification": self.evidence_classification,
            "data_source_kind": self.data_source_kind,
            "synthetic_row_target": self.synthetic_row_target,
            "synthetic_generation_seed": self.synthetic_generation_seed,
            "seeds": list(self.seeds),
            "target_budgets": list(self.target_budgets),
            "primary_budgets": list(self.primary_budgets),
            "supplementary_budgets": list(self.supplementary_budgets),
            "bce_oof_folds": self.bce_oof_folds,
            "inner_folds": self.inner_folds,
            "candidate_pool_size": self.candidate_pool_size,
            "enabled_gain_profiles": list(self.enabled_gain_profiles),
            "ranker_max_estimators": self.ranker_max_estimators,
            "ranker_early_stopping_rounds": (
                self.ranker_early_stopping_rounds
            ),
        }


def resolve_experiment_profile(
    profile_name: str,
) -> EffectiveExperimentConfig:
    """Resolve one public profile name without performing any operations."""

    if profile_name == "canonical":
        return EffectiveExperimentConfig(
            profile_name="canonical",
            evidence_classification="thesis-evidentiary",
            data_source_kind="real",
            synthetic_row_target=None,
            synthetic_generation_seed=None,
            seeds=OUTER_SEEDS,
            target_budgets=TARGET_BUDGETS,
            primary_budgets=PRIMARY_BUDGETS,
            supplementary_budgets=SUPPLEMENTARY_BUDGETS,
            bce_oof_folds=BCE_OOF_FOLDS,
            inner_folds=INNER_FOLDS,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
            enabled_gain_profiles=tuple(GAIN_PROFILES),
            ranker_max_estimators=RANKER_MAX_ESTIMATORS,
            ranker_early_stopping_rounds=RANKER_EARLY_STOPPING_ROUNDS,
        )
    if profile_name == "smoke-synthetic":
        return EffectiveExperimentConfig(
            profile_name="smoke-synthetic",
            evidence_classification="non-evidentiary",
            data_source_kind="synthetic",
            synthetic_row_target=5_000,
            synthetic_generation_seed=314_159,
            seeds=(42,),
            target_budgets=PRIMARY_BUDGETS,
            primary_budgets=PRIMARY_BUDGETS,
            supplementary_budgets=(),
            bce_oof_folds=2,
            inner_folds=2,
            candidate_pool_size=200,
            enabled_gain_profiles=tuple(GAIN_PROFILES),
            ranker_max_estimators=30,
            ranker_early_stopping_rounds=5,
        )
    if profile_name == "mini-real":
        return EffectiveExperimentConfig(
            profile_name="mini-real",
            evidence_classification=(
                "engineering mini profile — not thesis evidence"
            ),
            data_source_kind="real",
            synthetic_row_target=None,
            synthetic_generation_seed=None,
            seeds=OUTER_SEEDS[:3],
            target_budgets=PRIMARY_BUDGETS,
            primary_budgets=PRIMARY_BUDGETS,
            supplementary_budgets=(),
            bce_oof_folds=BCE_OOF_FOLDS,
            inner_folds=INNER_FOLDS,
            candidate_pool_size=CANDIDATE_POOL_SIZE,
            enabled_gain_profiles=("linear",),
            ranker_max_estimators=RANKER_MAX_ESTIMATORS,
            ranker_early_stopping_rounds=RANKER_EARLY_STOPPING_ROUNDS,
        )
    expected = ", ".join(EXPERIMENT_PROFILE_NAMES)
    raise ValueError(
        f"Unknown experiment profile {profile_name!r}; expected one of: "
        f"{expected}."
    )


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Paths, phase, profile, and structured-event boundary for one run."""

    data_path: Path
    output_root: Path
    phase: ExperimentPhase = "all"
    event_sink: ExperimentEventSink | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    profile: ExperimentProfileName = "canonical"
    repository_root: Path | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        resolve_experiment_profile(self.profile)

    @property
    def effective_config(self) -> EffectiveExperimentConfig:
        return resolve_experiment_profile(self.profile)


METHOD_BASELINE = "baseline_bce_probability"

METHOD_P_ONLY = "selected_candidate_p_only"

METHOD_AMOUNT_GAIN = "selected_candidate_amount_gain"

METHOD_FIXED = "candidate_postprocessing_p_times_log_amount"

METHOD_FAMILIES = (
    METHOD_BASELINE,
    METHOD_P_ONLY,
    METHOD_AMOUNT_GAIN,
    METHOD_FIXED,
)

MATCHED_METRIC_COLUMNS = (
    "prevented_loss_ratio_at_k",
    "frauds_at_k",
    "precision_at_k",
    "recall_at_k",
    "fraud_amount_sum_at_k",
    "legit_count_at_k",
    "amount_ndcg_at_k",
    "q90_captured_ratio_at_k",
    "q90_amount_ndcg_at_k",
    "high_amount_legit_count_at_k",
    "mean_legit_amount_at_k",
    "unique_raw_ranker_scores",
    "cutoff_tie_size",
)


_EVENT_SINK: contextvars.ContextVar[
    Callable[[str, dict[str, Any]], None] | None
] = contextvars.ContextVar("fraud_detection_experiment_event_sink", default=None)


@contextmanager
def _experiment_event_sink(
    sink: Callable[[str, dict[str, Any]], None] | None,
) -> Iterator[None]:
    token = _EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _EVENT_SINK.reset(token)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _status_utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _status_value(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def _emit_experiment_status(
    level: str,
    event: str,
    **fields: object,
) -> None:
    sink = _EVENT_SINK.get()
    if sink is not None:
        sink(
            "status",
            {
                "level": level,
                "event": event,
                "fields": fields,
            },
        )
        return
    line = " ".join(
        [level, event]
        + [f"{name}={_status_value(value)}" for name, value in fields.items()]
    )
    try:
        print(line, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _emit_experiment_log(message: str) -> None:
    sink = _EVENT_SINK.get()
    if sink is not None:
        sink("log", {"message": message})
        return
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def _inner_bce_completed(
    outer_seed: int,
    inner_fold: int,
    fit_number: int,
    effective_config: EffectiveExperimentConfig,
) -> int:
    return (
        effective_config.seeds.index(outer_seed)
        * effective_config.inner_folds
        * (effective_config.bce_oof_folds + 1)
        + (inner_fold - 1) * (effective_config.bce_oof_folds + 1)
        + fit_number
    )


def _inner_ranker_completed(
    outer_seed: int,
    inner_fold: int,
    target_budget: int,
    gain_profile: str,
    effective_config: EffectiveExperimentConfig,
) -> int:
    fits_per_fold = len(effective_config.target_budgets) * len(
        effective_config.enabled_gain_profiles
    )
    return (
        effective_config.seeds.index(outer_seed)
        * effective_config.inner_folds
        * fits_per_fold
        + (inner_fold - 1) * fits_per_fold
        + effective_config.target_budgets.index(target_budget)
        * len(effective_config.enabled_gain_profiles)
        + effective_config.enabled_gain_profiles.index(gain_profile)
        + 1
    )


def _selection_freeze_completed(
    outer_seed: int,
    target_budget: int,
    effective_config: EffectiveExperimentConfig,
) -> int:
    return (
        effective_config.seeds.index(outer_seed)
        * len(effective_config.target_budgets)
        + effective_config.target_budgets.index(target_budget)
        + 1
    )


def _final_bce_completed(
    outer_seed: int,
    fit_number: int,
    effective_config: EffectiveExperimentConfig,
) -> int:
    return (
        effective_config.seeds.index(outer_seed)
        * (effective_config.bce_oof_folds + 1)
        + fit_number
    )


def _final_ranker_completed(
    outer_seed: int,
    target_budget: int,
    path_index: int,
    effective_config: EffectiveExperimentConfig,
) -> int:
    paths_per_seed = len(effective_config.target_budgets) * 2
    return (
        effective_config.seeds.index(outer_seed) * paths_per_seed
        + effective_config.target_budgets.index(target_budget) * 2
        + path_index
        + 1
    )
