"""Deterministic serial experiment runner."""

from typing import Any

__all__ = [
    "ExperimentConfig",
    "ExperimentPhase",
    "ExperimentPlan",
    "ExperimentResult",
    "build_experiment_plan",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {"ExperimentConfig", "ExperimentPhase"}:
        from .config import ExperimentConfig, ExperimentPhase

        value = {
            "ExperimentConfig": ExperimentConfig,
            "ExperimentPhase": ExperimentPhase,
        }[name]
    elif name == "ExperimentPlan":
        from .execution.planning import ExperimentPlan

        value = ExperimentPlan
    elif name == "ExperimentResult":
        from .records import ExperimentResult

        value = ExperimentResult
    elif name == "build_experiment_plan":
        from .execution.planning import build_experiment_plan

        value = build_experiment_plan
    elif name == "run_experiment":
        from .execution.pipeline import run_experiment

        value = run_experiment
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
