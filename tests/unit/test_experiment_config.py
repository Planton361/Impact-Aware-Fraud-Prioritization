from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from fraud_detection.experiment.config import (
    EXPERIMENT_PROFILE_NAMES,
    EffectiveExperimentConfig,
    ExperimentConfig,
    resolve_experiment_profile,
)

pytestmark = pytest.mark.unit


def test_public_profile_names_and_canonical_default(tmp_path: Path) -> None:
    assert EXPERIMENT_PROFILE_NAMES == (
        "canonical",
        "smoke-synthetic",
        "mini-real",
    )
    config = ExperimentConfig(
        data_path=tmp_path / "creditcard.csv",
        output_root=tmp_path / "outputs" / "run",
    )
    assert config.profile == "canonical"
    assert config.effective_config == resolve_experiment_profile("canonical")


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    (
        (
            "canonical",
            EffectiveExperimentConfig(
                profile_name="canonical",
                evidence_classification="thesis-evidentiary",
                data_source_kind="real",
                synthetic_row_target=None,
                synthetic_generation_seed=None,
                seeds=(42, 7, 13, 123, 202),
                target_budgets=(5, 10, 20, 50, 100, 200, 500),
                primary_budgets=(20, 50, 100),
                supplementary_budgets=(5, 10, 200, 500),
                bce_oof_folds=5,
                inner_folds=3,
                candidate_pool_size=1000,
                enabled_gain_profiles=("exponential", "linear"),
                ranker_max_estimators=500,
                ranker_early_stopping_rounds=50,
            ),
        ),
        (
            "smoke-synthetic",
            EffectiveExperimentConfig(
                profile_name="smoke-synthetic",
                evidence_classification="non-evidentiary",
                data_source_kind="synthetic",
                synthetic_row_target=5000,
                synthetic_generation_seed=314159,
                seeds=(42,),
                target_budgets=(20, 50, 100),
                primary_budgets=(20, 50, 100),
                supplementary_budgets=(),
                bce_oof_folds=2,
                inner_folds=2,
                candidate_pool_size=200,
                enabled_gain_profiles=("exponential", "linear"),
                ranker_max_estimators=30,
                ranker_early_stopping_rounds=5,
            ),
        ),
        (
            "mini-real",
            EffectiveExperimentConfig(
                profile_name="mini-real",
                evidence_classification=(
                    "engineering mini profile — not thesis evidence"
                ),
                data_source_kind="real",
                synthetic_row_target=None,
                synthetic_generation_seed=None,
                seeds=(42, 7, 13),
                target_budgets=(20, 50, 100),
                primary_budgets=(20, 50, 100),
                supplementary_budgets=(),
                bce_oof_folds=5,
                inner_folds=3,
                candidate_pool_size=1000,
                enabled_gain_profiles=("linear",),
                ranker_max_estimators=500,
                ranker_early_stopping_rounds=50,
            ),
        ),
    ),
)
def test_profile_resolves_exact_effective_configuration(
    profile_name: str,
    expected: EffectiveExperimentConfig,
) -> None:
    assert resolve_experiment_profile(profile_name) == expected
    if profile_name != "canonical":
        assert "eviden" in expected.evidence_classification


def test_resolved_configuration_is_immutable() -> None:
    effective = resolve_experiment_profile("canonical")

    with pytest.raises(FrozenInstanceError):
        effective.candidate_pool_size = 200  # type: ignore[misc]


def test_unknown_profile_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown experiment profile"):
        resolve_experiment_profile("unknown")
    with pytest.raises(ValueError, match="Unknown experiment profile"):
        ExperimentConfig(
            data_path=tmp_path / "creditcard.csv",
            output_root=tmp_path / "outputs" / "run",
            profile="unknown",  # type: ignore[arg-type]
        )
