import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import (
    TARGET_BUDGETS,
    resolve_experiment_profile,
)
from fraud_detection.experiment.prioritization import selection as selection_module
from fraud_detection.experiment.prioritization.selection import (
    eval_at_for_budget,
    select_gain_from_inner_results,
    truncation_for_budget,
)

pytestmark = pytest.mark.unit

_CANONICAL = resolve_experiment_profile("canonical")


def _selection_frame(
    *,
    exponential_plr: tuple[float, float, float] = (0.7, 0.7, 0.7),
    linear_plr: tuple[float, float, float] = (0.6, 0.6, 0.6),
    exponential_fraud: tuple[int, int, int] = (9, 9, 9),
    linear_fraud: tuple[int, int, int] = (10, 10, 10),
    exponential_iterations: tuple[int, int, int] = (20, 21, 22),
    linear_iterations: tuple[int, int, int] = (30, 31, 32),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gain, plr, fraud, iterations in (
        (
            "exponential",
            exponential_plr,
            exponential_fraud,
            exponential_iterations,
        ),
        ("linear", linear_plr, linear_fraud, linear_iterations),
    ):
        for fold in range(3):
            rows.append(
                {
                    "gain_profile": gain,
                    "inner_fold": fold + 1,
                    "prevented_loss_ratio_at_k": plr[fold],
                    "frauds_at_k": fraud[fold],
                    "bce_prevented_loss_ratio_at_k": 0.5,
                    "bce_frauds_at_k": 10,
                    "amount_ndcg_at_k": 0.8,
                    "cutoff_tie_size": 1,
                    "best_iteration": iterations[fold],
                    "fit_valid": True,
                }
            )
    return pd.DataFrame(rows)


def test_budget_truncation_and_eval_at_are_frozen() -> None:
    assert TARGET_BUDGETS == (5, 10, 20, 50, 100, 200, 500)
    for budget in TARGET_BUDGETS:
        assert truncation_for_budget(budget, _CANONICAL) == budget + 3
        assert eval_at_for_budget(budget, _CANONICAL) == (budget,)
    for invalid in (True, 0, 6, 501):
        with pytest.raises(ValueError):
            truncation_for_budget(invalid, _CANONICAL)


def test_selection_preserves_deltas_retention_tiebreak_and_result_schemas() -> None:
    selection, enriched, summary = select_gain_from_inner_results(
        _selection_frame(),
        _CANONICAL,
    )
    assert selection["selected_gain"] == "linear"
    assert selection["selection_status"] == "POSITIVE_INNER_PLR_LIFT"
    assert selection["final_n_estimators"] == 31
    np.testing.assert_allclose(
        enriched.loc[enriched["gain_profile"] == "linear", "delta_plr"],
        [0.1, 0.1, 0.1],
    )
    np.testing.assert_allclose(
        enriched.loc[enriched["gain_profile"] == "linear", "fraud_retention"],
        [1.0, 1.0, 1.0],
    )
    assert list(enriched.columns[-2:]) == ["delta_plr", "fraud_retention"]
    assert list(summary.columns) == [
        "gain_profile",
        "mean_fraud_retention",
        "min_fraud_retention",
        "mean_delta_plr",
        "positive_delta_plr_fold_count",
        "mean_amount_ndcg_at_k",
        "mean_cutoff_tie_size",
        "all_configured_fits_valid",
        "all_selection_metrics_finite",
        "plr_eligible",
    ]
    assert summary["all_configured_fits_valid"].eq(True).all()


def test_positive_lift_gate_and_fallback_status_are_unchanged() -> None:
    positive = _selection_frame(
        exponential_plr=(0.6, 0.6, 0.4),
        linear_plr=(0.5, 0.5, 0.5),
        exponential_fraud=(10, 10, 10),
        linear_fraud=(9, 9, 9),
    )
    selection, _, summary = select_gain_from_inner_results(
        positive,
        _CANONICAL,
    )
    exponential = summary.loc[summary["gain_profile"] == "exponential"].iloc[0]
    assert exponential["positive_delta_plr_fold_count"] == 2
    assert exponential["plr_eligible"]
    assert selection["selected_gain"] == "exponential"

    fallback = _selection_frame(
        exponential_plr=(0.5, 0.5, 0.5),
        linear_plr=(0.5, 0.5, 0.5),
    )
    fallback_selection, _, fallback_summary = select_gain_from_inner_results(
        fallback,
        _CANONICAL,
    )
    assert not fallback_summary["plr_eligible"].any()
    assert (
        fallback_selection["selection_status"]
        == "NO_INNER_VALIDATED_POSITIVE_PLR_LIFT"
    )


def test_exponential_wins_the_final_exact_tie() -> None:
    tied = _selection_frame(
        exponential_plr=(0.6, 0.6, 0.6),
        linear_plr=(0.6, 0.6, 0.6),
        exponential_fraud=(10, 10, 10),
        linear_fraud=(10, 10, 10),
        exponential_iterations=(40, 41, 42),
        linear_iterations=(40, 41, 42),
    )
    selection, _, _ = select_gain_from_inner_results(tied, _CANONICAL)
    assert selection["selected_gain"] == "exponential"
    assert selection["final_n_estimators"] == 41


def test_invalid_gain_fold_shape_or_non_finite_input_is_rejected() -> None:
    invalid_frames = []
    missing_gain = _selection_frame()
    invalid_frames.append(missing_gain.loc[missing_gain["gain_profile"] == "linear"])
    missing_fold = _selection_frame()
    invalid_frames.append(missing_fold.loc[missing_fold["inner_fold"] != 3])
    duplicate = pd.concat(
        [_selection_frame(), _selection_frame().iloc[[0]]],
        ignore_index=True,
    )
    invalid_frames.append(duplicate)
    non_finite = _selection_frame()
    non_finite.loc[0, "amount_ndcg_at_k"] = np.nan
    invalid_frames.append(non_finite)
    invalid_fit = _selection_frame()
    invalid_fit.loc[0, "fit_valid"] = False
    invalid_frames.append(invalid_fit)

    for frame in invalid_frames:
        with pytest.raises(ValueError):
            select_gain_from_inner_results(frame, _CANONICAL)


def test_mini_real_selection_accepts_only_linear_gain() -> None:
    mini = resolve_experiment_profile("mini-real")
    linear = _selection_frame().loc[
        lambda frame: frame["gain_profile"] == "linear"
    ]

    selection, _enriched, summary = select_gain_from_inner_results(
        linear,
        mini,
    )

    assert selection["selected_gain"] == "linear"
    assert summary["gain_profile"].tolist() == ["linear"]


def test_smoke_inner_folds_and_pool_size_reach_selection_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundaryReached(RuntimeError):
        pass

    smoke = resolve_experiment_profile("smoke-synthetic")
    observed_splits: list[int] = []
    real_build_candidate_pool = selection_module.build_candidate_pool

    class RecordingSplitter:
        def __init__(
            self,
            *,
            n_splits: int,
            shuffle: bool,
            random_state: int,
        ) -> None:
            observed_splits.append(n_splits)
            assert shuffle is True
            assert random_state == 100042

        def split(self, _features: object, _labels: object):
            return iter(((np.array([0, 1]), np.array([2, 3])),))

    def fake_bce(**_kwargs: object) -> dict[str, object]:
        return {
            "p_train_oof": np.array([0.2, 0.8]),
            "p_validation": np.array([0.3, 0.7]),
            "diagnostics": pd.DataFrame(),
            "loss_history": pd.DataFrame(),
            "oof_fold_number": np.array([1, 2]),
        }

    def capture_pool(
        _scores: object,
        _index: object,
        *,
        candidate_pool_size: int,
    ) -> None:
        assert candidate_pool_size == 200
        raise BoundaryReached

    monkeypatch.setattr(selection_module, "StratifiedKFold", RecordingSplitter)
    monkeypatch.setattr(
        selection_module,
        "_validate_outer_train_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(selection_module, "fit_inner_bce_fold", fake_bce)
    monkeypatch.setattr(selection_module, "build_candidate_pool", capture_pool)
    dataframe = pd.DataFrame({"Class": [0, 1, 0, 1]})

    with pytest.raises(BoundaryReached):
        selection_module._inner_validation_for_seed(
            dataframe=dataframe,
            output_root=tmp_path,
            outer_seed=42,
            train_index=np.arange(4),
            effective_config=smoke,
        )

    assert observed_splits == [2]
    monkeypatch.setattr(
        selection_module,
        "build_candidate_pool",
        real_build_candidate_pool,
    )

    inner_train = pd.DataFrame({"Class": [0, 1]}, index=[10, 11])
    inner_validation = pd.DataFrame({"Class": [1, 0]}, index=[12, 13])
    for profile in ("canonical", "smoke-synthetic"):
        effective = resolve_experiment_profile(profile)
        bce_result = {
            "p_train_oof": np.array([0.2, 0.8]),
            "p_validation": np.array([0.3, 0.7]),
            "oof_fold_number": np.array([1, 2]),
            "diagnostics": pd.DataFrame(
                {
                    "converged_by_tolerance": [
                        True
                    ]
                    * (effective.bce_oof_folds + 1)
                }
            ),
            "scaler_mean": np.array([0.0]),
            "scaler_scale": np.array([1.0]),
        }
        train_scores, validation_scores = selection_module._inner_score_frames(
            outer_seed=42,
            inner_fold=1,
            inner_train=inner_train,
            inner_validation=inner_validation,
            inner_train_outer_positions=np.array([0, 1]),
            validation_outer_positions=np.array([2, 3]),
            bce_result=bce_result,
        )
        assert train_scores["score_source"].eq("inner_train_oof_bce").all()
        train_pool = selection_module.build_candidate_pool(
            bce_result["p_train_oof"],
            inner_train.index,
            candidate_pool_size=1,
        )
        validation_pool = selection_module.build_candidate_pool(
            bce_result["p_validation"],
            inner_validation.index,
            candidate_pool_size=1,
        )
        output_root = tmp_path / f"metadata-{profile}"
        selection_module._write_inner_fold_artifacts(
            output_root=output_root,
            outer_seed=42,
            inner_fold=1,
            train_scores=train_scores,
            validation_scores=validation_scores,
            train_pool=train_pool,
            validation_pool=validation_pool,
            thresholds=np.array([1.0, 2.0, 3.0]),
            train_relevance=np.array([0, 1]),
            validation_relevance=np.array([1, 0]),
            bce_result=bce_result,
            effective_config=effective,
        )
        metadata = json.loads(
            (
                output_root
                / "inner_validation"
                / "seed_42"
                / "fold_1"
                / "metadata.json"
            ).read_text(encoding="utf-8")
        )
        inner_bce = metadata["inner_bce_oof"]
        assert inner_bce["folds"] == effective.bce_oof_folds
        assert inner_bce["configured_fit_count"] == effective.bce_oof_folds + 1
        assert inner_bce["all_configured_bce_fits_tolerance_converged"] is True
        assert not any(
            "fits_tolerance_converged" in key
            and key != "all_configured_bce_fits_tolerance_converged"
            for key in inner_bce
        )

    smoke_frame = _selection_frame(
        exponential_iterations=(10, 11, 12),
        linear_iterations=(13, 14, 15),
    ).loc[lambda frame: frame["inner_fold"] <= 2]
    _smoke_selection, _smoke_enriched, smoke_summary = (
        select_gain_from_inner_results(smoke_frame, smoke)
    )
    assert smoke_summary["all_configured_fits_valid"].eq(True).all()
    assert not any(
        key.endswith("_three_fits_valid") for key in smoke_summary.columns
    )

    half_up_frame = _selection_frame(
        exponential_iterations=(10, 11, 12),
        linear_iterations=(20, 21, 22),
    ).loc[lambda frame: frame["inner_fold"] <= 2]
    half_up_selection, _, _ = select_gain_from_inner_results(
        half_up_frame,
        smoke,
    )
    assert np.median([20, 21]) == 20.5
    assert half_up_selection["selected_gain"] == "linear"
    assert half_up_selection["selection_status"] == "POSITIVE_INNER_PLR_LIFT"
    assert half_up_selection["best_iteration_fold_1"] == 20
    assert half_up_selection["best_iteration_fold_2"] == 21
    assert half_up_selection["final_n_estimators"] == 21
    assert half_up_selection["final_n_estimators"] != round(20.5)
    assert 1 <= half_up_selection["final_n_estimators"] <= smoke.ranker_max_estimators
