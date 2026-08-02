from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.experiment.config import (
    EXPECTED_DEDUPLICATED_SHA256,
    MATCHED_METRIC_COLUMNS,
    METHOD_AMOUNT_GAIN,
    METHOD_BASELINE,
    METHOD_FAMILIES,
    METHOD_P_ONLY,
    EffectiveExperimentConfig,
    resolve_experiment_profile,
)
from fraud_detection.experiment.evaluation import aggregation
from fraud_detection.experiment.records import score_path

pytestmark = pytest.mark.unit


def _synthetic_seed_result(
    seed: int,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    tie_rows = []
    replacement_rows = []
    boundary_rows = []
    high_legit_rows = []
    hard_rows = []
    global_rows = []
    for budget in effective_config.target_budgets:
        for method_index, method_family in enumerate(METHOD_FAMILIES):
            value = float(seed + 1) + method_index / 10.0 + budget / 1000.0
            metric_row = {
                "seed": seed,
                "target_budget": budget,
                "primary_budget": (
                    budget in effective_config.primary_budgets
                ),
                "method_family": method_family,
                "score_path": score_path(method_family, budget),
                "selected_gain": (
                    "linear"
                    if method_family in {METHOD_P_ONLY, METHOD_AMOUNT_GAIN}
                    else "not_applicable"
                ),
                "selection_status": "SELECTED",
            }
            metric_row.update(
                {metric: value for metric in MATCHED_METRIC_COLUMNS}
            )
            metric_rows.append(metric_row)
            tie_rows.append(
                {
                    "seed": seed,
                    "target_budget": budget,
                    "method_family": method_family,
                    "unique_raw_ranker_scores": value,
                    "cutoff_tie_size": value,
                    "cutoff_tie_rank_min": value,
                    "cutoff_tie_rank_max": value,
                }
            )
            high_legit_rows.append(
                {
                    "seed": seed,
                    "target_budget": budget,
                    "method_family": method_family,
                    "legit_count_at_k": value,
                    "high_amount_legit_count_at_k": value,
                    "mean_legit_amount_at_k": value,
                }
            )
            hard_rows.append(
                {
                    "seed": seed,
                    "target_budget": budget,
                    "method_family": method_family,
                    "prevented_loss_ratio_at_k": value,
                    "fraud_amount_sum_at_k": value,
                    "amount_ndcg_at_k": value,
                    "q90_captured_ratio_at_k": value,
                    "q90_amount_ndcg_at_k": value,
                }
            )
            global_rows.append(
                {
                    "seed": seed,
                    "target_budget": budget,
                    "method_family": method_family,
                    "metric": "roc_auc_of_final_order",
                    "value": value,
                }
            )
            if method_family != METHOD_BASELINE:
                for subset in ("added_vs_bce", "removed_from_bce"):
                    replacement_rows.append(
                        {
                            "seed": seed,
                            "target_budget": budget,
                            "method_family": method_family,
                            "subset": subset,
                            "case_count": value,
                            "fraud_count": value,
                            "fraud_amount_sum": value,
                            "legit_count": value,
                        }
                    )
                boundary_rows.append(
                    {
                        "seed": seed,
                        "target_budget": budget,
                        "method_family": method_family,
                        "rank_shift_vs_bce": value,
                        "crossed_into_topk": seed % 2 == 0,
                        "dropped_from_topk": seed % 2 == 1,
                    }
                )
    model_rows = [
        {"seed": seed, "target_budget": budget}
        for budget in effective_config.target_budgets
    ]
    return {
        "metrics": pd.DataFrame(metric_rows),
        "amount_models": pd.DataFrame(
            [{**row, "model_type": "amount_gain"} for row in model_rows]
        ),
        "p_only_models": pd.DataFrame(
            [{**row, "model_type": "p_only"} for row in model_rows]
        ),
        "pool_summary": pd.DataFrame([{"seed": seed}]),
        "ties": pd.DataFrame(tie_rows),
        "replacements": pd.DataFrame(replacement_rows),
        "boundaries": pd.DataFrame(boundary_rows),
        "high_legit": pd.DataFrame(high_legit_rows),
        "hard": pd.DataFrame(hard_rows),
        "global_metrics": pd.DataFrame(global_rows),
    }


def test_aggregation_preserves_rows_paths_ddof_and_paired_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "unwritten-output"
    written: dict[str, pd.DataFrame] = {}

    def capture(path: Path, frame: pd.DataFrame) -> None:
        written[path.relative_to(output_root).as_posix()] = frame.copy()

    def read_selection(path: Path) -> pd.DataFrame:
        if path.name == "selected_gain_by_seed_budget.csv":
            return pd.DataFrame([{"seed": 0, "target_budget": 5}])
        return pd.DataFrame([{"target_budget": 5, "selected_gain": "linear"}])

    monkeypatch.setattr(aggregation, "_write_csv", capture)
    monkeypatch.setattr(aggregation.pd, "read_csv", read_selection)
    effective_config = resolve_experiment_profile("canonical")
    aggregation.aggregate_final_outputs(
        output_root=output_root,
        seed_results=[
            _synthetic_seed_result(seed, effective_config)
            for seed in effective_config.seeds
        ],
        effective_config=effective_config,
        data_sha256=EXPECTED_DEDUPLICATED_SHA256,
    )

    aggregated = written["tables/all_budget_matched_results.csv"]
    assert len(aggregated) == 28
    row = aggregated.loc[
        (aggregated["target_budget"] == 5)
        & (aggregated["method_family"] == METHOD_BASELINE)
    ].iloc[0]
    expected = np.array([43.005, 8.005, 14.005, 124.005, 203.005])
    assert row["frauds_at_k_std"] == pytest.approx(np.std(expected, ddof=1))
    inventory = written["comparison/score_path_inventory.csv"]
    assert (
        inventory.loc[
            inventory["method_family"] == METHOD_BASELINE,
            "score_path",
        ].unique().tolist()
        == [METHOD_BASELINE]
    )
    paired = written["tables/paired_plr_deltas_vs_bce.csv"]
    order = list(
        paired[["target_budget", "method_family", "seed"]].itertuples(
            index=False,
            name=None,
        )
    )
    assert order == sorted(order)
    assert not output_root.exists()


def test_one_seed_aggregation_uses_profile_grid_and_preserves_nan_sd() -> None:
    smoke = resolve_experiment_profile("smoke-synthetic")
    seed_metrics = _synthetic_seed_result(
        smoke.seeds[0],
        smoke,
    )["metrics"]

    aggregated = aggregation._aggregate_matched_metrics(seed_metrics, smoke)

    assert len(aggregated) == 3 * len(METHOD_FAMILIES)
    assert aggregated.filter(regex="_std$").isna().all().all()
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregation._aggregate_matched_metrics(seed_metrics.iloc[:-1], smoke)
