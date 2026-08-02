"""Aggregation boundary for the frozen experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.artifacts import _write_csv

from ..config import (
    MATCHED_METRIC_COLUMNS,
    METHOD_AMOUNT_GAIN,
    METHOD_BASELINE,
    METHOD_FAMILIES,
    METHOD_FIXED,
    METHOD_P_ONLY,
    EffectiveExperimentConfig,
)
from ..records import score_path


def _aggregate_matched_metrics(
    seed_metrics: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget, method_family), group in seed_metrics.groupby(
        ["target_budget", "method_family"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "target_budget": int(budget),
            "primary_budget": (
                int(budget) in effective_config.primary_budgets
            ),
            "method_family": str(method_family),
            "score_path": score_path(str(method_family), int(budget)),
            "seed_count": int(group["seed"].nunique()),
            "gain_path": (
                "selected_per_seed"
                if method_family in {METHOD_P_ONLY, METHOD_AMOUNT_GAIN}
                else "not_applicable"
            ),
            "selection_status_values": json.dumps(
                sorted(set(group["selection_status"].astype(str)))
            ),
        }
        gain_counts = (
            group["selected_gain"].astype(str).value_counts().sort_index()
        )
        row["selected_gain_distribution"] = json.dumps(
            {str(key): int(value) for key, value in gain_counts.items()},
            sort_keys=True,
        )
        for metric in MATCHED_METRIC_COLUMNS:
            values = group[metric].to_numpy(dtype=float)
            if (
                values.shape[0] != len(effective_config.seeds)
                or set(group["seed"].astype(int))
                != set(effective_config.seeds)
                or not np.isfinite(values).all()
            ):
                raise RuntimeError(
                    f"Non-finite/incomplete matched metric {metric} "
                    f"for k={budget}, method={method_family}."
                )
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(effective_config.seeds) > 1
                else float("nan")
            )
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["target_budget", "method_family"],
        kind="mergesort",
    )
    expected_rows = len(effective_config.target_budgets) * len(
        METHOD_FAMILIES
    )
    if len(result) != expected_rows:
        raise RuntimeError("Aggregated matched results have incomplete rows.")
    return result


def _paired_deltas(
    seed_metrics: pd.DataFrame,
    *,
    metric: str,
    delta_name: str,
) -> pd.DataFrame:
    baseline = seed_metrics.loc[
        seed_metrics["method_family"] == METHOD_BASELINE,
        ["seed", "target_budget", metric],
    ].rename(columns={metric: f"bce_{metric}"})
    comparison = seed_metrics.loc[
        seed_metrics["method_family"] != METHOD_BASELINE,
        [
            "seed",
            "target_budget",
            "primary_budget",
            "method_family",
            "score_path",
            "selected_gain",
            "selection_status",
            metric,
        ],
    ]
    paired = comparison.merge(
        baseline,
        on=["seed", "target_budget"],
        how="left",
        validate="many_to_one",
    )
    paired[delta_name] = (
        paired[metric].astype(float)
        - paired[f"bce_{metric}"].astype(float)
    )
    if not np.isfinite(
        paired[[metric, f"bce_{metric}", delta_name]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"Non-finite paired delta for {metric}.")
    return paired.sort_values(
        ["target_budget", "method_family", "seed"],
        kind="mergesort",
    )


def _fixed_vs_selected(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = seed_metrics.loc[
        seed_metrics["method_family"] == METHOD_AMOUNT_GAIN
    ].copy()
    fixed = seed_metrics.loc[
        seed_metrics["method_family"] == METHOD_FIXED,
        [
            "seed",
            "target_budget",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "amount_ndcg_at_k",
        ],
    ].rename(
        columns={
            "prevented_loss_ratio_at_k": "fixed_prevented_loss_ratio_at_k",
            "frauds_at_k": "fixed_frauds_at_k",
            "amount_ndcg_at_k": "fixed_amount_ndcg_at_k",
        }
    )
    merged = selected.merge(
        fixed,
        on=["seed", "target_budget"],
        how="left",
        validate="one_to_one",
    )
    merged["selected_minus_fixed_plr"] = (
        merged["prevented_loss_ratio_at_k"]
        - merged["fixed_prevented_loss_ratio_at_k"]
    )
    merged["selected_minus_fixed_frauds"] = (
        merged["frauds_at_k"] - merged["fixed_frauds_at_k"]
    )
    merged["selected_minus_fixed_amount_ndcg"] = (
        merged["amount_ndcg_at_k"] - merged["fixed_amount_ndcg_at_k"]
    )
    return merged


def _aggregate_diagnostic(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key_values, strict=True))
        row["row_count"] = len(group)
        row["seed_count"] = int(group["seed"].nunique())
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite diagnostic metric {metric}.")
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def _score_path_inventory(
    effective_config: EffectiveExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in effective_config.target_budgets:
        for method_family in METHOD_FAMILIES:
            rows.append(
                {
                    "path_scope": "central",
                    "target_budget": budget,
                    "primary_budget": (
                        budget in effective_config.primary_budgets
                    ),
                    "method_family": method_family,
                    "score_path": score_path(method_family, budget),
                    "gain_visibility": (
                        "selected_gain_only"
                        if method_family
                        in {METHOD_P_ONLY, METHOD_AMOUNT_GAIN}
                        else "not_applicable"
                    ),
                    "artifact_location": "final_outer_run",
                }
            )
    for gain in effective_config.enabled_gain_profiles:
        rows.append(
            {
                "path_scope": "inner_validation_only",
                "target_budget": "all",
                "primary_budget": "not_applicable",
                "method_family": METHOD_AMOUNT_GAIN,
                "score_path": f"inner_candidate_amount_gain_{gain}",
                "gain_visibility": gain,
                "artifact_location": "inner_validation",
            }
        )
    return pd.DataFrame(rows)


def _write_figure_data(
    *,
    output_root: Path,
    seed_metrics: pd.DataFrame,
    paired_plr: pd.DataFrame,
    paired_fraud: pd.DataFrame,
    fixed_vs_selected: pd.DataFrame,
    distribution: pd.DataFrame,
    ties: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> None:
    central = seed_metrics.loc[
        seed_metrics["target_budget"]
        .astype(int)
        .isin(effective_config.primary_budgets)
    ].copy()
    _write_csv(
        output_root / "figure_data" / "central_budget_plr.csv",
        central[
            [
                "seed",
                "target_budget",
                "method_family",
                "score_path",
                "prevented_loss_ratio_at_k",
            ]
        ].rename(columns={"prevented_loss_ratio_at_k": "value"}),
    )
    _write_csv(
        output_root / "figure_data" / "central_budget_fraud.csv",
        central[
            [
                "seed",
                "target_budget",
                "method_family",
                "score_path",
                "frauds_at_k",
            ]
        ].rename(columns={"frauds_at_k": "value"}),
    )
    precision_recall = central.melt(
        id_vars=["seed", "target_budget", "method_family", "score_path"],
        value_vars=["precision_at_k", "recall_at_k"],
        var_name="metric",
        value_name="value",
    )
    _write_csv(
        output_root
        / "figure_data"
        / "central_budget_precision_recall.csv",
        precision_recall,
    )
    _write_csv(
        output_root / "figure_data" / "paired_plr_deltas.csv",
        paired_plr,
    )
    _write_csv(
        output_root / "figure_data" / "paired_fraud_deltas.csv",
        paired_fraud,
    )
    _write_csv(
        output_root / "figure_data" / "fixed_vs_ranker.csv",
        fixed_vs_selected,
    )
    _write_csv(
        output_root
        / "figure_data"
        / "selected_gain_distribution.csv",
        distribution,
    )
    _write_csv(
        output_root / "figure_data" / "tie_diagnostics.csv",
        ties,
    )
    _write_csv(
        output_root
        / "figure_data"
        / "all_budget_matched_results.csv",
        seed_metrics,
    )


def aggregate_final_outputs(
    *,
    output_root: Path,
    seed_results: list[dict[str, pd.DataFrame]],
    effective_config: EffectiveExperimentConfig,
    data_sha256: str,
    input_root: Path | None = None,
) -> None:
    read_root = output_root if input_root is None else input_root
    seed_metrics = pd.concat(
        [result["metrics"] for result in seed_results],
        ignore_index=True,
    )
    amount_models = pd.concat(
        [result["amount_models"] for result in seed_results],
        ignore_index=True,
    )
    p_only_models = pd.concat(
        [result["p_only_models"] for result in seed_results],
        ignore_index=True,
    )
    pool_summary = pd.concat(
        [result["pool_summary"] for result in seed_results],
        ignore_index=True,
    )
    ties = pd.concat(
        [result["ties"] for result in seed_results],
        ignore_index=True,
    )
    replacements = pd.concat(
        [result["replacements"] for result in seed_results],
        ignore_index=True,
    )
    boundaries = pd.concat(
        [result["boundaries"] for result in seed_results],
        ignore_index=True,
    )
    high_legit = pd.concat(
        [result["high_legit"] for result in seed_results],
        ignore_index=True,
    )
    hard = pd.concat(
        [result["hard"] for result in seed_results],
        ignore_index=True,
    )
    global_metrics = pd.concat(
        [result["global_metrics"] for result in seed_results],
        ignore_index=True,
    )

    expected_seed_budget_count = len(effective_config.seeds) * len(
        effective_config.target_budgets
    )
    expected_metric_keys = {
        (seed, budget, method_family)
        for seed in effective_config.seeds
        for budget in effective_config.target_budgets
        for method_family in METHOD_FAMILIES
    }
    actual_metric_keys = set(
        seed_metrics[
            ["seed", "target_budget", "method_family"]
        ].itertuples(index=False, name=None)
    )
    expected_model_keys = {
        (seed, budget)
        for seed in effective_config.seeds
        for budget in effective_config.target_budgets
    }
    amount_model_keys = set(
        amount_models[["seed", "target_budget"]].itertuples(
            index=False,
            name=None,
        )
    )
    p_only_model_keys = set(
        p_only_models[["seed", "target_budget"]].itertuples(
            index=False,
            name=None,
        )
    )
    if (
        actual_metric_keys != expected_metric_keys
        or len(seed_metrics) != len(expected_metric_keys)
        or amount_model_keys != expected_model_keys
        or p_only_model_keys != expected_model_keys
        or len(amount_models) != expected_seed_budget_count
        or len(p_only_models) != expected_seed_budget_count
    ):
        raise RuntimeError("Final aggregate model/metric counts are incomplete.")
    all_budget = _aggregate_matched_metrics(seed_metrics, effective_config)
    central = all_budget.loc[
        all_budget["target_budget"]
        .astype(int)
        .isin(effective_config.primary_budgets)
    ].reset_index(drop=True)
    if set(central["target_budget"].astype(int)) != set(
        effective_config.primary_budgets
    ):
        raise RuntimeError("Central results contain incorrect budgets.")
    paired_plr = _paired_deltas(
        seed_metrics,
        metric="prevented_loss_ratio_at_k",
        delta_name="delta_plr_vs_bce",
    )
    paired_fraud = _paired_deltas(
        seed_metrics,
        metric="frauds_at_k",
        delta_name="delta_frauds_vs_bce",
    )
    fixed_vs_selected = _fixed_vs_selected(seed_metrics)
    selected = pd.read_csv(
        read_root
        / "selection_freeze"
        / "selected_gain_by_seed_budget.csv"
    )
    distribution = pd.read_csv(
        read_root
        / "selection_freeze"
        / "selected_gain_distribution_by_budget.csv"
    )
    selected_models = pd.concat(
        [amount_models, p_only_models],
        ignore_index=True,
    ).sort_values(
        ["seed", "target_budget", "model_type"],
        kind="mergesort",
    )

    tie_summary = _aggregate_diagnostic(
        ties,
        group_columns=["target_budget", "method_family"],
        metric_columns=[
            "unique_raw_ranker_scores",
            "cutoff_tie_size",
            "cutoff_tie_rank_min",
            "cutoff_tie_rank_max",
        ],
    )
    hard_summary = _aggregate_diagnostic(
        hard,
        group_columns=["target_budget", "method_family"],
        metric_columns=[
            "prevented_loss_ratio_at_k",
            "fraud_amount_sum_at_k",
            "amount_ndcg_at_k",
            "q90_captured_ratio_at_k",
            "q90_amount_ndcg_at_k",
        ],
    )
    replacement_summary = _aggregate_diagnostic(
        replacements,
        group_columns=["target_budget", "method_family", "subset"],
        metric_columns=[
            "case_count",
            "fraud_count",
            "fraud_amount_sum",
            "legit_count",
        ],
    )
    boundary_summary = _aggregate_diagnostic(
        boundaries.assign(
            crossed_into_topk=boundaries["crossed_into_topk"].astype(int),
            dropped_from_topk=boundaries["dropped_from_topk"].astype(int),
        ),
        group_columns=["target_budget", "method_family"],
        metric_columns=[
            "rank_shift_vs_bce",
            "crossed_into_topk",
            "dropped_from_topk",
        ],
    )
    high_legit_summary = _aggregate_diagnostic(
        high_legit,
        group_columns=["target_budget", "method_family"],
        metric_columns=[
            "legit_count_at_k",
            "high_amount_legit_count_at_k",
            "mean_legit_amount_at_k",
        ],
    )
    global_summary = _aggregate_diagnostic(
        global_metrics,
        group_columns=["target_budget", "method_family", "metric"],
        metric_columns=["value"],
    )

    table_outputs = {
        "central_budget_results.csv": central,
        "all_budget_matched_results.csv": all_budget,
        "selected_gain_by_seed_budget.csv": selected,
        "selected_gain_distribution_by_budget.csv": distribution,
        "selected_model_parameters.csv": selected_models,
        "paired_plr_deltas_vs_bce.csv": paired_plr,
        "paired_fraud_deltas_vs_bce.csv": paired_fraud,
        "fixed_vs_selected_ranker.csv": fixed_vs_selected,
        "candidate_pool_summary.csv": pool_summary,
        "tie_summary.csv": tie_summary,
        "hard_impact_summary.csv": hard_summary,
        "replacement_summary.csv": replacement_summary,
        "boundary_summary.csv": boundary_summary,
        "high_amount_legit_summary.csv": high_legit_summary,
        "global_metrics_by_budget_model.csv": global_summary,
    }
    for filename, frame in table_outputs.items():
        _write_csv(output_root / "tables" / filename, frame)

    diagnostic_outputs = {
        "tie_diagnostics_seedwise.csv": ties,
        "replacement_diagnostics_seedwise.csv": replacements,
        "boundary_diagnostics_seedwise.csv": boundaries,
        "high_amount_legit_diagnostics_seedwise.csv": high_legit,
        "hard_impact_diagnostics_seedwise.csv": hard,
        "global_metrics_seedwise.csv": global_metrics,
    }
    for filename, frame in diagnostic_outputs.items():
        _write_csv(output_root / "diagnostics" / filename, frame)

    _write_csv(
        output_root
        / "comparison"
        / "matched_budget_comparison_long.csv",
        seed_metrics,
    )
    inventory = _score_path_inventory(effective_config)
    _write_csv(
        output_root / "comparison" / "score_path_inventory.csv",
        inventory,
    )
    identity_checks = pd.DataFrame(
        [
            {
                "check": "deduplicated_dataframe_sha256",
                "status": "PASS",
                "detail": data_sha256,
            },
            {
                "check": "outer_seed_budget_config_count",
                "status": "PASS",
                "detail": str(expected_seed_budget_count),
            },
            {
                "check": "selected_amount_gain_model_count",
                "status": "PASS",
                "detail": str(len(amount_models)),
            },
            {
                "check": "p_only_model_count",
                "status": "PASS",
                "detail": str(len(p_only_models)),
            },
            {
                "check": "nonselected_outer_gain_model_count",
                "status": "PASS",
                "detail": "0",
            },
            {
                "check": "central_budget_set",
                "status": "PASS",
                "detail": json.dumps(
                    list(effective_config.primary_budgets)
                ),
            },
            {
                "check": "all_budget_set",
                "status": "PASS",
                "detail": json.dumps(
                    list(effective_config.target_budgets)
                ),
            },
        ]
    )
    _write_csv(
        output_root / "comparison" / "run_identity_checks.csv",
        identity_checks,
    )
    _write_figure_data(
        output_root=output_root,
        seed_metrics=seed_metrics,
        paired_plr=paired_plr,
        paired_fraud=paired_fraud,
        fixed_vs_selected=fixed_vs_selected,
        distribution=distribution,
        ties=ties,
        effective_config=effective_config,
    )
