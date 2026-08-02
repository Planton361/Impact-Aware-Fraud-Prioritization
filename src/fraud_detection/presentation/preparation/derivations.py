"""Pure, deterministic Chapter-5 presentation-data derivations.

The functions in this module consume validated frozen metric rows and frozen
full-ranking dumps.  They aggregate, align, and summarize existing evidence;
they do not fit models, create scores, select parameters, or alter rankings.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .. import (
    GERMAN_PATH_LABELS,
    METHOD_AMOUNT_GAIN,
    METHOD_BCE,
    METHOD_FIXED,
    METHOD_ORDER,
    METHOD_P_ONLY,
    PATH_IDS,
)
from ..catalog import engineering_evidence_statement

CENTRAL_BUDGETS = (20, 50, 100)
DEPTH_BUDGETS = (20, 50)
DEPTH_MAX_RANK = 100
CURVE_GRID = np.linspace(0.0, 1.0, 1001)
TRAINED_METHODS = (METHOD_P_ONLY, METHOD_AMOUNT_GAIN)
COMPARISON_METHODS = (METHOD_P_ONLY, METHOD_AMOUNT_GAIN, METHOD_FIXED)
CANONICAL_OUTPUT_PATHS = (
    "figures/ch5_tradeoff_seedwise.csv",
    "figures/ch5_tradeoff_summary.csv",
    "figures/ch5_budget_policy_seedwise.csv",
    "figures/ch5_budget_policy_summary.csv",
    "figures/ch5_depth_seedwise.csv",
    "figures/ch5_depth_summary.csv",
    "figures/ch5_global_pool_curves_seedwise.csv",
    "figures/ch5_global_pool_curves_summary.csv",
    "figures/ch5_global_metrics_seedwise.csv",
    "figures/ch5_global_metrics_summary.csv",
    "figures/ch5_hard_impact_seedwise.csv",
    "figures/ch5_hard_impact_summary.csv",
    "figures/ch5_replacement_events.csv",
    "figures/ch5_seedwise_k50_diagnostic.csv",
    "figures/app_seed_budget_delta_heatmap.csv",
    "figures/app_exact_tie_intervals.csv",
    "figures/app_candidate_pool_ceiling.csv",
    "tables/ch5_t1_central_topk_results.csv",
    "tables/ch5_t2_seedwise_k50_diagnostic.csv",
    "tables/ch5_t3_replacement_seedwise.csv",
    "tables/ch5_t3_replacement_summary.csv",
    "tables/ch5_t3_boundary_pooled.csv",
    "tables/ch5_t4_high_amount_legit_seedwise.csv",
    "tables/ch5_t4_high_amount_legit_summary.csv",
    "tables/app_t1_hard_impact_exact_values.csv",
    "tables/app_t2_global_metrics_by_budget.csv",
    "tables/app_t3_exact_tie_bounds.csv",
    "tables/app_t4_candidate_pool_coverage.csv",
    "tables/app_t5_seedwise_central_results.csv",
)
ENGINEERING_OUTPUT_PATHS = (
    "engineering/figures/engineering_seed_budget_delta_heatmap.csv",
    "engineering/tables/engineering_central_topk_summary.csv",
)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing required columns: {missing}")


def _decorate_paths(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["path_id"] = result["method_family"].map(PATH_IDS)
    result["path_label"] = result["method_family"].map(GERMAN_PATH_LABELS)
    if result[["path_id", "path_label"]].isna().any().any():
        raise RuntimeError("Unknown method family in presentation derivation.")
    return result


def _ordered(
    frame: pd.DataFrame,
    keys: list[str],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    work = frame.copy()
    seed_order = {seed: index for index, seed in enumerate(seeds)}
    budget_order = {budget: index for index, budget in enumerate(budgets)}
    method_order = {method: index for index, method in enumerate(METHOD_ORDER)}
    helpers: list[str] = []
    translated: list[str] = []
    for key in keys:
        if key == "seed":
            work["__seed_order"] = work[key].astype(int).map(seed_order)
            translated.append("__seed_order")
            helpers.append("__seed_order")
        elif key == "target_budget":
            work["__budget_order"] = work[key].astype(int).map(budget_order)
            translated.append("__budget_order")
            helpers.append("__budget_order")
        elif key == "method_family":
            work["__method_order"] = work[key].map(method_order)
            translated.append("__method_order")
            helpers.append("__method_order")
        else:
            translated.append(key)
    if any(work[column].isna().any() for column in helpers):
        raise RuntimeError("Unexpected seed, budget, or path in derivation output.")
    return (
        work.sort_values(translated, kind="mergesort")
        .drop(columns=helpers)
        .reset_index(drop=True)
    )


def _summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    result = grouped.size().rename("row_count").reset_index()
    for metric in metrics:
        aggregate = (
            grouped[metric]
            .agg(["count", "mean", "std"])
            .rename(
                columns={
                    "count": f"{metric}_n",
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_sd",
                }
            )
            .reset_index()
        )
        result = result.merge(
            aggregate,
            on=group_columns,
            how="left",
            validate="one_to_one",
        )
    return result


def _validate_frozen_actuals(
    actual_fraud: object,
    actual_plr: object,
    actual_q90: object,
    frozen: pd.Series,
    *,
    context: str,
) -> None:
    """Require ranking-derived central actuals to reproduce frozen metrics."""

    if int(actual_fraud) != int(frozen["frauds_at_k"]):
        raise RuntimeError(f"Frozen Fraud@k mismatch for {context}.")
    if not math.isclose(
        float(actual_plr),
        float(frozen["prevented_loss_ratio_at_k"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"Frozen PLR@k mismatch for {context}.")
    if int(actual_q90) != int(frozen["q90_captured_fraud_count"]):
        raise RuntimeError(f"Frozen q90 capture mismatch for {context}.")


def derive_tradeoff(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive seed-paired central-budget PLR/Fraud trade-offs."""

    central = seed_metrics.loc[
        seed_metrics["target_budget"].astype(int).isin(CENTRAL_BUDGETS)
    ].copy()
    baseline = central.loc[
        central["method_family"] == METHOD_BCE,
        ["seed", "target_budget", "prevented_loss_ratio_at_k", "frauds_at_k"],
    ].rename(
        columns={
            "prevented_loss_ratio_at_k": "bce_plr_at_k",
            "frauds_at_k": "bce_fraud_at_k",
        }
    )
    paired = central.loc[
        central["method_family"].isin(COMPARISON_METHODS)
    ].merge(
        baseline,
        on=["seed", "target_budget"],
        validate="many_to_one",
    )
    paired = paired.rename(
        columns={
            "prevented_loss_ratio_at_k": "plr_at_k",
            "frauds_at_k": "fraud_at_k",
        }
    )
    paired["delta_plr_vs_bce"] = paired["plr_at_k"] - paired["bce_plr_at_k"]
    paired["delta_fraud_at_k_vs_bce"] = (
        paired["fraud_at_k"] - paired["bce_fraud_at_k"]
    )
    paired = _decorate_paths(paired)
    columns = [
        "seed",
        "target_budget",
        "method_family",
        "path_id",
        "path_label",
        "score_path",
        "plr_at_k",
        "fraud_at_k",
        "bce_plr_at_k",
        "bce_fraud_at_k",
        "delta_plr_vs_bce",
        "delta_fraud_at_k_vs_bce",
    ]
    paired = _ordered(
        paired[columns],
        ["target_budget", "method_family", "seed"],
        seeds,
        budgets,
    )
    expected = len(seeds) * len(CENTRAL_BUDGETS) * len(COMPARISON_METHODS)
    if len(paired) != expected or paired.duplicated(
        ["seed", "target_budget", "method_family"]
    ).any():
        raise RuntimeError("Central paired trade-off grid is incomplete.")
    summary = _summary(
        paired,
        ["target_budget", "method_family", "path_id", "path_label"],
        [
            "plr_at_k",
            "fraud_at_k",
            "bce_plr_at_k",
            "bce_fraud_at_k",
            "delta_plr_vs_bce",
            "delta_fraud_at_k_vs_bce",
        ],
    )
    return paired, summary


def derive_budget_profile(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the seven-point discrete budget-policy profile."""

    baseline = seed_metrics.loc[
        seed_metrics["method_family"] == METHOD_BCE,
        ["seed", "target_budget", "frauds_at_k"],
    ].rename(columns={"frauds_at_k": "bce_fraud_at_k"})
    profile = seed_metrics.merge(
        baseline,
        on=["seed", "target_budget"],
        validate="many_to_one",
    )
    profile["delta_fraud_at_k_vs_bce"] = (
        profile["frauds_at_k"] - profile["bce_fraud_at_k"]
    )
    profile["central_budget"] = profile["target_budget"].astype(int).isin(
        CENTRAL_BUDGETS
    )
    profile["separate_budget_conditioned_model"] = profile[
        "method_family"
    ].isin(TRAINED_METHODS)
    budget_positions = {budget: index for index, budget in enumerate(budgets)}
    profile["budget_position"] = (
        profile["target_budget"].astype(int).map(budget_positions).astype(int)
    )
    profile = _decorate_paths(profile)
    keep = [
        "seed",
        "target_budget",
        "budget_position",
        "central_budget",
        "method_family",
        "path_id",
        "path_label",
        "score_path",
        "separate_budget_conditioned_model",
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "bce_fraud_at_k",
        "delta_fraud_at_k_vs_bce",
    ]
    profile = _ordered(
        profile[keep],
        ["target_budget", "method_family", "seed"],
        seeds,
        budgets,
    )
    expected = len(seeds) * len(budgets) * len(METHOD_ORDER)
    if len(profile) != expected or profile.duplicated(
        ["seed", "target_budget", "method_family"]
    ).any():
        raise RuntimeError("Budget-policy grid is incomplete.")
    summary = _summary(
        profile,
        [
            "target_budget",
            "budget_position",
            "central_budget",
            "method_family",
            "path_id",
            "path_label",
            "separate_budget_conditioned_model",
        ],
        [
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "delta_fraud_at_k_vs_bce",
        ],
    )
    return profile, summary


def derive_depth_profiles(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build within-model cumulative profiles without cross-budget subtraction."""

    metric_lookup = seed_metrics.set_index(
        ["seed", "target_budget", "method_family"]
    )
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        for budget in DEPTH_BUDGETS:
            if budget not in budgets:
                continue
            for method in METHOD_ORDER:
                ranking = groups[(seed, budget, method)].sort_values(
                    "final_rank_position", kind="mergesort"
                )
                if len(ranking) < DEPTH_MAX_RANK:
                    raise RuntimeError(
                        f"Ranking shorter than depth profile: {(seed, budget, method)}"
                    )
                labels = ranking["Class"].to_numpy(int)
                amounts = ranking["Amount"].to_numpy(float)
                total_fraud_amount = float(amounts[labels == 1].sum())
                if total_fraud_amount <= 0:
                    raise RuntimeError("Fraud Amount denominator must be positive.")
                top_labels = labels[:DEPTH_MAX_RANK]
                top_fraud_amount = (labels * amounts)[:DEPTH_MAX_RANK]
                frame = pd.DataFrame(
                    {
                        "seed": seed,
                        "target_budget": budget,
                        "method_family": method,
                        "rank_depth": np.arange(1, DEPTH_MAX_RANK + 1),
                        "cumulative_fraud_count": np.cumsum(top_labels),
                        "cumulative_fraud_amount": np.cumsum(top_fraud_amount),
                        "cumulative_plr": np.cumsum(top_fraud_amount)
                        / total_fraud_amount,
                        "fixed_model_depth_profile": True,
                        "cross_budget_subtraction": False,
                    }
                )
                for metric in (
                    "cumulative_fraud_count",
                    "cumulative_fraud_amount",
                    "cumulative_plr",
                ):
                    values = frame[metric].to_numpy(float)
                    if np.any(np.diff(values) < -1e-12):
                        raise RuntimeError(
                            f"Non-monotone depth profile: "
                            f"{(seed, budget, method, metric)}"
                        )
                endpoint = frame.iloc[budget - 1]
                frozen = metric_lookup.loc[(seed, budget, method)]
                if int(endpoint["cumulative_fraud_count"]) != int(
                    frozen["frauds_at_k"]
                ):
                    raise RuntimeError(
                        f"Depth Fraud endpoint mismatch: {(seed, budget, method)}"
                    )
                if not math.isclose(
                    float(endpoint["cumulative_fraud_amount"]),
                    float(frozen["fraud_amount_sum_at_k"]),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(
                        "Depth Fraud Amount endpoint mismatch: "
                        f"{(seed, budget, method)}"
                    )
                if not math.isclose(
                    float(endpoint["cumulative_plr"]),
                    float(frozen["prevented_loss_ratio_at_k"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Depth PLR endpoint mismatch: {(seed, budget, method)}"
                    )
                rows.append(frame)
    seedwise = _decorate_paths(pd.concat(rows, ignore_index=True))
    seedwise = _ordered(
        seedwise,
        ["target_budget", "method_family", "seed", "rank_depth"],
        seeds,
        budgets,
    )
    summary = _summary(
        seedwise,
        [
            "target_budget",
            "method_family",
            "path_id",
            "path_label",
            "rank_depth",
            "fixed_model_depth_profile",
            "cross_budget_subtraction",
        ],
        [
            "cumulative_fraud_count",
            "cumulative_fraud_amount",
            "cumulative_plr",
        ],
    )
    return seedwise, summary


def _interpolated_curves(
    labels: np.ndarray,
    order_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    fpr, tpr, _ = roc_curve(labels, order_score)
    roc_values = np.interp(CURVE_GRID, fpr, tpr)
    roc_values[0] = 0.0
    roc_values[-1] = 1.0

    precision, recall, _ = precision_recall_curve(labels, order_score)
    curve = pd.DataFrame(
        {"recall": recall[::-1], "precision": precision[::-1]}
    )
    curve = (
        curve.groupby("recall", sort=True, as_index=False)["precision"]
        .max()
        .sort_values("recall", kind="mergesort")
    )
    pr_values = np.interp(
        CURVE_GRID,
        curve["recall"].to_numpy(float),
        curve["precision"].to_numpy(float),
    )
    return (
        roc_values,
        pr_values,
        float(roc_auc_score(labels, order_score)),
        float(average_precision_score(labels, order_score)),
    )


def derive_global_and_pool_curves(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    diagnostics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive k=50 full-order/pool curves and central global metrics."""

    _require_columns(
        diagnostics,
        ["seed", "target_budget", "method_family", "metric", "value"],
        "global metric diagnostics",
    )
    lookup = diagnostics.loc[
        diagnostics["metric"].ne("brier_score_probability")
    ].set_index(
        ["seed", "target_budget", "method_family", "metric"]
    )["value"]
    metric_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for seed in seeds:
        for budget in CENTRAL_BUDGETS:
            if budget not in budgets:
                continue
            for method in METHOD_ORDER:
                ranking = groups[(seed, budget, method)]
                labels = ranking["Class"].to_numpy(int)
                order_score = -ranking["final_rank_position"].to_numpy(float)
                roc_values, pr_values, auc, ap = _interpolated_curves(
                    labels, order_score
                )
                for metric, observed in (
                    ("roc_auc", auc),
                    ("average_precision", ap),
                ):
                    source_metric = {
                        "roc_auc": "roc_auc_of_final_order",
                        "average_precision": "average_precision_of_final_order",
                    }[metric]
                    try:
                        expected = float(
                            lookup.loc[(seed, budget, method, source_metric)]
                        )
                    except KeyError as exc:
                        raise RuntimeError(
                            "Missing registered global metric diagnostic for "
                            f"{(seed, budget, method, source_metric)}"
                        ) from exc
                    if not math.isclose(
                        observed, expected, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        raise RuntimeError(
                            "Global metric mismatch for "
                            f"{(seed, budget, method, source_metric)}"
                        )
                    metric_rows.append(
                        {
                            "seed": seed,
                            "target_budget": budget,
                            "scope": "full_order",
                            "scope_label": "vollständige Testordnung",
                            "method_family": method,
                            "metric": metric,
                            "value": observed,
                            "score_interpretation": (
                                "BCE-Wahrscheinlichkeit"
                                if method == METHOD_BCE
                                else "ordinaler Full-Order-Score"
                            ),
                        }
                    )
                if method == METHOD_BCE:
                    try:
                        brier = float(
                            brier_score_loss(
                                labels, ranking["p_fraud"].to_numpy(float)
                            )
                        )
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "Computed BCE Brier is non-finite for "
                            f"seed={seed}, k={budget}"
                        ) from exc
                    brier_rows = diagnostics.loc[
                        diagnostics["seed"].eq(seed)
                        & diagnostics["target_budget"].eq(budget)
                        & diagnostics["method_family"].eq(METHOD_BCE)
                        & diagnostics["metric"].eq(
                            "brier_score_probability"
                        )
                    ]
                    if brier_rows.empty:
                        raise RuntimeError(
                            "Missing registered BCE Brier diagnostic for "
                            f"seed={seed}, k={budget}"
                        )
                    if len(brier_rows) != 1:
                        raise RuntimeError(
                            "Duplicate registered BCE Brier diagnostic for "
                            f"seed={seed}, k={budget}"
                        )
                    try:
                        registered_brier = float(brier_rows.iloc[0]["value"])
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "Registered BCE Brier diagnostic is non-finite "
                            f"for seed={seed}, k={budget}"
                        ) from exc
                    if not math.isfinite(brier):
                        raise RuntimeError(
                            "Computed BCE Brier is non-finite for "
                            f"seed={seed}, k={budget}"
                        )
                    if not math.isfinite(registered_brier):
                        raise RuntimeError(
                            "Registered BCE Brier diagnostic is non-finite "
                            f"for seed={seed}, k={budget}"
                        )
                    if not math.isclose(
                        brier,
                        registered_brier,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"BCE Brier mismatch for seed={seed}, k={budget}"
                        )
                    metric_rows.append(
                        {
                            "seed": seed,
                            "target_budget": budget,
                            "scope": "full_order",
                            "scope_label": "vollständige Testordnung",
                            "method_family": method,
                            "metric": "brier",
                            "value": brier,
                            "score_interpretation": "BCE-Wahrscheinlichkeit",
                        }
                    )
                if budget == 50:
                    for curve_type, x_name, y_name, values in (
                        ("roc", "false_positive_rate", "true_positive_rate", roc_values),
                        ("pr", "recall", "precision", pr_values),
                    ):
                        curve_rows.extend(
                            {
                                "seed": seed,
                                "target_budget": 50,
                                "scope": "full_order",
                                "scope_label": "vollständige Testordnung",
                                "curve_type": curve_type,
                                "method_family": method,
                                "grid_index": index,
                                "x_name": x_name,
                                "y_name": y_name,
                                "x": float(grid_value),
                                "y": float(values[index]),
                            }
                            for index, grid_value in enumerate(CURVE_GRID)
                        )

                    candidate = ranking.loc[
                        ranking["candidate_flag"].astype(bool)
                    ].sort_values("final_rank_position", kind="mergesort")
                    candidate_labels = candidate["Class"].to_numpy(int)
                    if np.unique(candidate_labels).size != 2:
                        raise RuntimeError(
                            f"Candidate pool has only one class: {(seed, method)}"
                        )
                    candidate_score = -candidate[
                        "final_rank_position"
                    ].to_numpy(float)
                    pool_roc, pool_pr, pool_auc, pool_ap = _interpolated_curves(
                        candidate_labels, candidate_score
                    )
                    metric_rows.extend(
                        [
                            {
                                "seed": seed,
                                "target_budget": 50,
                                "scope": "candidate_pool",
                                "scope_label": "BCE-Top-1000-Kandidatenpool",
                                "method_family": method,
                                "metric": "roc_auc",
                                "value": pool_auc,
                                "score_interpretation": (
                                    "BCE-Reihenfolge im Kandidatenpool"
                                    if method == METHOD_BCE
                                    else "ordinaler Kandidatenpool-Score"
                                ),
                            },
                            {
                                "seed": seed,
                                "target_budget": 50,
                                "scope": "candidate_pool",
                                "scope_label": "BCE-Top-1000-Kandidatenpool",
                                "method_family": method,
                                "metric": "average_precision",
                                "value": pool_ap,
                                "score_interpretation": (
                                    "BCE-Reihenfolge im Kandidatenpool"
                                    if method == METHOD_BCE
                                    else "ordinaler Kandidatenpool-Score"
                                ),
                            },
                        ]
                    )
                    for curve_type, x_name, y_name, values in (
                        (
                            "roc",
                            "false_positive_rate",
                            "true_positive_rate",
                            pool_roc,
                        ),
                        ("pr", "recall", "precision", pool_pr),
                    ):
                        curve_rows.extend(
                            {
                                "seed": seed,
                                "target_budget": 50,
                                "scope": "candidate_pool",
                                "scope_label": "BCE-Top-1000-Kandidatenpool",
                                "curve_type": curve_type,
                                "method_family": method,
                                "grid_index": index,
                                "x_name": x_name,
                                "y_name": y_name,
                                "x": float(grid_value),
                                "y": float(values[index]),
                            }
                            for index, grid_value in enumerate(CURVE_GRID)
                        )

    curve_seedwise = _decorate_paths(pd.DataFrame(curve_rows))
    curve_seedwise = _ordered(
        curve_seedwise,
        ["scope", "curve_type", "method_family", "seed", "grid_index"],
        seeds,
        budgets,
    )
    curve_summary = _summary(
        curve_seedwise,
        [
            "target_budget",
            "scope",
            "scope_label",
            "curve_type",
            "method_family",
            "path_id",
            "path_label",
            "grid_index",
            "x_name",
            "y_name",
            "x",
        ],
        ["y"],
    )
    metric_seedwise = _decorate_paths(pd.DataFrame(metric_rows))
    metric_seedwise = _ordered(
        metric_seedwise,
        ["scope", "target_budget", "method_family", "metric", "seed"],
        seeds,
        budgets,
    )
    if not metric_seedwise.loc[
        metric_seedwise["metric"] == "brier", "method_family"
    ].eq(METHOD_BCE).all():
        raise RuntimeError("Brier must be restricted to BCE.")
    metric_summary = _summary(
        metric_seedwise,
        [
            "target_budget",
            "scope",
            "scope_label",
            "method_family",
            "path_id",
            "path_label",
            "metric",
            "score_interpretation",
        ],
        ["value"],
    )
    return curve_seedwise, curve_summary, metric_seedwise, metric_summary


def derive_hard_impact(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive q90 capture, BCE-miss recovery, and Amount-nDCG at k=50."""

    if 50 not in budgets:
        return pd.DataFrame(), pd.DataFrame()
    metric_lookup = seed_metrics.set_index(
        ["seed", "target_budget", "method_family"]
    )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        baseline = groups[(seed, 50, METHOD_BCE)].set_index(
            "row_index", drop=False
        )
        fraud = baseline.loc[baseline["Class"].astype(int) == 1]
        q90_threshold = float(
            np.quantile(fraud["Amount"].to_numpy(float), 0.90)
        )
        q90_ids = set(
            fraud.loc[fraud["Amount"].astype(float) >= q90_threshold]
            .index.astype(int)
        )
        bce_top = set(
            baseline.loc[
                baseline["final_rank_position"].astype(int) <= 50,
                "row_index",
            ].astype(int)
        )
        missed = q90_ids - bce_top
        for method in METHOD_ORDER:
            ranking = groups[(seed, 50, method)]
            top_ids = set(
                ranking.loc[
                    ranking["final_rank_position"].astype(int) <= 50,
                    "row_index",
                ].astype(int)
            )
            captured = len(top_ids & q90_ids) / len(q90_ids)
            frozen = metric_lookup.loc[(seed, 50, method)]
            if not math.isclose(
                captured,
                float(frozen["q90_captured_ratio_at_k"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"q90 capture mismatch for {(seed, method)}"
                )
            recovery = (
                np.nan
                if method == METHOD_BCE
                else (len(top_ids & missed) / len(missed) if missed else np.nan)
            )
            rows.append(
                {
                    "seed": seed,
                    "target_budget": 50,
                    "q": 0.90,
                    "q90_threshold": q90_threshold,
                    "method_family": method,
                    "q90_captured_ratio": captured,
                    "baseline_miss_recovery": recovery,
                    "amount_ndcg_at_50": float(frozen["amount_ndcg_at_k"]),
                }
            )
    seedwise = _decorate_paths(pd.DataFrame(rows))
    seedwise = _ordered(
        seedwise,
        ["method_family", "seed"],
        seeds,
        budgets,
    )
    bce_recovery = seedwise.loc[
        seedwise["method_family"] == METHOD_BCE, "baseline_miss_recovery"
    ]
    if bce_recovery.notna().any():
        raise RuntimeError("BCE miss recovery must remain missing.")
    summary = _summary(
        seedwise,
        ["target_budget", "q", "method_family", "path_id", "path_label"],
        [
            "q90_captured_ratio",
            "baseline_miss_recovery",
            "amount_ndcg_at_50",
        ],
    )
    return seedwise, summary


def _tie_interval(
    ranking: pd.DataFrame,
    budget: int,
    q90_threshold: float,
) -> dict[str, object]:
    ordered = ranking.sort_values("final_rank_position", kind="mergesort")
    cutoff = ordered.iloc[budget - 1]
    cutoff_score = float(cutoff["raw_ranker_score"])
    tie = ordered.loc[
        ordered["raw_ranker_score"].to_numpy(float) == cutoff_score
    ].copy()
    if tie.empty:
        raise RuntimeError("Cutoff tie block is empty.")
    first_rank = int(tie["final_rank_position"].min())
    last_rank = int(tie["final_rank_position"].max())
    if not first_rank <= budget <= last_rank:
        raise RuntimeError("Cutoff is not contained in exact tie block.")
    if len(tie) != last_rank - first_rank + 1:
        raise RuntimeError("Exact cutoff tie block is not rank-contiguous.")
    above = ordered.loc[
        ordered["final_rank_position"].astype(int) < first_rank
    ]
    available = budget - len(above)
    frauds_above = int(above["Class"].astype(int).sum())
    fraud_amount_above = float(
        above.loc[above["Class"].astype(int) == 1, "Amount"].sum()
    )
    tie_fraud_amounts = sorted(
        tie.loc[tie["Class"].astype(int) == 1, "Amount"].astype(float)
    )
    choose_fraud_min = max(0, available - int((tie["Class"] == 0).sum()))
    choose_fraud_max = min(available, len(tie_fraud_amounts))
    fraud_min = frauds_above + choose_fraud_min
    fraud_max = frauds_above + choose_fraud_max
    plr_min_num = fraud_amount_above + sum(
        tie_fraud_amounts[:choose_fraud_min]
    )
    plr_max_num = fraud_amount_above + sum(
        tie_fraud_amounts[-choose_fraud_max:] if choose_fraud_max else []
    )
    labels = ordered["Class"].to_numpy(int)
    amounts = ordered["Amount"].to_numpy(float)
    total_fraud_amount = float(amounts[labels == 1].sum())
    actual_top = ordered.iloc[:budget]
    fraud_actual = int(actual_top["Class"].astype(int).sum())
    plr_actual = float(
        actual_top.loc[actual_top["Class"].astype(int) == 1, "Amount"].sum()
        / total_fraud_amount
    )
    q90_above = int(
        (
            (above["Class"].astype(int) == 1)
            & (above["Amount"].astype(float) >= q90_threshold)
        ).sum()
    )
    q90_in_tie = int(
        (
            (tie["Class"].astype(int) == 1)
            & (tie["Amount"].astype(float) >= q90_threshold)
        ).sum()
    )
    choose_q90_min = max(0, available - (len(tie) - q90_in_tie))
    choose_q90_max = min(available, q90_in_tie)
    q90_min = q90_above + choose_q90_min
    q90_max = q90_above + choose_q90_max
    q90_actual = int(
        (
            (actual_top["Class"].astype(int) == 1)
            & (actual_top["Amount"].astype(float) >= q90_threshold)
        ).sum()
    )
    if not (
        fraud_min <= fraud_actual <= fraud_max
        and plr_min_num / total_fraud_amount - 1e-12
        <= plr_actual
        <= plr_max_num / total_fraud_amount + 1e-12
        and q90_min <= q90_actual <= q90_max
    ):
        raise RuntimeError("Actual metric is outside exact tie interval.")
    classification = (
        "NO_CUTOFF_TIE_EFFECT"
        if len(tie) == 1
        else (
            "TIE_ROBUST_BOTH"
            if fraud_min == fraud_max
            and math.isclose(plr_min_num, plr_max_num, abs_tol=1e-15)
            and q90_min == q90_max
            else "TIE_SENSITIVE"
        )
    )
    return {
        "cutoff_raw_score": cutoff_score,
        "exact_tie_block_size": len(tie),
        "first_tie_rank": first_rank,
        "last_tie_rank": last_rank,
        "fraud_at_k_min": fraud_min,
        "fraud_at_k_actual": fraud_actual,
        "fraud_at_k_max": fraud_max,
        "plr_at_k_min": plr_min_num / total_fraud_amount,
        "plr_at_k_actual": plr_actual,
        "plr_at_k_max": plr_max_num / total_fraud_amount,
        "q90_at_k_min": q90_min,
        "q90_at_k_actual": q90_actual,
        "q90_at_k_max": q90_max,
        "technical_tie_classification": classification,
    }


def derive_tie_intervals(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    """Derive exact technical permutation bounds for trained paths."""

    metric_lookup = seed_metrics.set_index(
        ["seed", "target_budget", "method_family"]
    )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for budget in CENTRAL_BUDGETS:
            if budget not in budgets:
                continue
            baseline = groups[(seed, budget, METHOD_BCE)]
            fraud = baseline.loc[baseline["Class"].astype(int) == 1]
            q90_threshold = float(
                np.quantile(fraud["Amount"].to_numpy(float), 0.90)
            )
            for method in TRAINED_METHODS:
                interval = _tie_interval(
                    groups[(seed, budget, method)],
                    budget,
                    q90_threshold,
                )
                _validate_frozen_actuals(
                    interval["fraud_at_k_actual"],
                    interval["plr_at_k_actual"],
                    interval["q90_at_k_actual"],
                    metric_lookup.loc[(seed, budget, method)],
                    context=f"tie interval {(seed, budget, method)}",
                )
                rows.append(
                    {
                        "seed": seed,
                        "target_budget": budget,
                        "method_family": method,
                        **interval,
                    }
                )
    result = _decorate_paths(pd.DataFrame(rows))
    result["interval_interpretation"] = (
        "technische Permutationsgrenze; kein Konfidenzintervall"
    )
    return _ordered(
        result,
        ["target_budget", "method_family", "seed"],
        seeds,
        budgets,
    )


def derive_replacement_events(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align BCE/Amount-Gain by row_index and derive replacement summaries."""

    if 50 not in budgets:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    event_frames: list[pd.DataFrame] = []
    seedwise_rows: list[dict[str, object]] = []
    direction_order = {
        "added_by_amount_gain": 0,
        "removed_from_bce": 1,
    }
    for seed in seeds:
        baseline = groups[(seed, 50, METHOD_BCE)][
            [
                "row_index",
                "Class",
                "Amount",
                "p_fraud",
                "final_rank_position",
            ]
        ].rename(columns={"final_rank_position": "bce_rank"})
        comparator = groups[(seed, 50, METHOD_AMOUNT_GAIN)][
            [
                "row_index",
                "Class",
                "Amount",
                "p_fraud",
                "final_rank_position",
            ]
        ].rename(columns={"final_rank_position": "amount_gain_rank"})
        baseline_ids = set(baseline["row_index"].astype(int))
        comparator_ids = set(comparator["row_index"].astype(int))
        if (
            len(baseline) != len(comparator)
            or baseline_ids != comparator_ids
        ):
            raise RuntimeError(
                f"Replacement row-index universe mismatch for seed {seed}."
            )
        aligned = baseline.merge(
            comparator,
            on="row_index",
            suffixes=("_bce", "_amount_gain"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if (
            len(aligned) != len(baseline)
            or not aligned["_merge"].eq("both").all()
        ):
            raise RuntimeError(
                f"Replacement alignment lost cases for seed {seed}."
            )
        aligned = aligned.drop(columns="_merge")
        if not (
            aligned["Class_bce"].astype(int).equals(
                aligned["Class_amount_gain"].astype(int)
            )
            and np.allclose(
                aligned["Amount_bce"].to_numpy(float),
                aligned["Amount_amount_gain"].to_numpy(float),
                rtol=0.0,
                atol=0.0,
            )
            and np.allclose(
                aligned["p_fraud_bce"].to_numpy(float),
                aligned["p_fraud_amount_gain"].to_numpy(float),
                rtol=0.0,
                atol=0.0,
            )
        ):
            raise RuntimeError(
                f"Replacement alignment changed case values for seed {seed}."
            )
        aligned = aligned.rename(
            columns={
                "Class_bce": "Class",
                "Amount_bce": "Amount",
                "p_fraud_bce": "p_fraud",
            }
        ).drop(
            columns=[
                "Class_amount_gain",
                "Amount_amount_gain",
                "p_fraud_amount_gain",
            ]
        )
        bce_top = set(
            aligned.loc[aligned["bce_rank"].astype(int) <= 50, "row_index"]
            .astype(int)
        )
        amount_top = set(
            aligned.loc[
                aligned["amount_gain_rank"].astype(int) <= 50, "row_index"
            ].astype(int)
        )
        added = amount_top - bce_top
        removed = bce_top - amount_top
        if added & removed or len(added) != len(removed):
            raise RuntimeError(
                f"Replacement-set invariant failed for seed {seed}."
            )
        fraud_amounts = aligned.loc[
            aligned["Class"].astype(int) == 1, "Amount"
        ].to_numpy(float)
        q90_threshold = float(np.quantile(fraud_amounts, 0.90))
        for direction, ids in (
            ("added_by_amount_gain", added),
            ("removed_from_bce", removed),
        ):
            events = aligned.loc[
                aligned["row_index"].astype(int).isin(ids)
            ].copy()
            events["seed"] = seed
            events["target_budget"] = 50
            events["direction"] = direction
            events["log1p_amount"] = np.log1p(events["Amount"].astype(float))
            events["q90_fraud_flag"] = (
                (events["Class"].astype(int) == 1)
                & (events["Amount"].astype(float) >= q90_threshold)
            )
            events["q90_threshold"] = q90_threshold
            event_frames.append(events)
            fraud_events = events.loc[events["Class"].astype(int) == 1]
            seedwise_rows.append(
                {
                    "seed": seed,
                    "target_budget": 50,
                    "direction": direction,
                    "case_count": len(events),
                    "fraud_count": len(fraud_events),
                    "q90_fraud_count": int(
                        events["q90_fraud_flag"].astype(bool).sum()
                    ),
                    "fraud_amount_sum": float(fraud_events["Amount"].sum()),
                    "mean_amount": (
                        float(events["Amount"].mean())
                        if len(events)
                        else np.nan
                    ),
                    "mean_bce_base_score": (
                        float(events["p_fraud"].mean())
                        if len(events)
                        else np.nan
                    ),
                }
            )
    events = pd.concat(event_frames, ignore_index=True)
    events["__direction_order"] = events["direction"].map(direction_order)
    seed_order = {seed: index for index, seed in enumerate(seeds)}
    events["__seed_order"] = events["seed"].map(seed_order)
    events = (
        events.sort_values(
            ["__seed_order", "__direction_order", "row_index"],
            kind="mergesort",
        )
        .drop(columns=["__seed_order", "__direction_order"])
        .reset_index(drop=True)
    )
    events = events[
        [
            "seed",
            "target_budget",
            "row_index",
            "direction",
            "Class",
            "Amount",
            "log1p_amount",
            "p_fraud",
            "q90_fraud_flag",
            "q90_threshold",
            "bce_rank",
            "amount_gain_rank",
        ]
    ]
    seedwise = pd.DataFrame(seedwise_rows)
    seedwise["__direction_order"] = seedwise["direction"].map(direction_order)
    seedwise = _ordered(
        seedwise,
        ["__direction_order", "seed"],
        seeds,
        budgets,
    ).drop(columns="__direction_order")
    summary = _summary(
        seedwise,
        ["target_budget", "direction"],
        [
            "case_count",
            "fraud_count",
            "q90_fraud_count",
            "fraud_amount_sum",
            "mean_amount",
            "mean_bce_base_score",
        ],
    )

    boundary = events.loc[
        events["bce_rank"].between(30, 70, inclusive="both")
        | events["amount_gain_rank"].between(30, 70, inclusive="both")
    ].copy()
    pooled_rows: list[dict[str, object]] = []
    for direction in direction_order:
        cases = boundary.loc[boundary["direction"] == direction]
        fraud_cases = cases.loc[cases["Class"].astype(int) == 1]
        pooled_rows.append(
            {
                "target_budget": 50,
                "window_lower_rank": 30,
                "window_upper_rank": 70,
                "direction": direction,
                "seed_count": len(seeds),
                "pooled_case_count": len(cases),
                "pooled_fraud_count": len(fraud_cases),
                "pooled_q90_fraud_count": int(
                    cases["q90_fraud_flag"].astype(bool).sum()
                ),
                "pooled_fraud_amount_sum": float(
                    fraud_cases["Amount"].sum()
                ),
                "pooled_mean_amount": float(cases["Amount"].mean()),
                "pooled_mean_bce_base_score": float(cases["p_fraud"].mean()),
                "aggregation_basis": (
                    "gepoolte Boundary-Events; Rangfenster 30–70 in mindestens "
                    "einer der beiden Ordnungen"
                ),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    return events, seedwise, summary, pooled


def derive_high_amount_legitimate(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the Amount-Gain@50 legitimate/high-Amount guardrail."""

    if 50 not in budgets:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    for seed in seeds:
        baseline = groups[(seed, 50, METHOD_BCE)].set_index(
            "row_index", drop=False
        )
        legitimate = baseline.loc[baseline["Class"].astype(int) == 0]
        threshold = float(
            np.quantile(legitimate["Amount"].to_numpy(float), 0.90)
        )
        amount_gain = groups[(seed, 50, METHOD_AMOUNT_GAIN)]
        selected = amount_gain.loc[
            amount_gain["final_rank_position"].astype(int) <= 50
        ]
        selected_legitimate = selected.loc[
            selected["Class"].astype(int) == 0
        ]
        count = len(selected_legitimate)
        high_count = int(
            (
                selected_legitimate["Amount"].astype(float) >= threshold
            ).sum()
        )
        rows.append(
            {
                "seed": seed,
                "target_budget": 50,
                "method_family": METHOD_AMOUNT_GAIN,
                "legitimate_count": count,
                "q90_legitimate_amount_threshold": threshold,
                "q90_high_amount_legitimate_count": high_count,
                "high_amount_legitimate_share": (
                    high_count / count if count else np.nan
                ),
                "mean_legitimate_amount": (
                    float(selected_legitimate["Amount"].mean())
                    if count
                    else np.nan
                ),
                "mean_bce_base_score": (
                    float(selected_legitimate["p_fraud"].mean())
                    if count
                    else np.nan
                ),
            }
        )
    seedwise = _decorate_paths(pd.DataFrame(rows))
    seedwise = _ordered(seedwise, ["seed"], seeds, budgets)
    summary = _summary(
        seedwise,
        ["target_budget", "method_family", "path_id", "path_label"],
        [
            "legitimate_count",
            "q90_high_amount_legitimate_count",
            "high_amount_legitimate_share",
            "mean_legitimate_amount",
            "mean_bce_base_score",
        ],
    )
    return seedwise, summary


def derive_pool_ceiling(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    """Derive Amount-Gain candidate-pool availability and ceiling utilization."""

    metric_lookup = seed_metrics.set_index(
        ["seed", "target_budget", "method_family"]
    )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for budget in CENTRAL_BUDGETS:
            if budget not in budgets:
                continue
            baseline = groups[(seed, budget, METHOD_BCE)].set_index(
                "row_index", drop=False
            )
            labels = baseline["Class"].to_numpy(int)
            amounts = baseline["Amount"].to_numpy(float)
            total_fraud_amount = float(amounts[labels == 1].sum())
            if total_fraud_amount <= 0:
                raise RuntimeError(
                    f"Fraud Amount denominator is not positive: {(seed, budget)}"
                )
            fraud = baseline.loc[baseline["Class"].astype(int) == 1]
            q90_threshold = float(
                np.quantile(fraud["Amount"].to_numpy(float), 0.90)
            )
            q90_ids = set(
                fraud.loc[fraud["Amount"].astype(float) >= q90_threshold]
                .index.astype(int)
            )
            candidate = baseline.loc[baseline["candidate_flag"].astype(bool)]
            candidate_ids = set(candidate["row_index"].astype(int))
            pool_fraud = fraud.loc[fraud.index.isin(candidate_ids)]
            pool_q90 = pool_fraud.loc[
                pool_fraud["Amount"].astype(float) >= q90_threshold
            ]
            max_fraud = min(budget, len(pool_fraud))
            max_fraud_amount = float(
                pool_fraud["Amount"]
                .nlargest(min(budget, len(pool_fraud)))
                .sum()
            )
            max_plr = max_fraud_amount / total_fraud_amount
            max_q90 = min(budget, len(pool_q90))
            amount_gain = groups[(seed, budget, METHOD_AMOUNT_GAIN)]
            top = amount_gain.loc[
                amount_gain["final_rank_position"].astype(int) <= budget
            ]
            top_ids = set(top["row_index"].astype(int))
            if not top_ids.issubset(candidate_ids):
                raise RuntimeError(
                    "Amount-Gain Top-k contains cases outside the registered "
                    f"candidate pool: {(seed, budget)}"
                )
            actual_fraud = int(top["Class"].astype(int).sum())
            actual_plr = float(
                top.loc[top["Class"].astype(int) == 1, "Amount"].sum()
                / total_fraud_amount
            )
            actual_q90 = int(
                (
                    (top["Class"].astype(int) == 1)
                    & (top["Amount"].astype(float) >= q90_threshold)
                ).sum()
            )
            _validate_frozen_actuals(
                actual_fraud,
                actual_plr,
                actual_q90,
                metric_lookup.loc[(seed, budget, METHOD_AMOUNT_GAIN)],
                context=f"candidate-pool ceiling {(seed, budget)}",
            )
            if (
                actual_fraud > max_fraud
                or actual_plr > max_plr + 1e-12
                or actual_q90 > max_q90
            ):
                raise RuntimeError(
                    f"Actual Top-k value exceeds its pool ceiling: {(seed, budget)}"
                )
            fraud_utilization = (
                actual_fraud / max_fraud if max_fraud else np.nan
            )
            plr_utilization = actual_plr / max_plr if max_plr else np.nan
            q90_utilization = actual_q90 / max_q90 if max_q90 else np.nan
            fraud_case_coverage = len(pool_fraud) / len(fraud)
            fraud_amount_coverage = float(
                pool_fraud["Amount"].sum() / fraud["Amount"].sum()
            )
            for name, value in (
                ("fraud_ceiling_utilization", fraud_utilization),
                ("plr_ceiling_utilization", plr_utilization),
                ("q90_ceiling_utilization", q90_utilization),
                ("candidate_pool_fraud_case_coverage", fraud_case_coverage),
                (
                    "candidate_pool_fraud_amount_coverage",
                    fraud_amount_coverage,
                ),
            ):
                if not np.isnan(value) and not -1e-12 <= value <= 1.0 + 1e-12:
                    raise RuntimeError(
                        f"{name} is outside [0, 1]: {(seed, budget, value)}"
                    )
            rows.append(
                {
                    "seed": seed,
                    "target_budget": budget,
                    "method_family": METHOD_AMOUNT_GAIN,
                    "candidate_pool_size": int(
                        candidate["candidate_pool_size"].iloc[0]
                    ),
                    "test_fraud_count": len(fraud),
                    "candidate_pool_fraud_count": len(pool_fraud),
                    "candidate_pool_fraud_case_coverage": fraud_case_coverage,
                    "test_fraud_amount": float(fraud["Amount"].sum()),
                    "candidate_pool_fraud_amount": float(
                        pool_fraud["Amount"].sum()
                    ),
                    "candidate_pool_fraud_amount_coverage": (
                        fraud_amount_coverage
                    ),
                    "test_q90_fraud_count": len(q90_ids),
                    "candidate_pool_q90_fraud_count": len(pool_q90),
                    "actual_fraud_at_k": actual_fraud,
                    "fraud_ceiling": max_fraud,
                    "fraud_ceiling_utilization": fraud_utilization,
                    "actual_plr_at_k": actual_plr,
                    "plr_ceiling": max_plr,
                    "plr_ceiling_utilization": plr_utilization,
                    "actual_q90_at_k": actual_q90,
                    "q90_ceiling": max_q90,
                    "q90_ceiling_utilization": q90_utilization,
                    "ceiling_interpretation": (
                        "Verfügbarkeitsgrenze; keine erwartete Modellleistung; "
                        "Outer-Test-Labels nur post hoc"
                    ),
                }
            )
    result = _decorate_paths(pd.DataFrame(rows))
    return _ordered(
        result,
        ["target_budget", "seed"],
        seeds,
        budgets,
    )


def derive_k50_diagnostic(
    tradeoff: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    pool: pd.DataFrame,
    ties: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    """Assemble the compact seedwise Amount-Gain-vs-BCE diagnostic."""

    paired = tradeoff.loc[
        (tradeoff["target_budget"].astype(int) == 50)
        & (tradeoff["method_family"] == METHOD_AMOUNT_GAIN),
        [
            "seed",
            "delta_plr_vs_bce",
            "delta_fraud_at_k_vs_bce",
            "fraud_at_k",
            "plr_at_k",
        ],
    ].rename(
        columns={
            "fraud_at_k": "actual_fraud_at_50",
            "plr_at_k": "actual_plr_at_50",
        }
    )
    selected_pool = pool.loc[
        pool["target_budget"].astype(int) == 50,
        [
            "seed",
            "fraud_ceiling_utilization",
            "plr_ceiling_utilization",
            "candidate_pool_fraud_case_coverage",
        ],
    ]
    selected_ties = ties.loc[
        (ties["target_budget"].astype(int) == 50)
        & (ties["method_family"] == METHOD_AMOUNT_GAIN),
        [
            "seed",
            "exact_tie_block_size",
            "fraud_at_k_min",
            "fraud_at_k_max",
            "plr_at_k_min",
            "plr_at_k_max",
            "technical_tie_classification",
        ],
    ]
    result = (
        paired.merge(selected_pool, on="seed", validate="one_to_one")
        .merge(selected_ties, on="seed", validate="one_to_one")
    )
    result.insert(1, "target_budget", 50)
    result.insert(2, "comparison", "Amount-Gain versus BCE")
    if len(result) != len(seeds):
        raise RuntimeError("Seedwise k=50 diagnostic is incomplete.")
    return _ordered(result, ["seed"], seeds, budgets)


def derive_heatmap(
    profile: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    """Prepare the fixed 5 x 7 Amount-Gain delta matrix in long form."""

    result = profile.loc[
        profile["method_family"] == METHOD_AMOUNT_GAIN,
        [
            "seed",
            "target_budget",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "delta_fraud_at_k_vs_bce",
        ],
    ].copy()
    baseline = profile.loc[
        profile["method_family"] == METHOD_BCE,
        ["seed", "target_budget", "prevented_loss_ratio_at_k"],
    ].rename(columns={"prevented_loss_ratio_at_k": "bce_plr_at_k"})
    result = result.merge(
        baseline,
        on=["seed", "target_budget"],
        validate="one_to_one",
    )
    result["delta_plr_vs_bce"] = (
        result["prevented_loss_ratio_at_k"] - result["bce_plr_at_k"]
    )
    result["seed_order"] = result["seed"].map(
        {seed: index for index, seed in enumerate(seeds)}
    )
    result["budget_order"] = result["target_budget"].map(
        {budget: index for index, budget in enumerate(budgets)}
    )
    result = result.sort_values(
        ["seed_order", "budget_order"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != len(seeds) * len(budgets):
        raise RuntimeError("Heatmap grid is not exactly seed x budget.")
    for metric in ("delta_plr_vs_bce", "delta_fraud_at_k_vs_bce"):
        matrix = result.pivot(
            index="seed", columns="target_budget", values=metric
        ).reindex(index=seeds, columns=budgets)
        if matrix.shape != (len(seeds), len(budgets)) or matrix.isna().any().any():
            raise RuntimeError(f"Incomplete heatmap matrix for {metric}.")
    return result


def derive_central_results(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare exact and summarized central Top-k results."""

    seedwise = _decorate_paths(
        seed_metrics.loc[
            seed_metrics["target_budget"].astype(int).isin(CENTRAL_BUDGETS)
        ].copy()
    )
    seedwise = _ordered(
        seedwise,
        ["target_budget", "method_family", "seed"],
        seeds,
        budgets,
    )
    summary = _summary(
        seedwise,
        ["target_budget", "method_family", "path_id", "path_label"],
        [
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "precision_at_k",
            "recall_at_k",
        ],
    )
    return seedwise, summary


def _validated_engineering_metrics(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    _require_columns(
        seed_metrics,
        [
            "seed",
            "target_budget",
            "method_family",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "precision_at_k",
            "recall_at_k",
        ],
        "engineering seed metrics",
    )
    missing_budgets = sorted(set(CENTRAL_BUDGETS) - set(budgets))
    if missing_budgets:
        raise RuntimeError(
            f"Engineering presentation is missing budget(s): {missing_budgets}."
        )
    if budgets != CENTRAL_BUDGETS:
        raise RuntimeError("Engineering presentation budget grid is incomplete.")
    unknown_methods = sorted(
        set(seed_metrics["method_family"].astype(str)) - set(METHOD_ORDER)
    )
    if unknown_methods:
        raise RuntimeError(
            f"Unknown method family in engineering data: {unknown_methods}."
        )
    selected = seed_metrics.loc[
        seed_metrics["seed"].astype(int).isin(seeds)
        & seed_metrics["target_budget"].astype(int).isin(budgets)
    ].copy()
    expected = {
        (seed, budget, method)
        for seed in seeds
        for budget in budgets
        for method in METHOD_ORDER
    }
    observed = set(
        selected[["seed", "target_budget", "method_family"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or selected.duplicated(
        ["seed", "target_budget", "method_family"]
    ).any():
        raise RuntimeError("Engineering seed-budget-path grid is incomplete.")
    return selected


def _engineering_metadata(
    frame: pd.DataFrame,
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "data_source_kind", data_source_kind)
    result.insert(0, "evidence_classification", evidence_classification)
    result.insert(0, "profile", profile)
    result.insert(
        3,
        "evidence_statement",
        engineering_evidence_statement(
            profile=profile,
            evidence_classification=evidence_classification,
            data_source_kind=data_source_kind,
        ),
    )
    return result


def derive_engineering_heatmap(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> pd.DataFrame:
    """Derive comparison-path deltas for the engineering heatmap."""

    selected = _validated_engineering_metrics(seed_metrics, seeds, budgets)
    baseline = selected.loc[
        selected["method_family"] == METHOD_BCE,
        ["seed", "target_budget", "prevented_loss_ratio_at_k", "frauds_at_k"],
    ].rename(
        columns={
            "prevented_loss_ratio_at_k": "bce_plr_at_k",
            "frauds_at_k": "bce_fraud_at_k",
        }
    )
    result = selected.loc[
        selected["method_family"].isin(COMPARISON_METHODS),
        [
            "seed",
            "target_budget",
            "method_family",
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
        ],
    ].rename(
        columns={
            "prevented_loss_ratio_at_k": "plr_at_k",
            "frauds_at_k": "fraud_at_k",
        }
    )
    result = result.merge(
        baseline,
        on=["seed", "target_budget"],
        how="left",
        validate="many_to_one",
    )
    if result[["bce_plr_at_k", "bce_fraud_at_k"]].isna().any().any():
        raise RuntimeError("Engineering BCE comparison rows are incomplete.")
    result["delta_plr_vs_bce"] = result["plr_at_k"] - result["bce_plr_at_k"]
    result["delta_fraud_at_k_vs_bce"] = (
        result["fraud_at_k"] - result["bce_fraud_at_k"]
    )
    result = _decorate_paths(result)
    result = _ordered(
        result,
        ["seed", "target_budget", "method_family"],
        seeds,
        budgets,
    )
    expected_rows = len(seeds) * len(budgets) * len(COMPARISON_METHODS)
    if len(result) != expected_rows:
        raise RuntimeError("Engineering heatmap inventory drift.")
    result = result[
        [
            "seed",
            "target_budget",
            "method_family",
            "path_id",
            "plr_at_k",
            "fraud_at_k",
            "bce_plr_at_k",
            "bce_fraud_at_k",
            "delta_plr_vs_bce",
            "delta_fraud_at_k_vs_bce",
        ]
    ]
    return _engineering_metadata(
        result,
        profile=profile,
        evidence_classification=evidence_classification,
        data_source_kind=data_source_kind,
    )


def derive_engineering_summary(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> pd.DataFrame:
    """Derive central Top-k means and sample SDs for engineering rendering."""

    selected = _decorate_paths(
        _validated_engineering_metrics(seed_metrics, seeds, budgets)
    )
    selected = _ordered(
        selected,
        ["target_budget", "method_family", "seed"],
        seeds,
        budgets,
    )
    summary = _summary(
        selected,
        ["target_budget", "method_family", "path_id"],
        [
            "prevented_loss_ratio_at_k",
            "frauds_at_k",
            "precision_at_k",
            "recall_at_k",
        ],
    ).rename(
        columns={
            "row_count": "seed_count",
            "prevented_loss_ratio_at_k_mean": "plr_mean",
            "prevented_loss_ratio_at_k_sd": "plr_sample_sd",
            "frauds_at_k_mean": "fraud_at_k_mean",
            "frauds_at_k_sd": "fraud_at_k_sample_sd",
            "precision_at_k_mean": "precision_at_k_mean",
            "precision_at_k_sd": "precision_at_k_sample_sd",
            "recall_at_k_mean": "recall_at_k_mean",
            "recall_at_k_sd": "recall_at_k_sample_sd",
        }
    )
    metric_counts = [
        "prevented_loss_ratio_at_k_n",
        "frauds_at_k_n",
        "precision_at_k_n",
        "recall_at_k_n",
    ]
    if not summary[metric_counts].eq(len(seeds)).all().all():
        raise RuntimeError("Engineering summary seed counts are incomplete.")
    sample_sd_columns = [
        "plr_sample_sd",
        "fraud_at_k_sample_sd",
        "precision_at_k_sample_sd",
        "recall_at_k_sample_sd",
    ]
    if len(seeds) == 1:
        if not summary[sample_sd_columns].isna().all().all():
            raise RuntimeError(
                "One-seed engineering sample SD must remain missing/NaN."
            )
    elif summary[sample_sd_columns].isna().any().any():
        raise RuntimeError("Engineering sample SD is unexpectedly missing.")
    expected_rows = len(budgets) * len(METHOD_ORDER)
    if len(summary) != expected_rows:
        raise RuntimeError("Engineering table inventory drift.")
    summary = summary[
        [
            "target_budget",
            "method_family",
            "path_id",
            "seed_count",
            "plr_mean",
            "plr_sample_sd",
            "fraud_at_k_mean",
            "fraud_at_k_sample_sd",
            "precision_at_k_mean",
            "precision_at_k_sample_sd",
            "recall_at_k_mean",
            "recall_at_k_sample_sd",
        ]
    ]
    return _engineering_metadata(
        summary,
        profile=profile,
        evidence_classification=evidence_classification,
        data_source_kind=data_source_kind,
    )


def derive_engineering(
    seed_metrics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    *,
    profile: str,
    evidence_classification: str,
    data_source_kind: str,
) -> dict[str, pd.DataFrame]:
    """Return the exact two-file engineering presentation-data inventory."""

    outputs = (
        derive_engineering_heatmap(
            seed_metrics,
            seeds,
            budgets,
            profile=profile,
            evidence_classification=evidence_classification,
            data_source_kind=data_source_kind,
        ),
        derive_engineering_summary(
            seed_metrics,
            seeds,
            budgets,
            profile=profile,
            evidence_classification=evidence_classification,
            data_source_kind=data_source_kind,
        ),
    )
    return dict(zip(ENGINEERING_OUTPUT_PATHS, outputs, strict=True))


def derive_all(
    groups: dict[tuple[int, int, str], pd.DataFrame],
    seed_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    """Return every R6 figure/table input keyed by data-root relative path."""

    tradeoff, tradeoff_summary = derive_tradeoff(
        seed_metrics, seeds, budgets
    )
    budget_profile, budget_summary = derive_budget_profile(
        seed_metrics, seeds, budgets
    )
    depth, depth_summary = derive_depth_profiles(
        groups, seed_metrics, seeds, budgets
    )
    (
        curves_seedwise,
        curves_summary,
        global_metrics_seedwise,
        global_metrics_summary,
    ) = derive_global_and_pool_curves(groups, diagnostics, seeds, budgets)
    hard, hard_summary = derive_hard_impact(
        groups, seed_metrics, seeds, budgets
    )
    ties = derive_tie_intervals(groups, seed_metrics, seeds, budgets)
    (
        replacement_events,
        replacement_seedwise,
        replacement_summary,
        boundary_pooled,
    ) = derive_replacement_events(groups, seeds, budgets)
    high_legit, high_legit_summary = derive_high_amount_legitimate(
        groups, seeds, budgets
    )
    pool = derive_pool_ceiling(groups, seed_metrics, seeds, budgets)
    diagnostic = derive_k50_diagnostic(
        tradeoff,
        seed_metrics,
        pool,
        ties,
        seeds,
        budgets,
    )
    heatmap = derive_heatmap(budget_profile, seeds, budgets)
    central_seedwise, central_summary = derive_central_results(
        seed_metrics, seeds, budgets
    )

    return {
        "figures/ch5_tradeoff_seedwise.csv": tradeoff,
        "figures/ch5_tradeoff_summary.csv": tradeoff_summary,
        "figures/ch5_budget_policy_seedwise.csv": budget_profile,
        "figures/ch5_budget_policy_summary.csv": budget_summary,
        "figures/ch5_depth_seedwise.csv": depth,
        "figures/ch5_depth_summary.csv": depth_summary,
        "figures/ch5_global_pool_curves_seedwise.csv": curves_seedwise,
        "figures/ch5_global_pool_curves_summary.csv": curves_summary,
        "figures/ch5_global_metrics_seedwise.csv": global_metrics_seedwise,
        "figures/ch5_global_metrics_summary.csv": global_metrics_summary,
        "figures/ch5_hard_impact_seedwise.csv": hard,
        "figures/ch5_hard_impact_summary.csv": hard_summary,
        "figures/ch5_replacement_events.csv": replacement_events,
        "figures/ch5_seedwise_k50_diagnostic.csv": diagnostic,
        "figures/app_seed_budget_delta_heatmap.csv": heatmap,
        "figures/app_exact_tie_intervals.csv": ties,
        "figures/app_candidate_pool_ceiling.csv": pool,
        "tables/ch5_t1_central_topk_results.csv": central_summary,
        "tables/ch5_t2_seedwise_k50_diagnostic.csv": diagnostic,
        "tables/ch5_t3_replacement_seedwise.csv": replacement_seedwise,
        "tables/ch5_t3_replacement_summary.csv": replacement_summary,
        "tables/ch5_t3_boundary_pooled.csv": boundary_pooled,
        "tables/ch5_t4_high_amount_legit_seedwise.csv": high_legit,
        "tables/ch5_t4_high_amount_legit_summary.csv": high_legit_summary,
        "tables/app_t1_hard_impact_exact_values.csv": hard,
        "tables/app_t2_global_metrics_by_budget.csv": global_metrics_summary,
        "tables/app_t3_exact_tie_bounds.csv": ties,
        "tables/app_t4_candidate_pool_coverage.csv": pool,
        "tables/app_t5_seedwise_central_results.csv": central_seedwise,
    }
