"""Rendering-independent serial orchestration for the frozen experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.artifacts import (
    _ensure_run_directories,
    _index_score_mapping_sha256,
    _write_csv,
    _write_json,
    _write_parquet,
    find_repository_root,
    score_vector_sha256,
)
from fraud_detection.errors import ProductError

from ..comparison_paths.amount_gain import fit_final_amount_gain_path
from ..comparison_paths.bce_baseline import baseline_bce_ranking
from ..comparison_paths.fixed_reference import fixed_reference_ranking
from ..comparison_paths.p_only import fit_final_p_only_path
from ..config import (
    METHOD_AMOUNT_GAIN,
    METHOD_BASELINE,
    METHOD_FAMILIES,
    METHOD_FIXED,
    METHOD_P_ONLY,
    RANKER_SCOPE,
    EffectiveExperimentConfig,
    ExperimentConfig,
    _emit_experiment_status,
    _experiment_event_sink,
    _final_ranker_completed,
    _status_utc_now,
    _utc_now,
    resolve_experiment_profile,
)
from ..evaluation.aggregation import aggregate_final_outputs
from ..evaluation.diagnostics import (
    boundary_rows,
    global_metric_rows,
    hard_impact_row,
    high_amount_legit_row,
    replacement_rows,
    tie_diagnostic_row,
)
from ..preparation.bce import fit_outer_bce_scores_after_freeze
from ..preparation.data import (
    _expected_outer_split_frauds,
    _expected_outer_split_rows,
    _load_preflight,
    _prepare_output_root,
    load_experiment_data,
    verify_outer_split_without_test_labels,
)
from ..prioritization.composition import validate_full_ranking
from ..prioritization.inputs import (
    amount_gain_candidate_features,
    build_amount_gain_relevance_labels,
    build_candidate_pool,
    candidate_group,
    candidate_pool_hash,
    candidate_rows_by_bce,
    p_only_candidate_features,
    relevance_distribution,
)
from ..prioritization.selection import (
    _load_and_verify_freeze,
    run_inner_phase,
)
from ..records import (
    ExperimentResult,
    fixed_reference_path_row,
    matched_metric_row,
    model_row,
    ranking_with_context,
    score_path,
    selected_gain_row,
)
from .integrity import run_validation_phase, write_failure_record
from .manifest import write_completed_run_manifest


def _validate_final_split_identity(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    outer_seed: int,
    effective_config: EffectiveExperimentConfig,
) -> None:
    expected_rows = _expected_outer_split_rows(effective_config)
    expected_frauds = _expected_outer_split_frauds(effective_config)
    actual_rows = (len(y_train), len(y_test))
    actual_frauds = (int(y_train.sum()), int(y_test.sum()))
    if actual_rows != expected_rows or (
        expected_frauds is not None and actual_frauds != expected_frauds
    ):
        raise RuntimeError(
            f"Final outer split identity failed for seed {outer_seed}."
        )


def _final_outer_seed(
    *,
    dataframe: pd.DataFrame,
    output_root: Path,
    outer_seed: int,
    configs: dict[tuple[int, int], dict[str, Any]],
    effective_config: EffectiveExperimentConfig,
    source_data_sha256: str,
    data_sha256: str,
) -> dict[str, pd.DataFrame]:
    train_index, test_index = verify_outer_split_without_test_labels(
        dataframe,
        output_root,
        outer_seed,
        effective_config,
    )
    (
        oof_scores,
        test_scores,
        bce_metadata,
        bce_diagnostics,
        bce_loss_history,
    ) = (
        fit_outer_bce_scores_after_freeze(
            dataframe=dataframe,
            output_root=output_root,
            outer_seed=outer_seed,
            train_index=train_index,
            test_index=test_index,
            effective_config=effective_config,
            data_sha256=data_sha256,
        )
    )
    p_train_oof = oof_scores["p_oof"].to_numpy(dtype=float)
    p_test = test_scores["p_full_train_test"].to_numpy(dtype=float)
    y_train = dataframe.loc[train_index, "Class"].to_numpy(dtype=int)
    y_test = dataframe.loc[test_index, "Class"].to_numpy(dtype=int)
    amount_train = dataframe.loc[train_index, "Amount"].to_numpy(dtype=float)
    amount_test = dataframe.loc[test_index, "Amount"].to_numpy(dtype=float)
    _validate_final_split_identity(
        y_train,
        y_test,
        outer_seed=outer_seed,
        effective_config=effective_config,
    )

    train_pool = build_candidate_pool(
        p_train_oof,
        train_index,
        candidate_pool_size=effective_config.candidate_pool_size,
    )
    test_pool = build_candidate_pool(
        p_test,
        test_index,
        candidate_pool_size=effective_config.candidate_pool_size,
    )
    train_pool_hash = candidate_pool_hash(train_pool)
    test_pool_hash = candidate_pool_hash(test_pool)
    train_relevance, thresholds = build_amount_gain_relevance_labels(
        y_train,
        amount_train,
    )
    train_candidate_positions = candidate_rows_by_bce(train_pool)[
        "original_position"
    ].to_numpy(dtype=int)
    train_candidate_relevance = train_relevance[train_candidate_positions]
    amount_train_features = amount_gain_candidate_features(
        train_pool,
        amount_train,
    )
    amount_test_features = amount_gain_candidate_features(
        test_pool,
        amount_test,
    )
    p_only_train_features = p_only_candidate_features(train_pool)
    p_only_test_features = p_only_candidate_features(test_pool)
    baseline_ranking = baseline_bce_ranking(test_pool)
    fixed_raw, fixed_ranking = fixed_reference_ranking(test_pool, amount_test)

    seed_dir = output_root / "final_outer_run" / f"seed_{outer_seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(
        seed_dir / "bce_oof_scores.csv.gz",
        oof_scores,
        compressed=True,
    )
    _write_csv(
        seed_dir / "bce_test_scores.csv.gz",
        test_scores,
        compressed=True,
    )
    _write_csv(seed_dir / "bce_fit_diagnostics.csv", bce_diagnostics)
    _write_csv(
        seed_dir / "bce_loss_history.csv.gz",
        bce_loss_history,
        compressed=True,
    )
    train_pool_artifact = train_pool.assign(
        seed=outer_seed,
        pool_role="outer_train_oof_bce",
        Class=y_train,
        Amount=amount_train,
        score_source="recreated_converged_outer_oof_bce",
    )
    test_pool_artifact = test_pool.assign(
        seed=outer_seed,
        pool_role="outer_test_full_train_bce",
        Class=y_test,
        Amount=amount_test,
        score_source="recreated_converged_full_outer_train_bce",
    )
    _write_parquet(seed_dir / "candidate_pool_train.parquet", train_pool_artifact)
    _write_parquet(seed_dir / "candidate_pool_test.parquet", test_pool_artifact)

    pool_summary = pd.DataFrame(
        [
            {
                "seed": outer_seed,
                "pool_role": "outer_train_oof_bce",
                "rows": len(train_pool),
                "candidate_count": int(train_pool["candidate_flag"].sum()),
                "candidate_pool_size": effective_config.candidate_pool_size,
                "candidate_pool_sha256": train_pool_hash,
                "score_sha256": score_vector_sha256(
                    p_train_oof,
                    score_type=f"outer_train_oof.seed_{outer_seed}",
                ),
                "index_score_mapping_sha256": _index_score_mapping_sha256(
                    train_index,
                    p_train_oof,
                    score_type=f"outer_train_oof.seed_{outer_seed}",
                ),
            },
            {
                "seed": outer_seed,
                "pool_role": "outer_test_full_train_bce",
                "rows": len(test_pool),
                "candidate_count": int(test_pool["candidate_flag"].sum()),
                "candidate_pool_size": effective_config.candidate_pool_size,
                "candidate_pool_sha256": test_pool_hash,
                "score_sha256": score_vector_sha256(
                    p_test,
                    score_type=f"outer_test_bce.seed_{outer_seed}",
                ),
                "index_score_mapping_sha256": _index_score_mapping_sha256(
                    test_index,
                    p_test,
                    score_type=f"outer_test_bce.seed_{outer_seed}",
                ),
            },
        ]
    )
    _write_csv(seed_dir / "candidate_pool_summary.csv", pool_summary)

    metric_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []
    amount_model_records: list[dict[str, Any]] = []
    p_only_model_records: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    replacement_row_records: list[dict[str, Any]] = []
    boundary_row_records: list[dict[str, Any]] = []
    high_legit_rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    selected_gain_rows: list[dict[str, Any]] = []

    for target_budget in effective_config.target_budgets:
        config = configs[(outer_seed, target_budget)]
        message = (
            f"[{_utc_now()}] final models seed={outer_seed} "
            f"k={target_budget} gain={config['selected_gain']} "
            f"trees={config['final_n_estimators']}"
        )
        print(message, flush=True)
        (
            _amount_ranker_config,
            amount_ranker,
            amount_raw,
            amount_ranking,
        ) = fit_final_amount_gain_path(
            train_features=amount_train_features,
            test_features=amount_test_features,
            train_candidate_relevance=train_candidate_relevance,
            test_candidate_pool=test_pool,
            target_budget=target_budget,
            selected_gain=str(config["selected_gain"]),
            random_state=outer_seed,
            final_n_estimators=int(config["final_n_estimators"]),
            effective_config=effective_config,
        )
        _emit_experiment_status(
            "INFO",
            "final-ranker-complete",
            seed=outer_seed,
            budget=target_budget,
            path=METHOD_AMOUNT_GAIN,
            completed=_final_ranker_completed(
                outer_seed,
                target_budget,
                0,
                effective_config,
            ),
            total=(
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
                * 2
            ),
        )
        (
            _p_only_ranker_config,
            p_only_ranker,
            p_only_raw,
            p_only_ranking,
        ) = fit_final_p_only_path(
            train_features=p_only_train_features,
            test_features=p_only_test_features,
            train_candidate_relevance=train_candidate_relevance,
            test_candidate_pool=test_pool,
            target_budget=target_budget,
            selected_gain=str(config["selected_gain"]),
            random_state=outer_seed,
            final_n_estimators=int(config["final_n_estimators"]),
            effective_config=effective_config,
        )
        _emit_experiment_status(
            "INFO",
            "final-ranker-complete",
            seed=outer_seed,
            budget=target_budget,
            path=METHOD_P_ONLY,
            completed=_final_ranker_completed(
                outer_seed,
                target_budget,
                1,
                effective_config,
            ),
            total=(
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
                * 2
            ),
        )
        amount_model_records.append(
            model_row(
                seed=outer_seed,
                budget=target_budget,
                model_type="amount_gain",
                config=config,
                ranker=amount_ranker,
                train_pool_hash=train_pool_hash,
                test_pool_hash=test_pool_hash,
                raw_scores=amount_raw,
                feature_names=[
                    "p_fraud",
                    "log1p_amount",
                    "p_fraud_x_log1p_amount",
                ],
            )
        )
        p_only_model_records.append(
            model_row(
                seed=outer_seed,
                budget=target_budget,
                model_type="p_only",
                config=config,
                ranker=p_only_ranker,
                train_pool_hash=train_pool_hash,
                test_pool_hash=test_pool_hash,
                raw_scores=p_only_raw,
                feature_names=["p_fraud"],
            )
        )
        selected_gain_rows.append(
            selected_gain_row(
                seed=outer_seed,
                target_budget=target_budget,
                config=config,
            )
        )
        fixed_rows.append(
            fixed_reference_path_row(
                seed=outer_seed,
                target_budget=target_budget,
                train_pool_hash=train_pool_hash,
                test_pool_hash=test_pool_hash,
                raw_scores=fixed_raw,
                effective_config=effective_config,
            )
        )

        rankings = {
            METHOD_BASELINE: baseline_ranking,
            METHOD_P_ONLY: p_only_ranking,
            METHOD_AMOUNT_GAIN: amount_ranking,
            METHOD_FIXED: fixed_ranking,
        }
        for method_family, ranking in rankings.items():
            path_name = score_path(method_family, target_budget)
            metric_row = matched_metric_row(
                outer_seed=outer_seed,
                target_budget=target_budget,
                method_family=method_family,
                score_path=path_name,
                ranking=ranking,
                y_true=y_test,
                amount=amount_test,
                selected_gain=str(config["selected_gain"]),
                selection_status=str(config["selection_status"]),
                truncation_level=int(config["truncation"]),
                final_n_estimators=int(config["final_n_estimators"]),
                effective_config=effective_config,
            )
            metric_rows.append(metric_row)
            ranking_frames.append(
                ranking_with_context(
                    ranking,
                    row_labels=y_test,
                    amounts=amount_test,
                    outer_seed=outer_seed,
                    target_budget=target_budget,
                    score_path=path_name,
                    method_family=method_family,
                    selected_gain=str(config["selected_gain"]),
                    selection_status=str(config["selection_status"]),
                    truncation_level=int(config["truncation"]),
                    final_n_estimators=int(config["final_n_estimators"]),
                    effective_config=effective_config,
                )
            )
            tie_rows.append(
                tie_diagnostic_row(
                    outer_seed=outer_seed,
                    target_budget=target_budget,
                    method_family=method_family,
                    score_path=path_name,
                    metric_row=metric_row,
                )
            )
            high_legit_rows.append(
                high_amount_legit_row(
                    outer_seed=outer_seed,
                    target_budget=target_budget,
                    method_family=method_family,
                    metric_row=metric_row,
                )
            )
            hard_rows.append(
                hard_impact_row(
                    outer_seed=outer_seed,
                    target_budget=target_budget,
                    method_family=method_family,
                    metric_row=metric_row,
                )
            )
            global_rows.extend(
                global_metric_rows(
                    seed=outer_seed,
                    budget=target_budget,
                    method_family=method_family,
                    score_path=path_name,
                    y_true=y_test,
                    ranking=ranking,
                )
            )
            if method_family != METHOD_BASELINE:
                replacement_row_records.extend(
                    replacement_rows(
                        seed=outer_seed,
                        budget=target_budget,
                        method_family=method_family,
                        baseline_ranking=baseline_ranking,
                        comparison_ranking=ranking,
                        y_true=y_test,
                        amount=amount_test,
                    )
                )
                boundary_row_records.extend(
                    boundary_rows(
                        seed=outer_seed,
                        budget=target_budget,
                        method_family=method_family,
                        baseline_ranking=baseline_ranking,
                        comparison_ranking=ranking,
                        y_true=y_test,
                        amount=amount_test,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    ranking_dump = pd.concat(ranking_frames, ignore_index=True)
    amount_models = pd.DataFrame(amount_model_records)
    p_only_models = pd.DataFrame(p_only_model_records)
    fixed_paths = pd.DataFrame(fixed_rows)
    ties = pd.DataFrame(tie_rows)
    replacements = pd.DataFrame(replacement_row_records)
    boundaries = pd.DataFrame(boundary_row_records)
    high_legit = pd.DataFrame(high_legit_rows)
    hard = pd.DataFrame(hard_rows)
    global_metrics = pd.DataFrame(global_rows)
    selected_gain_frame = pd.DataFrame(selected_gain_rows)

    budget_count = len(effective_config.target_budgets)
    method_count = len(METHOD_FAMILIES)
    if (
        len(amount_models) != budget_count
        or len(p_only_models) != budget_count
        or len(metrics) != budget_count * method_count
        or len(ranking_dump)
        != budget_count * method_count * len(test_index)
    ):
        raise RuntimeError(f"Final seed artifact counts failed for {outer_seed}.")
    for (_budget, _path), group in ranking_dump.groupby(
        ["target_budget", "score_path"],
        sort=False,
    ):
        if len(group) != len(test_index):
            raise RuntimeError("Incomplete final ranking path.")
        validate_full_ranking(group.reset_index(drop=True))
        if set(group["row_index"].astype(int)) != set(test_index):
            raise RuntimeError("Final ranking does not cover the test index set.")

    _write_csv(seed_dir / "selected_gain_by_budget.csv", selected_gain_frame)
    _write_csv(seed_dir / "matched_budget_metrics.csv", metrics)
    _write_parquet(seed_dir / "ranking_dump.parquet", ranking_dump)
    _write_csv(seed_dir / "topk_selected_paths.csv", metrics)
    _write_csv(seed_dir / "p_only_budget_models.csv", p_only_models)
    _write_csv(seed_dir / "amount_gain_budget_models.csv", amount_models)
    _write_csv(seed_dir / "fixed_reference_budget_paths.csv", fixed_paths)
    _write_csv(seed_dir / "tie_diagnostics.csv", ties)
    _write_csv(seed_dir / "replacement_diagnostics.csv", replacements)
    _write_csv(seed_dir / "boundary_diagnostics.csv", boundaries)
    _write_csv(
        seed_dir / "high_amount_legit_diagnostics.csv",
        high_legit,
    )
    _write_csv(seed_dir / "hard_impact_diagnostics.csv", hard)
    _write_csv(seed_dir / "global_metrics.csv", global_metrics)
    _write_json(
        seed_dir / "metadata.json",
        {
            "schema": "ranker_gain_validation.final_seed.v1",
            "completed_at_utc": _utc_now(),
            "seed": outer_seed,
            "ranker_scope": RANKER_SCOPE,
            "candidate_pool_size": effective_config.candidate_pool_size,
            "candidate_group": candidate_group(
                effective_config.candidate_pool_size
            ),
            "candidate_group_note": (
                "Technical ranking context; not an observed queue or real review list."
            ),
            "candidate_selection": (
                "BCE score descending; original position ascending on ties"
            ),
            "full_ranking_composition": (
                "candidates by raw score with BCE candidate-rank tie-break, "
                "then non-candidates in unchanged BCE order"
            ),
            "priority_order_score_semantics": (
                "Pure ordinal encoding of final order; not a probability, "
                "model output, or monetary value."
            ),
            "target_budgets": list(effective_config.target_budgets),
            "primary_budgets": list(effective_config.primary_budgets),
            "supplementary_budgets": list(
                effective_config.supplementary_budgets
            ),
            "selected_amount_gain_model_count": len(amount_models),
            "p_only_model_count": len(p_only_models),
            "p_only_configuration_matches_amount_gain": bool(
                np.array_equal(
                    amount_models[
                        [
                            "target_budget",
                            "selected_gain",
                            "truncation_level",
                            "configured_n_estimators",
                            "train_candidate_pool_sha256",
                            "test_candidate_pool_sha256",
                        ]
                    ].to_numpy(),
                    p_only_models[
                        [
                            "target_budget",
                            "selected_gain",
                            "truncation_level",
                            "configured_n_estimators",
                            "train_candidate_pool_sha256",
                            "test_candidate_pool_sha256",
                        ]
                    ].to_numpy(),
                )
            ),
            "fixed_reference_formula": "p_fraud * log1p(Amount)",
            "fixed_reference_same_candidate_pool": bool(
                (fixed_paths["test_candidate_pool_sha256"] == test_pool_hash).all()
            ),
            "fraud_amount_thresholds": [
                float(value) for value in thresholds
            ],
            "relevance_distribution": relevance_distribution(train_relevance),
            "train_index_sha256": bce_metadata["train_index_sha256"],
            "test_index_sha256": bce_metadata["test_index_sha256"],
            "bce_score_source": bce_metadata["score_source"],
            "bce_oof_score_sha256": bce_metadata["oof_score_sha256"],
            "bce_test_score_sha256": bce_metadata["test_score_sha256"],
            "bce_fit_count": bce_metadata["fit_count"],
            "bce_converged_fit_count": bce_metadata[
                "converged_fit_count"
            ],
            "source_data_sha256": source_data_sha256,
            "deduplicated_dataframe_sha256": data_sha256,
            "outer_test_selection_locked_before_access": True,
            "nonselected_gain_outer_model_count": 0,
        },
    )
    return {
        "metrics": metrics,
        "amount_models": amount_models,
        "p_only_models": p_only_models,
        "pool_summary": pool_summary,
        "ties": ties,
        "replacements": replacements,
        "boundaries": boundaries,
        "high_legit": high_legit,
        "hard": hard,
        "global_metrics": global_metrics,
    }


def run_final_phase(
    args: argparse.Namespace,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    output_root = Path(args.output_dir).resolve()
    _ensure_run_directories(output_root)
    _load_preflight(output_root, effective_config)
    freeze_manifest, configs = _load_and_verify_freeze(
        output_root,
        effective_config,
    )
    if not freeze_manifest["outer_test_selection_locked"]:
        raise RuntimeError("Outer test selection is not locked.")
    final_manifest_path = (
        output_root / "final_outer_run" / "final_outer_manifest.json"
    )
    if final_manifest_path.exists():
        raise FileExistsError("Final outer run already exists.")

    # This is the first phase that materializes outer test labels and metrics.
    dataframe, data_metadata = load_experiment_data(
        Path(args.data_path),
        effective_config,
    )
    seed_results: list[dict[str, pd.DataFrame]] = []
    for outer_seed in effective_config.seeds:
        seed_results.append(
            _final_outer_seed(
                dataframe=dataframe,
                output_root=output_root,
                outer_seed=outer_seed,
                configs=configs,
                effective_config=effective_config,
                source_data_sha256=str(data_metadata["source_data_sha256"]),
                data_sha256=str(
                    data_metadata["deduplicated_dataframe_sha256"]
                ),
            )
        )
    _emit_experiment_status(
        "INFO",
        "aggregation-start",
        phase="final",
        seeds=len(effective_config.seeds),
        utc=_status_utc_now(),
    )
    aggregate_final_outputs(
        output_root=output_root,
        seed_results=seed_results,
        effective_config=effective_config,
        data_sha256=str(data_metadata["deduplicated_dataframe_sha256"]),
    )
    _emit_experiment_status(
        "PASS",
        "aggregation-complete",
        phase="final",
    )
    _write_json(
        final_manifest_path,
        {
            "schema": "ranker_gain_validation.final_outer.v1",
            "status": "PASS",
            "completed_at_utc": _utc_now(),
            "selection_freeze_timestamp_utc": freeze_manifest[
                "freeze_timestamp_utc"
            ],
            "outer_test_selection_locked_before_access": True,
            "outer_seed_count": len(effective_config.seeds),
            "target_budget_count": len(effective_config.target_budgets),
            "selected_amount_gain_model_count": (
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
            ),
            "p_only_model_count": (
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
            ),
            "fixed_reference_path_count": (
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
            ),
            "baseline_matched_path_count": (
                len(effective_config.seeds)
                * len(effective_config.target_budgets)
            ),
            "nonselected_gain_outer_model_count": 0,
            "ranking_dump_count": len(effective_config.seeds),
            "ranker_scope": RANKER_SCOPE,
        },
    )
    _emit_experiment_status(
        "PASS",
        "artifact-manifest-complete",
        kind="final",
        manifest=(
            final_manifest_path
        ),
    )
    return data_metadata


def _namespace_from_config(
    config: ExperimentConfig,
    effective_config: EffectiveExperimentConfig,
    repository_root: Path | None = None,
) -> argparse.Namespace:
    data_path = config.data_path
    output_root = config.output_root
    if repository_root is not None:
        data_path = (
            data_path
            if data_path.is_absolute()
            else repository_root / data_path
        )
        output_root = (
            output_root
            if output_root.is_absolute()
            else repository_root / output_root
        )
    return argparse.Namespace(
        data_path=str(data_path),
        output_dir=str(output_root),
        phase=config.phase,
        ranker_scope=RANKER_SCOPE,
        candidate_pool_size=effective_config.candidate_pool_size,
        target_budgets=effective_config.target_budgets,
        gain_candidates=effective_config.enabled_gain_profiles,
        ranker_early_stopping_rounds=(
            effective_config.ranker_early_stopping_rounds
        ),
        seeds=effective_config.seeds,
    )


def _discard_event(_kind: str, _payload: dict[str, object]) -> None:
    return None


def _resolve_repository_root(config: ExperimentConfig) -> Path:
    if config.repository_root is not None:
        supplied_root = config.repository_root.resolve()
        discovered_root = find_repository_root(supplied_root)
        if discovered_root != supplied_root:
            raise ProductError(
                "FD-ROOT-NOT-FOUND",
                "Repository root was not found.",
                ("Run the command from the repository checkout.",),
                {"path": str(supplied_root)},
            )
        return supplied_root

    for candidate in (config.output_root, Path.cwd()):
        discovered_root = find_repository_root(candidate)
        if discovered_root is not None:
            return discovered_root.resolve()
    raise ProductError(
        "FD-ROOT-NOT-FOUND",
        "Repository root was not found.",
        ("Run the command from the repository checkout.",),
    )


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run the requested deterministic phase sequence without rendering."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig instance.")
    if config.phase not in ("inner", "final", "qa", "all"):
        raise ValueError(f"Unsupported experiment phase: {config.phase!r}.")
    effective_config = resolve_experiment_profile(config.profile)
    repository_root = _resolve_repository_root(config)
    sink = config.event_sink or _discard_event
    with _experiment_event_sink(sink):
        return _run_experiment(config, effective_config, repository_root)


def _run_experiment(
    config: ExperimentConfig,
    effective_config: EffectiveExperimentConfig,
    repository_root: Path,
) -> ExperimentResult:
    args = _namespace_from_config(config, effective_config, repository_root)
    run_started = time.perf_counter()
    _emit_experiment_status(
        "INFO",
        "experiment-start",
        requested_phase=args.phase,
        output_root=Path(args.output_dir).resolve(),
        utc=_status_utc_now(),
    )
    preflight_started = time.perf_counter()
    _emit_experiment_status(
        "INFO",
        "preflight-start",
        requested_phase=args.phase,
        utc=_status_utc_now(),
    )
    try:
        _prepare_output_root(args, effective_config, repository_root)
        output_root = Path(args.output_dir).resolve()
    except (KeyboardInterrupt, Exception) as error:
        if isinstance(error, KeyboardInterrupt):
            _emit_experiment_status(
                "FAIL",
                "experiment-interrupted",
                active_phase="preflight",
                elapsed_seconds=time.perf_counter() - run_started,
            )
        else:
            _emit_experiment_status(
                "FAIL",
                "experiment-failed",
                active_phase="preflight",
                exception=type(error).__name__,
                elapsed_seconds=time.perf_counter() - run_started,
            )
        raise
    _emit_experiment_status(
        "PASS",
        "preflight-complete",
        output_root=output_root,
        elapsed_seconds=time.perf_counter() - preflight_started,
    )
    active_phase = args.phase
    phase_started = time.perf_counter()
    try:
        if args.phase == "inner":
            active_phase = "inner"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            run_inner_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        elif args.phase == "final":
            active_phase = "final"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            run_final_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        elif args.phase == "qa":
            active_phase = "qa"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            run_validation_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
        else:
            active_phase = "inner"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            run_inner_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
            active_phase = "final"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            data_metadata = run_final_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
            active_phase = "qa"
            phase_started = time.perf_counter()
            _emit_experiment_status(
                "INFO",
                "phase-start",
                phase=active_phase,
                utc=_status_utc_now(),
            )
            run_validation_phase(args, effective_config)
            _emit_experiment_status(
                "PASS",
                "phase-complete",
                phase=active_phase,
                elapsed_seconds=time.perf_counter() - phase_started,
            )
            write_completed_run_manifest(
                output_root=output_root,
                effective_config=effective_config,
                data_metadata=data_metadata,
            )
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            _emit_experiment_status(
                "FAIL",
                "experiment-interrupted",
                active_phase=active_phase,
                elapsed_seconds=time.perf_counter() - run_started,
            )
        else:
            _emit_experiment_status(
                "FAIL",
                "experiment-failed",
                active_phase=active_phase,
                exception=type(error).__name__,
                elapsed_seconds=time.perf_counter() - run_started,
            )
        if output_root.is_dir() and (output_root / "logs").is_dir():
            write_failure_record(
                output_root=output_root,
                phase=args.phase,
                error=error,
            )
        raise
    _emit_experiment_status(
        "PASS",
        "experiment-exit",
        requested_phase=args.phase,
        status="COMPLETE" if args.phase == "all" else "PHASE_COMPLETE",
        exit_code=0,
        elapsed_seconds=time.perf_counter() - run_started,
    )
    return ExperimentResult(
        output_root=output_root,
        requested_phase=config.phase,
        status="COMPLETE" if config.phase == "all" else "PHASE_COMPLETE",
        completed_phases=(
            ("inner", "final", "qa")
            if config.phase == "all"
            else (config.phase,)
        ),
    )
