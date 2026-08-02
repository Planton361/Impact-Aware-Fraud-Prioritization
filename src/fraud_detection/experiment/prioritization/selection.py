"""Frozen inner policy selection for the learned comparison paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fraud_detection.artifacts import (
    _ensure_run_directories,
    _integer_vector_sha256,
    _sha256_file,
    _verify_checksum_manifest,
    _write_checksum_manifest,
    _write_csv,
    _write_json,
    _write_parquet,
    canonical_json_sha256,
    score_vector_sha256,
)

from ..comparison_paths.bce_baseline import baseline_bce_ranking
from ..comparison_paths.fixed_reference import fixed_reference_ranking
from ..config import (
    BCE_FEATURES,
    BCE_L2_ALPHA,
    BCE_LEARNING_RATE,
    BCE_MAX_ITER,
    BCE_TOL,
    GAIN_PROFILES,
    METHOD_AMOUNT_GAIN,
    METHOD_BASELINE,
    METHOD_FIXED,
    RANKER_LEARNING_RATE,
    RANKER_MIN_CHILD_SAMPLES,
    RANKER_MIN_CHILD_WEIGHT,
    RANKER_N_JOBS,
    RANKER_NUM_LEAVES,
    RANKER_REG_LAMBDA,
    RANKER_SCOPE,
    EffectiveExperimentConfig,
    _emit_experiment_log,
    _emit_experiment_status,
    _inner_ranker_completed,
    _selection_freeze_completed,
    _utc_now,
)
from ..evaluation.metrics import matched_budget_metrics
from ..preparation.bce import fit_inner_bce_fold
from ..preparation.data import (
    _expected_outer_split_frauds,
    _expected_outer_split_rows,
    _load_preflight,
    load_experiment_data,
    preflight_split_identity,
    verify_outer_split_without_test_labels,
)
from .inputs import (
    amount_gain_candidate_features,
    build_amount_gain_relevance_labels,
    build_candidate_pool,
    candidate_group,
    candidate_pool_hash,
    candidate_rows_by_bce,
    relevance_distribution,
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def truncation_for_budget(
    target_budget: object,
    effective_config: EffectiveExperimentConfig,
) -> int:
    budget = _positive_int(target_budget, "target_budget")
    if budget not in effective_config.target_budgets:
        raise ValueError(
            "target_budget must be one of "
            f"{list(effective_config.target_budgets)}."
        )
    return budget + 3


def eval_at_for_budget(
    target_budget: object,
    effective_config: EffectiveExperimentConfig,
) -> tuple[int]:
    budget = _positive_int(target_budget, "target_budget")
    truncation_for_budget(budget, effective_config)
    return (budget,)


def select_gain_from_inner_results(
    inner_amount_gain_results: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Apply the locked two-gain, three-fold selection rule."""

    required = {
        "gain_profile",
        "inner_fold",
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "bce_prevented_loss_ratio_at_k",
        "bce_frauds_at_k",
        "amount_ndcg_at_k",
        "cutoff_tie_size",
        "best_iteration",
        "fit_valid",
    }
    missing = sorted(required - set(inner_amount_gain_results.columns))
    if missing:
        raise ValueError(f"inner results are missing columns: {missing}")
    frame = inner_amount_gain_results.copy()
    expected_gains = set(effective_config.enabled_gain_profiles)
    expected_folds = set(range(1, effective_config.inner_folds + 1))
    if set(frame["gain_profile"].astype(str)) != expected_gains:
        raise ValueError("Inner results do not contain the configured gains.")
    if set(frame["inner_fold"].astype(int)) != expected_folds:
        raise ValueError("Inner results do not contain the configured folds.")
    expected_rows = len(expected_gains) * len(expected_folds)
    if len(frame) != expected_rows or frame.duplicated(
        ["gain_profile", "inner_fold"]
    ).any():
        raise ValueError("Inner results must contain one row per gain and fold.")
    if not frame["fit_valid"].astype(bool).all():
        raise ValueError("All configured inner Amount-Gain fits must be valid.")
    if (frame["bce_frauds_at_k"].astype(float) == 0.0).any():
        raise ValueError(
            "Fraud_BCE is zero in at least one fold; selection is invalid."
        )

    finite_columns = [
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "bce_prevented_loss_ratio_at_k",
        "bce_frauds_at_k",
        "amount_ndcg_at_k",
        "cutoff_tie_size",
        "best_iteration",
    ]
    values = frame[finite_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All inner selection metrics must be finite.")
    if (frame["best_iteration"].astype(int) < 1).any() or (
        frame["best_iteration"].astype(int)
        > effective_config.ranker_max_estimators
    ).any():
        raise ValueError("best_iteration exceeds the configured tree ceiling.")

    frame["delta_plr"] = (
        frame["prevented_loss_ratio_at_k"].astype(float)
        - frame["bce_prevented_loss_ratio_at_k"].astype(float)
    )
    frame["fraud_retention"] = (
        frame["frauds_at_k"].astype(float)
        / frame["bce_frauds_at_k"].astype(float)
    )
    summaries: list[dict[str, object]] = []
    for gain_profile in effective_config.enabled_gain_profiles:
        rows = frame.loc[
            frame["gain_profile"].astype(str) == gain_profile
        ].sort_values("inner_fold")
        summary = {
            "gain_profile": gain_profile,
            "mean_fraud_retention": float(rows["fraud_retention"].mean()),
            "min_fraud_retention": float(rows["fraud_retention"].min()),
            "mean_delta_plr": float(rows["delta_plr"].mean()),
            "positive_delta_plr_fold_count": int((rows["delta_plr"] > 0.0).sum()),
            "mean_amount_ndcg_at_k": float(rows["amount_ndcg_at_k"].mean()),
            "mean_cutoff_tie_size": float(rows["cutoff_tie_size"].mean()),
            "all_configured_fits_valid": bool(
                rows["fit_valid"].astype(bool).all()
            ),
            "all_selection_metrics_finite": bool(
                np.isfinite(
                    rows[
                        [
                            "fraud_retention",
                            "delta_plr",
                            "amount_ndcg_at_k",
                            "cutoff_tie_size",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            ),
        }
        summary["plr_eligible"] = bool(
            summary["mean_delta_plr"] > 0.0
            and summary["positive_delta_plr_fold_count"] >= 2
            and summary["all_configured_fits_valid"]
            and summary["all_selection_metrics_finite"]
        )
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    eligible = summary_frame.loc[summary_frame["plr_eligible"].astype(bool)]
    candidates = eligible if not eligible.empty else summary_frame
    gain_tiebreak = {"exponential": 0, "linear": 1}
    ordered = sorted(
        candidates.to_dict(orient="records"),
        key=lambda row: (
            -float(row["mean_fraud_retention"]),
            -float(row["min_fraud_retention"]),
            -float(row["mean_delta_plr"]),
            -float(row["mean_amount_ndcg_at_k"]),
            float(row["mean_cutoff_tie_size"]),
            gain_tiebreak[str(row["gain_profile"])],
        ),
    )
    selected_gain = str(ordered[0]["gain_profile"])
    selected_rows = frame.loc[
        frame["gain_profile"].astype(str) == selected_gain
    ].sort_values("inner_fold")
    iterations = selected_rows["best_iteration"].astype(int).tolist()
    median_iteration = float(np.median(iterations))
    final_n_estimators = int(
        min(
            effective_config.ranker_max_estimators,
            max(1, math.floor(median_iteration + 0.5)),
        )
    )
    selection_status = (
        "POSITIVE_INNER_PLR_LIFT"
        if not eligible.empty
        else "NO_INNER_VALIDATED_POSITIVE_PLR_LIFT"
    )
    selected_summary = summary_frame.loc[
        summary_frame["gain_profile"] == selected_gain
    ].iloc[0]
    selection = {
        "selected_gain": selected_gain,
        "selection_status": selection_status,
        "best_iteration_fold_1": iterations[0],
        "best_iteration_fold_2": iterations[1],
        "best_iteration_fold_3": (
            iterations[2] if len(iterations) == 3 else None
        ),
        "final_n_estimators": final_n_estimators,
        **{
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in selected_summary.to_dict().items()
            if key != "gain_profile"
        },
    }
    return selection, frame, summary_frame


def _inner_matched_comparison(fit_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "precision_at_k",
        "recall_at_k",
        "fraud_amount_sum_at_k",
        "legit_count_at_k",
        "amount_ndcg_at_k",
        "q90_captured_ratio_at_k",
        "q90_amount_ndcg_at_k",
        "unique_raw_ranker_scores",
        "cutoff_tie_size",
    ]
    rows: list[dict[str, Any]] = []
    group_columns = ["outer_seed", "inner_fold", "target_budget"]
    for key, group in fit_frame.groupby(group_columns, sort=False):
        first = group.iloc[0]
        for method_family, prefix in (
            (METHOD_BASELINE, "bce_"),
            (METHOD_FIXED, "fixed_reference_"),
        ):
            row = {
                "outer_seed": int(key[0]),
                "inner_fold": int(key[1]),
                "target_budget": int(key[2]),
                "method_family": method_family,
                "gain_profile": "not_applicable",
                "inner_only": True,
            }
            for metric in metric_columns:
                row[metric] = first[f"{prefix}{metric}"]
            rows.append(row)
        for fit in group.itertuples(index=False):
            row = {
                "outer_seed": int(fit.outer_seed),
                "inner_fold": int(fit.inner_fold),
                "target_budget": int(fit.target_budget),
                "method_family": METHOD_AMOUNT_GAIN,
                "gain_profile": str(fit.gain_profile),
                "inner_only": True,
            }
            for metric in metric_columns:
                row[metric] = getattr(fit, metric)
            rows.append(row)
    return pd.DataFrame(rows)


def _selection_config_payload(
    *,
    output_root: Path,
    data_metadata: dict[str, Any],
    outer_seed: int,
    target_budget: int,
    selection: dict[str, Any],
    gain_summaries: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    identity = preflight_split_identity(output_root, outer_seed)
    split_hash = canonical_json_sha256(
        {
            "train_index_sha256": identity["train_index_sha256"],
            "test_index_sha256": identity["test_index_sha256"],
        }
    )
    selected_gain = str(selection["selected_gain"])
    payload: dict[str, Any] = {
        "schema": "ranker_gain_validation.selected_config.v1",
        "outer_seed": outer_seed,
        "target_budget": target_budget,
        "primary_budget": target_budget in effective_config.primary_budgets,
        "ranker_scope": RANKER_SCOPE,
        "candidate_pool_size": effective_config.candidate_pool_size,
        "candidate_training_score_source": "outer_train_oof_bce",
        "candidate_test_score_source": "full_outer_train_bce_test_score",
        "selected_gain": selected_gain,
        "label_gain": list(GAIN_PROFILES[selected_gain]),
        "selection_status": selection["selection_status"],
        "best_iteration_fold_1": selection["best_iteration_fold_1"],
        "best_iteration_fold_2": selection["best_iteration_fold_2"],
        "best_iteration_fold_3": selection["best_iteration_fold_3"],
        "final_n_estimators": selection["final_n_estimators"],
        "truncation": truncation_for_budget(target_budget, effective_config),
        "eval_at": [target_budget],
        "fixed_model_capacity": {
            "objective": "lambdarank",
            "learning_rate": RANKER_LEARNING_RATE,
            "num_leaves": RANKER_NUM_LEAVES,
            "min_child_samples": RANKER_MIN_CHILD_SAMPLES,
            "min_child_weight": RANKER_MIN_CHILD_WEIGHT,
            "reg_lambda": RANKER_REG_LAMBDA,
            "n_jobs": RANKER_N_JOBS,
            "verbosity": -1,
            "early_stopping_rounds_inner_only": (
                effective_config.ranker_early_stopping_rounds
            ),
            "n_estimators_inner_safety_cap": (
                effective_config.ranker_max_estimators
            ),
        },
        "selected_inner_metrics": {
            key: value
            for key, value in selection.items()
            if key
            not in {
                "selected_gain",
                "selection_status",
                "best_iteration_fold_1",
                "best_iteration_fold_2",
                "best_iteration_fold_3",
                "final_n_estimators",
            }
        },
        "inner_gain_summaries": gain_summaries.to_dict(orient="records"),
        "data_sha256": data_metadata["deduplicated_dataframe_sha256"],
        "split_sha256": split_hash,
        "train_index_sha256": identity["train_index_sha256"],
        "test_index_sha256": identity["test_index_sha256"],
        "outer_test_selection_locked": True,
        "outer_test_labels_used_for_selection": False,
        "outer_test_metrics_used_for_selection": False,
    }
    payload["config_hash"] = canonical_json_sha256(payload)
    return payload


def _load_and_verify_freeze(
    output_root: Path,
    effective_config: EffectiveExperimentConfig,
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    freeze_dir = output_root / "selection_freeze"
    manifest_path = freeze_dir / "selection_manifest.json"
    checksum_path = freeze_dir / "checksums.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise RuntimeError("Selection freeze is incomplete.")
    _verify_checksum_manifest(output_root, checksum_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("outer_test_selection_locked") is not True:
        raise RuntimeError("outer_test_selection_locked is not true.")
    expected_config_count = len(effective_config.seeds) * len(
        effective_config.target_budgets
    )
    if manifest.get("selected_config_count") != expected_config_count:
        raise RuntimeError("Selection freeze has an incorrect config count.")
    if manifest.get("gain_candidates") != list(
        effective_config.enabled_gain_profiles
    ):
        raise RuntimeError("Selection freeze contains unexpected gains.")

    configs: dict[tuple[int, int], dict[str, Any]] = {}
    for entry in manifest["selected_configs"]:
        path = output_root / entry["path"]
        if _sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Selected config hash mismatch: {path}")
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        key = (int(config["outer_seed"]), int(config["target_budget"]))
        if key in configs:
            raise RuntimeError(f"Duplicate selected config key: {key}")
        payload_without_hash = {
            name: value
            for name, value in config.items()
            if name != "config_hash"
        }
        if canonical_json_sha256(payload_without_hash) != config["config_hash"]:
            raise RuntimeError(f"Config hash mismatch for {key}.")
        preflight_split_identity(output_root, key[0])
        if (
            config["selected_gain"]
            not in effective_config.enabled_gain_profiles
            or config["label_gain"]
            != list(GAIN_PROFILES[config["selected_gain"]])
            or int(config["truncation"])
            != truncation_for_budget(key[1], effective_config)
            or config["eval_at"] != [key[1]]
            or not 1
            <= int(config["final_n_estimators"])
            <= effective_config.ranker_max_estimators
        ):
            raise RuntimeError(f"Invalid selected configuration for {key}.")
        configs[key] = config
    expected = {
        (seed, budget)
        for seed in effective_config.seeds
        for budget in effective_config.target_budgets
    }
    if set(configs) != expected:
        raise RuntimeError("Selection freeze seed-budget coverage is incomplete.")
    return manifest, configs


def _validate_outer_train_identity(
    outer_train: pd.DataFrame,
    outer_y: np.ndarray,
    *,
    outer_seed: int,
    effective_config: EffectiveExperimentConfig,
) -> None:
    expected_train_rows = _expected_outer_split_rows(effective_config)[0]
    expected_frauds = _expected_outer_split_frauds(effective_config)
    if len(outer_train) != expected_train_rows or (
        expected_frauds is not None
        and int(outer_y.sum()) != expected_frauds[0]
    ):
        raise RuntimeError(
            f"Outer training identity mismatch for seed {outer_seed}."
        )


def _inner_score_frames(
    *,
    outer_seed: int,
    inner_fold: int,
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    inner_train_outer_positions: np.ndarray,
    validation_outer_positions: np.ndarray,
    bce_result: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_scores = pd.DataFrame(
        {
            "outer_seed": outer_seed,
            "inner_fold": inner_fold,
            "score_source": "inner_train_oof_bce",
            "inner_train_position": np.arange(len(inner_train)),
            "outer_train_position": inner_train_outer_positions,
            "row_index": inner_train.index.to_numpy(dtype=int),
            "Class": inner_train["Class"].to_numpy(dtype=int),
            "oof_fold_number": bce_result["oof_fold_number"],
            "p_fraud": bce_result["p_train_oof"],
        }
    )
    validation_scores = pd.DataFrame(
        {
            "outer_seed": outer_seed,
            "inner_fold": inner_fold,
            "score_source": "full_inner_train_bce_validation_score",
            "inner_validation_position": np.arange(len(inner_validation)),
            "outer_train_position": validation_outer_positions,
            "row_index": inner_validation.index.to_numpy(dtype=int),
            "Class": inner_validation["Class"].to_numpy(dtype=int),
            "p_fraud": bce_result["p_validation"],
        }
    )
    return train_scores, validation_scores


def _write_inner_fold_artifacts(
    *,
    output_root: Path,
    outer_seed: int,
    inner_fold: int,
    train_scores: pd.DataFrame,
    validation_scores: pd.DataFrame,
    train_pool: pd.DataFrame,
    validation_pool: pd.DataFrame,
    thresholds: np.ndarray,
    train_relevance: np.ndarray,
    validation_relevance: np.ndarray,
    bce_result: dict[str, Any],
    effective_config: EffectiveExperimentConfig,
) -> None:
    fold_dir = (
        output_root
        / "inner_validation"
        / f"seed_{outer_seed}"
        / f"fold_{inner_fold}"
    )
    fold_dir.mkdir(parents=True, exist_ok=False)
    _write_parquet(fold_dir / "bce_oof_scores.parquet", train_scores)
    _write_parquet(
        fold_dir / "bce_validation_scores.parquet",
        validation_scores,
    )
    _write_parquet(
        fold_dir / "candidate_pool_train.parquet",
        train_pool.assign(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            pool_role="inner_train_oof",
        ),
    )
    _write_parquet(
        fold_dir / "candidate_pool_validation.parquet",
        validation_pool.assign(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            pool_role="inner_validation_full_train_score",
        ),
    )
    train_label_frame = pd.DataFrame(
        {
            "outer_seed": outer_seed,
            "inner_fold": inner_fold,
            "row_index": train_scores["row_index"].to_numpy(dtype=int),
            "Class": train_scores["Class"].to_numpy(dtype=int),
            "Amount": np.nan,
            "relevance_label": train_relevance,
            "label_role": "inner_train_threshold_fit",
        }
    ).drop(columns=["Amount"])
    validation_label_frame = pd.DataFrame(
        {
            "outer_seed": outer_seed,
            "inner_fold": inner_fold,
            "row_index": validation_scores["row_index"].to_numpy(dtype=int),
            "Class": validation_scores["Class"].to_numpy(dtype=int),
            "relevance_label": validation_relevance,
            "label_role": "inner_validation_reused_train_thresholds",
        }
    )
    _write_parquet(
        fold_dir / "relevance_labels_train.parquet",
        train_label_frame,
    )
    _write_parquet(
        fold_dir / "relevance_labels_validation.parquet",
        validation_label_frame,
    )
    metadata = {
        "outer_seed": outer_seed,
        "inner_fold": inner_fold,
        "inner_splitter": {
            "class": "StratifiedKFold",
            "n_splits": effective_config.inner_folds,
            "shuffle": True,
            "random_state": 100000 + outer_seed,
        },
        "inner_bce_oof": {
            "folds": effective_config.bce_oof_folds,
            "configured_fit_count": effective_config.bce_oof_folds + 1,
            "random_state": 100000 + outer_seed,
            "features": list(BCE_FEATURES),
            "learning_rate": BCE_LEARNING_RATE,
            "tol": BCE_TOL,
            "max_iter": BCE_MAX_ITER,
            "l2_alpha": BCE_L2_ALPHA,
            "initialization": "zero coefficient vector; intercept 0",
            "fold_local_standard_scaler": True,
            "all_configured_bce_fits_tolerance_converged": bool(
                bce_result["diagnostics"][
                    "converged_by_tolerance"
                ].astype(bool).all()
            ),
        },
        "full_inner_train_scaler": {
            "mean_sha256": score_vector_sha256(
                bce_result["scaler_mean"],
                score_type=(
                    f"inner_scaler_mean.seed_{outer_seed}.fold_{inner_fold}"
                ),
            ),
            "scale_sha256": score_vector_sha256(
                bce_result["scaler_scale"],
                score_type=(
                    f"inner_scaler_scale.seed_{outer_seed}.fold_{inner_fold}"
                ),
            ),
        },
        "relevance_thresholds": [float(value) for value in thresholds],
        "train_relevance_distribution": relevance_distribution(
            train_relevance
        ),
        "validation_relevance_distribution": relevance_distribution(
            validation_relevance
        ),
        "train_relevance_sha256": _integer_vector_sha256(
            train_relevance,
            vector_type=(
                f"inner_train_relevance.seed_{outer_seed}.fold_{inner_fold}"
            ),
        ),
        "validation_relevance_sha256": _integer_vector_sha256(
            validation_relevance,
            vector_type=(
                f"inner_validation_relevance.seed_{outer_seed}.fold_{inner_fold}"
            ),
        ),
        "train_candidate_pool_sha256": candidate_pool_hash(train_pool),
        "validation_candidate_pool_sha256": candidate_pool_hash(validation_pool),
        "candidate_group_definition": candidate_group(
            effective_config.candidate_pool_size
        ),
        "candidate_group_note": (
            "Technical ranking context; not an observed queue or real review list."
        ),
        "outer_test_labels_used": False,
        "outer_test_metrics_used": False,
    }
    _write_json(fold_dir / "metadata.json", metadata)


def _inner_validation_for_seed(
    *,
    dataframe: pd.DataFrame,
    output_root: Path,
    outer_seed: int,
    train_index: np.ndarray,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, pd.DataFrame]:
    from ..comparison_paths.amount_gain import fit_inner_amount_gain_path

    outer_train = dataframe.loc[train_index]
    outer_y = outer_train["Class"].to_numpy(dtype=int)
    _validate_outer_train_identity(
        outer_train,
        outer_y,
        outer_seed=outer_seed,
        effective_config=effective_config,
    )

    splitter = StratifiedKFold(
        n_splits=effective_config.inner_folds,
        shuffle=True,
        random_state=100000 + int(outer_seed),
    )
    bce_diagnostic_frames: list[pd.DataFrame] = []
    bce_loss_frames: list[pd.DataFrame] = []
    fit_rows: list[dict[str, Any]] = []
    ranker_history_rows: list[dict[str, Any]] = []
    candidate_score_frames: list[pd.DataFrame] = []
    pool_summary_rows: list[dict[str, Any]] = []

    splits = splitter.split(np.zeros(len(outer_train)), outer_y)
    for inner_fold, (inner_train_position, validation_position) in enumerate(
        splits,
        start=1,
    ):
        _emit_experiment_log(
            f"[{_utc_now()}] inner BCE seed={outer_seed} fold={inner_fold}",
        )
        inner_train = outer_train.iloc[inner_train_position]
        inner_validation = outer_train.iloc[validation_position]
        bce_result = fit_inner_bce_fold(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            inner_train=inner_train,
            inner_validation=inner_validation,
            effective_config=effective_config,
        )
        bce_diagnostic_frames.append(bce_result["diagnostics"])
        bce_loss_frames.append(bce_result["loss_history"])
        train_scores, validation_scores = _inner_score_frames(
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            inner_train=inner_train,
            inner_validation=inner_validation,
            inner_train_outer_positions=inner_train_position,
            validation_outer_positions=validation_position,
            bce_result=bce_result,
        )
        train_pool = build_candidate_pool(
            bce_result["p_train_oof"],
            inner_train.index,
            candidate_pool_size=effective_config.candidate_pool_size,
        )
        validation_pool = build_candidate_pool(
            bce_result["p_validation"],
            inner_validation.index,
            candidate_pool_size=effective_config.candidate_pool_size,
        )
        if (
            int(train_pool["candidate_flag"].sum())
            != effective_config.candidate_pool_size
            or int(validation_pool["candidate_flag"].sum())
            != effective_config.candidate_pool_size
        ):
            raise RuntimeError(
                "Inner candidate pool has an incorrect configured size."
            )

        y_inner_train = inner_train["Class"].to_numpy(dtype=int)
        amount_inner_train = inner_train["Amount"].to_numpy(dtype=float)
        y_validation = inner_validation["Class"].to_numpy(dtype=int)
        amount_validation = inner_validation["Amount"].to_numpy(dtype=float)
        train_relevance, thresholds = build_amount_gain_relevance_labels(
            y_inner_train,
            amount_inner_train,
        )
        validation_relevance, reused_thresholds = (
            build_amount_gain_relevance_labels(
                y_validation,
                amount_validation,
                fraud_amount_thresholds=thresholds,
            )
        )
        if not np.array_equal(thresholds, reused_thresholds):
            raise RuntimeError("Validation relevance thresholds were not reused.")

        _write_inner_fold_artifacts(
            output_root=output_root,
            outer_seed=outer_seed,
            inner_fold=inner_fold,
            train_scores=train_scores,
            validation_scores=validation_scores,
            train_pool=train_pool,
            validation_pool=validation_pool,
            thresholds=thresholds,
            train_relevance=train_relevance,
            validation_relevance=validation_relevance,
            bce_result=bce_result,
            effective_config=effective_config,
        )
        for role, pool, scores in (
            ("inner_train_oof", train_pool, bce_result["p_train_oof"]),
            (
                "inner_validation_full_train_score",
                validation_pool,
                bce_result["p_validation"],
            ),
        ):
            pool_summary_rows.append(
                {
                    "outer_seed": outer_seed,
                    "inner_fold": inner_fold,
                    "pool_role": role,
                    "rows": len(pool),
                    "candidate_count": int(pool["candidate_flag"].sum()),
                    "candidate_pool_size": (
                        effective_config.candidate_pool_size
                    ),
                    "candidate_pool_sha256": candidate_pool_hash(pool),
                    "score_sha256": score_vector_sha256(
                        scores,
                        score_type=f"{role}.seed_{outer_seed}.fold_{inner_fold}",
                    ),
                }
            )

        baseline_ranking = baseline_bce_ranking(validation_pool)
        _fixed_raw, fixed_ranking = fixed_reference_ranking(
            validation_pool,
            amount_validation,
        )
        baseline_metrics = {
            budget: matched_budget_metrics(
                y_validation,
                amount_validation,
                baseline_ranking,
                budget,
            )
            for budget in effective_config.target_budgets
        }
        fixed_metrics = {
            budget: matched_budget_metrics(
                y_validation,
                amount_validation,
                fixed_ranking,
                budget,
            )
            for budget in effective_config.target_budgets
        }

        train_candidate_positions = candidate_rows_by_bce(train_pool)[
            "original_position"
        ].to_numpy(dtype=int)
        validation_candidate_positions = candidate_rows_by_bce(
            validation_pool
        )["original_position"].to_numpy(dtype=int)
        train_candidate_relevance = train_relevance[train_candidate_positions]
        validation_candidate_relevance = validation_relevance[
            validation_candidate_positions
        ]
        train_features = amount_gain_candidate_features(
            train_pool,
            amount_inner_train,
        )
        validation_features = amount_gain_candidate_features(
            validation_pool,
            amount_validation,
        )

        for target_budget in effective_config.target_budgets:
            for gain_profile in effective_config.enabled_gain_profiles:
                _emit_experiment_log(
                    f"[{_utc_now()}] inner ranker seed={outer_seed} "
                    f"fold={inner_fold} k={target_budget} gain={gain_profile}",
                )
                config, ranker, raw_scores, ranking = fit_inner_amount_gain_path(
                    train_features=train_features,
                    validation_features=validation_features,
                    train_candidate_relevance=train_candidate_relevance,
                    validation_candidate_relevance=validation_candidate_relevance,
                    train_candidate_pool=train_pool,
                    validation_candidate_pool=validation_pool,
                    target_budget=target_budget,
                    gain_profile=gain_profile,
                    random_state=outer_seed,
                    effective_config=effective_config,
                )
                _emit_experiment_status(
                    "INFO",
                    "inner-ranker-complete",
                    seed=outer_seed,
                    inner_fold=inner_fold,
                    budget=target_budget,
                    gain=gain_profile,
                    completed=_inner_ranker_completed(
                        outer_seed,
                        inner_fold,
                        target_budget,
                        gain_profile,
                        effective_config,
                    ),
                    total=(
                        len(effective_config.seeds)
                        * effective_config.inner_folds
                        * len(effective_config.target_budgets)
                        * len(effective_config.enabled_gain_profiles)
                    ),
                )
                metrics = matched_budget_metrics(
                    y_validation,
                    amount_validation,
                    ranking,
                    target_budget,
                )
                bce = baseline_metrics[target_budget]
                fixed = fixed_metrics[target_budget]
                expected_metric = f"ndcg@{target_budget}"
                if (
                    ranker.best_iteration_ is None
                    or ranker.evaluation_metric_ != expected_metric
                    or not ranker.used_early_stopping_
                    or ranker.evals_result_ is None
                    or expected_metric
                    not in ranker.evals_result_["inner_validation"]
                ):
                    raise RuntimeError("Inner early-stopping audit failed.")
                evaluation_history = ranker.evals_result_[
                    "inner_validation"
                ][expected_metric]
                best_value = float(
                    evaluation_history[int(ranker.best_iteration_) - 1]
                )
                if not np.isfinite(best_value):
                    raise RuntimeError("Non-finite best validation nDCG.")

                fit_rows.append(
                    {
                        "outer_seed": outer_seed,
                        "inner_fold": inner_fold,
                        "target_budget": target_budget,
                        "primary_budget": (
                            target_budget in effective_config.primary_budgets
                        ),
                        "gain_profile": gain_profile,
                        "label_gain": json.dumps(list(config.label_gain)),
                        "truncation_level": config.truncation_level,
                        "eval_at": target_budget,
                        "objective": "lambdarank",
                        "n_estimators_safety_cap": (
                            effective_config.ranker_max_estimators
                        ),
                        "learning_rate": RANKER_LEARNING_RATE,
                        "num_leaves": RANKER_NUM_LEAVES,
                        "min_child_samples": RANKER_MIN_CHILD_SAMPLES,
                        "min_child_weight": RANKER_MIN_CHILD_WEIGHT,
                        "reg_lambda": RANKER_REG_LAMBDA,
                        "n_jobs": RANKER_N_JOBS,
                        "early_stopping_rounds": (
                            effective_config.ranker_early_stopping_rounds
                        ),
                        "first_metric_only": True,
                        "evaluation_metric": expected_metric,
                        "best_iteration": int(ranker.best_iteration_),
                        "best_validation_ndcg": best_value,
                        "fit_valid": True,
                        "train_group": effective_config.candidate_pool_size,
                        "validation_group": (
                            effective_config.candidate_pool_size
                        ),
                        "train_candidate_pool_sha256": candidate_pool_hash(
                            train_pool
                        ),
                        "validation_candidate_pool_sha256": candidate_pool_hash(
                            validation_pool
                        ),
                        "raw_ranker_score_sha256": score_vector_sha256(
                            raw_scores,
                            score_type=(
                                f"inner_raw_ranker.seed_{outer_seed}."
                                f"fold_{inner_fold}.k_{target_budget}."
                                f"gain_{gain_profile}"
                            ),
                        ),
                        "relevance_q25": float(thresholds[0]),
                        "relevance_q50": float(thresholds[1]),
                        "relevance_q75": float(thresholds[2]),
                        "train_relevance_distribution": json.dumps(
                            relevance_distribution(train_relevance),
                            sort_keys=True,
                        ),
                        "validation_relevance_distribution": json.dumps(
                            relevance_distribution(validation_relevance),
                            sort_keys=True,
                        ),
                        **metrics,
                        **{
                            f"bce_{key}": value
                            for key, value in bce.items()
                            if key != "target_budget"
                        },
                        **{
                            f"fixed_reference_{key}": value
                            for key, value in fixed.items()
                            if key != "target_budget"
                        },
                    }
                )
                for iteration, value in enumerate(
                    evaluation_history,
                    start=1,
                ):
                    ranker_history_rows.append(
                        {
                            "outer_seed": outer_seed,
                            "inner_fold": inner_fold,
                            "target_budget": target_budget,
                            "gain_profile": gain_profile,
                            "iteration": iteration,
                            "metric": expected_metric,
                            "value": float(value),
                            "is_best_iteration": (
                                iteration == int(ranker.best_iteration_)
                            ),
                        }
                    )
                candidate_rows = candidate_rows_by_bce(validation_pool)
                candidate_score_frames.append(
                    pd.DataFrame(
                        {
                            "outer_seed": outer_seed,
                            "inner_fold": inner_fold,
                            "target_budget": target_budget,
                            "gain_profile": gain_profile,
                            "row_index": candidate_rows["row_index"].to_numpy(
                                dtype=int
                            ),
                            "candidate_rank_by_bce": candidate_rows[
                                "candidate_rank_by_bce"
                            ].to_numpy(dtype=int),
                            "raw_ranker_score": raw_scores,
                            "candidate_rank_by_ranker": ranking.loc[
                                validation_candidate_positions,
                                "candidate_rank_by_ranker",
                            ].to_numpy(dtype=int),
                        }
                    )
                )

    seed_dir = output_root / "inner_validation" / f"seed_{outer_seed}"
    fit_frame = pd.DataFrame(fit_rows)
    bce_diagnostics = pd.concat(bce_diagnostic_frames, ignore_index=True)
    bce_losses = pd.concat(bce_loss_frames, ignore_index=True)
    ranker_history = pd.DataFrame(ranker_history_rows)
    candidate_scores = pd.concat(candidate_score_frames, ignore_index=True)
    pool_summary = pd.DataFrame(pool_summary_rows)
    expected_ranker_fits = (
        effective_config.inner_folds
        * len(effective_config.target_budgets)
        * len(effective_config.enabled_gain_profiles)
    )
    expected_bce_fits = effective_config.inner_folds * (
        effective_config.bce_oof_folds + 1
    )
    if len(fit_frame) != expected_ranker_fits:
        raise RuntimeError("Unexpected inner ranker fit count.")
    if len(bce_diagnostics) != expected_bce_fits:
        raise RuntimeError("Unexpected inner BCE fit count.")
    if not bce_diagnostics["converged_by_tolerance"].astype(bool).all():
        raise RuntimeError("Not all inner BCE fits converged by tolerance.")
    _write_csv(seed_dir / "bce_fit_diagnostics.csv", bce_diagnostics)
    _write_csv(
        seed_dir / "bce_loss_history.csv.gz",
        bce_losses,
        compressed=True,
    )
    _write_csv(seed_dir / "amount_gain_ranker_fits.csv", fit_frame)
    _write_csv(
        seed_dir / "ranker_evaluation_history.csv.gz",
        ranker_history,
        compressed=True,
    )
    _write_parquet(
        seed_dir / "validation_candidate_ranker_scores.parquet",
        candidate_scores,
    )
    _write_csv(seed_dir / "candidate_pool_summary.csv", pool_summary)
    return {
        "fits": fit_frame,
        "bce_diagnostics": bce_diagnostics,
        "bce_losses": bce_losses,
        "ranker_history": ranker_history,
        "candidate_scores": candidate_scores,
        "pool_summary": pool_summary,
    }


def run_inner_phase(
    args: argparse.Namespace,
    effective_config: EffectiveExperimentConfig,
) -> None:
    output_root = Path(args.output_dir).resolve()
    _ensure_run_directories(output_root)
    preflight = _load_preflight(output_root, effective_config)
    selection_manifest_path = (
        output_root / "selection_freeze" / "selection_manifest.json"
    )
    if selection_manifest_path.exists():
        raise FileExistsError("A selection freeze already exists.")
    dataframe, data_metadata = load_experiment_data(
        Path(args.data_path),
        effective_config,
    )

    all_fit_frames: list[pd.DataFrame] = []
    all_bce_frames: list[pd.DataFrame] = []
    all_pool_frames: list[pd.DataFrame] = []
    for outer_seed in effective_config.seeds:
        train_index, _test_index = verify_outer_split_without_test_labels(
            dataframe,
            output_root,
            outer_seed,
            effective_config,
        )
        result = _inner_validation_for_seed(
            dataframe=dataframe,
            output_root=output_root,
            outer_seed=outer_seed,
            train_index=train_index,
            effective_config=effective_config,
        )
        all_fit_frames.append(result["fits"])
        all_bce_frames.append(result["bce_diagnostics"])
        all_pool_frames.append(result["pool_summary"])
    fits = pd.concat(all_fit_frames, ignore_index=True)
    bce_diagnostics = pd.concat(all_bce_frames, ignore_index=True)
    pool_summary = pd.concat(all_pool_frames, ignore_index=True)
    expected_ranker_fits = (
        len(effective_config.seeds)
        * effective_config.inner_folds
        * len(effective_config.target_budgets)
        * len(effective_config.enabled_gain_profiles)
    )
    expected_bce_fits = (
        len(effective_config.seeds)
        * effective_config.inner_folds
        * (effective_config.bce_oof_folds + 1)
    )
    if len(fits) != expected_ranker_fits:
        raise RuntimeError(
            "Expected "
            f"{expected_ranker_fits} inner ranker fits, found {len(fits)}."
        )
    if len(bce_diagnostics) != expected_bce_fits:
        raise RuntimeError(
            "Expected "
            f"{expected_bce_fits} inner BCE fits, "
            f"found {len(bce_diagnostics)}."
        )
    if not bce_diagnostics["converged_by_tolerance"].astype(bool).all():
        raise RuntimeError("At least one inner BCE fit failed tolerance convergence.")

    _write_csv(
        output_root
        / "inner_validation"
        / "inner_amount_gain_fit_results.csv",
        fits,
    )
    _write_csv(
        output_root / "inner_validation" / "bce_fit_diagnostics_all.csv",
        bce_diagnostics,
    )
    _write_csv(
        output_root
        / "inner_validation"
        / "candidate_pool_summary_all.csv",
        pool_summary,
    )
    inner_comparison = _inner_matched_comparison(fits)
    _write_csv(
        output_root
        / "inner_validation"
        / "inner_matched_comparison_long.csv",
        inner_comparison,
    )

    selected_rows: list[dict[str, Any]] = []
    gain_summary_frames: list[pd.DataFrame] = []
    selected_nonselected_frames: list[pd.DataFrame] = []
    config_paths: list[Path] = []
    for outer_seed in effective_config.seeds:
        for target_budget in effective_config.target_budgets:
            subset = fits.loc[
                (fits["outer_seed"].astype(int) == outer_seed)
                & (fits["target_budget"].astype(int) == target_budget)
            ].copy()
            selection_input = subset[
                [
                    "gain_profile",
                    "inner_fold",
                    "prevented_loss_ratio_at_k",
                    "frauds_at_k",
                    "bce_prevented_loss_ratio_at_k",
                    "bce_frauds_at_k",
                    "amount_ndcg_at_k",
                    "cutoff_tie_size",
                    "best_iteration",
                    "fit_valid",
                ]
            ]
            selection, enriched, gain_summaries = (
                select_gain_from_inner_results(
                    selection_input,
                    effective_config,
                )
            )
            enriched.insert(0, "outer_seed", outer_seed)
            enriched.insert(1, "target_budget", target_budget)
            gain_summaries.insert(0, "outer_seed", outer_seed)
            gain_summaries.insert(1, "target_budget", target_budget)
            gain_summaries["selected_gain"] = selection["selected_gain"]
            gain_summaries["selection_status"] = selection["selection_status"]
            gain_summaries["selected"] = (
                gain_summaries["gain_profile"]
                == selection["selected_gain"]
            )
            gain_summary_frames.append(gain_summaries.copy())
            selected_nonselected_frames.append(gain_summaries.copy())

            payload = _selection_config_payload(
                output_root=output_root,
                data_metadata=data_metadata,
                outer_seed=outer_seed,
                target_budget=target_budget,
                selection=selection,
                gain_summaries=gain_summaries,
                effective_config=effective_config,
            )
            config_path = (
                output_root
                / "selection_freeze"
                / f"selected_config_seed_{outer_seed}_k_{target_budget}.json"
            )
            _write_json(config_path, payload)
            _emit_experiment_status(
                "INFO",
                "selection-freeze-config-complete",
                seed=outer_seed,
                budget=target_budget,
                completed=_selection_freeze_completed(
                    outer_seed,
                    target_budget,
                    effective_config,
                ),
                total=(
                    len(effective_config.seeds)
                    * len(effective_config.target_budgets)
                ),
            )
            config_paths.append(config_path)
            selected_rows.append(
                {
                    "outer_seed": outer_seed,
                    "target_budget": target_budget,
                    "primary_budget": (
                        target_budget in effective_config.primary_budgets
                    ),
                    "selected_gain": selection["selected_gain"],
                    "selection_status": selection["selection_status"],
                    "label_gain": json.dumps(
                        list(GAIN_PROFILES[str(selection["selected_gain"])])
                    ),
                    "best_iteration_fold_1": selection[
                        "best_iteration_fold_1"
                    ],
                    "best_iteration_fold_2": selection[
                        "best_iteration_fold_2"
                    ],
                    "best_iteration_fold_3": selection[
                        "best_iteration_fold_3"
                    ],
                    "final_n_estimators": selection["final_n_estimators"],
                    "truncation": truncation_for_budget(
                        target_budget,
                        effective_config,
                    ),
                    "eval_at": target_budget,
                    "mean_fraud_retention": selection[
                        "mean_fraud_retention"
                    ],
                    "min_fraud_retention": selection[
                        "min_fraud_retention"
                    ],
                    "mean_delta_plr": selection["mean_delta_plr"],
                    "positive_delta_plr_fold_count": selection[
                        "positive_delta_plr_fold_count"
                    ],
                    "mean_amount_ndcg_at_k": selection[
                        "mean_amount_ndcg_at_k"
                    ],
                    "mean_cutoff_tie_size": selection[
                        "mean_cutoff_tie_size"
                    ],
                    "config_hash": payload["config_hash"],
                    "data_sha256": payload["data_sha256"],
                    "split_sha256": payload["split_sha256"],
                }
            )

    selected = pd.DataFrame(selected_rows).sort_values(
        ["outer_seed", "target_budget"],
        kind="mergesort",
    )
    gain_comparison = pd.concat(gain_summary_frames, ignore_index=True)
    selected_nonselected = pd.concat(
        selected_nonselected_frames,
        ignore_index=True,
    )
    expected_selected = len(effective_config.seeds) * len(
        effective_config.target_budgets
    )
    if len(selected) != expected_selected or selected.duplicated(
        ["outer_seed", "target_budget"]
    ).any():
        raise RuntimeError(
            "Selection did not produce the configured unique configurations."
        )
    distribution = (
        selected.groupby(
            ["target_budget", "selected_gain", "selection_status"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "seed_count"})
    )
    distribution["primary_budget"] = distribution["target_budget"].isin(
        effective_config.primary_budgets
    )

    selected_path = (
        output_root
        / "selection_freeze"
        / "selected_gain_by_seed_budget.csv"
    )
    distribution_path = (
        output_root
        / "selection_freeze"
        / "selected_gain_distribution_by_budget.csv"
    )
    _write_csv(selected_path, selected)
    _write_csv(distribution_path, distribution)
    _write_csv(
        output_root / "comparison" / "inner_gain_comparison.csv",
        gain_comparison,
    )
    _write_csv(
        output_root
        / "comparison"
        / "selected_vs_nonselected_gain_inner_only.csv",
        selected_nonselected.assign(inner_only=True),
    )
    selection_status_summary = (
        selected.groupby(
            ["target_budget", "selection_status"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "seed_count"})
    )
    _write_csv(
        output_root / "comparison" / "selection_status_summary.csv",
        selection_status_summary,
    )

    freeze_time = _utc_now()
    freeze_artifacts = [
        *config_paths,
        selected_path,
        distribution_path,
    ]
    manifest = {
        "schema": "ranker_gain_validation.selection_manifest.v1",
        "freeze_timestamp_utc": freeze_time,
        "outer_test_selection_locked": True,
        "outer_test_labels_used_for_selection": False,
        "outer_test_metrics_used_for_selection": False,
        "preflight_status": preflight["status"],
        "ranker_scope": RANKER_SCOPE,
        "outer_seeds": list(effective_config.seeds),
        "target_budgets": list(effective_config.target_budgets),
        "gain_candidates": list(effective_config.enabled_gain_profiles),
        "candidate_pool_size": effective_config.candidate_pool_size,
        "selected_config_count": len(config_paths),
        "selected_configs": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in config_paths
        ],
        "aggregate_selection_artifacts": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in (selected_path, distribution_path)
        ],
        "data_sha256": data_metadata["deduplicated_dataframe_sha256"],
        "inner_amount_gain_ranker_fit_count": len(fits),
        "inner_bce_fit_count": len(bce_diagnostics),
        "inner_bce_tolerance_converged_count": int(
            bce_diagnostics["converged_by_tolerance"].astype(bool).sum()
        ),
        "selection_rule_locked": True,
    }
    manifest_path = (
        output_root
        / "selection_freeze"
        / "selection_manifest.json"
    )
    _write_json(manifest_path, manifest)
    freeze_artifacts.append(manifest_path)
    _write_checksum_manifest(
        output_root,
        freeze_artifacts,
        output_root / "selection_freeze" / "checksums.sha256",
    )
    inner_manifest_path = (
        output_root
        / "inner_validation"
        / "inner_validation_manifest.json"
    )
    _write_json(
        inner_manifest_path,
        {
            "status": "PASS",
            "completed_at_utc": freeze_time,
            "outer_seed_count": len(effective_config.seeds),
            "inner_fold_count_per_seed": effective_config.inner_folds,
            "inner_bce_fit_count": len(bce_diagnostics),
            "inner_bce_tolerance_converged_count": int(
                bce_diagnostics[
                    "converged_by_tolerance"
                ].astype(bool).sum()
            ),
            "inner_amount_gain_ranker_fit_count": len(fits),
            "expected_inner_amount_gain_ranker_fit_count": (
                expected_ranker_fits
            ),
            "outer_test_selection_locked": True,
            "outer_test_labels_used": False,
            "outer_test_metrics_used": False,
        },
    )
    _emit_experiment_status(
        "PASS",
        "artifact-manifest-complete",
        kind="selection",
        manifest=manifest_path,
    )
    _emit_experiment_status(
        "PASS",
        "selection-freeze-complete",
        completed=expected_selected,
        total=expected_selected,
        manifest=manifest_path,
    )
