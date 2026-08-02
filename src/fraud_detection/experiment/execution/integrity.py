"""Qa boundary for the frozen experiment runner."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.artifacts import (
    _ensure_run_directories,
    _verify_checksum_manifest,
    _write_checksum_manifest,
    _write_json,
)

from ..config import (
    METHOD_AMOUNT_GAIN,
    METHOD_BASELINE,
    METHOD_FAMILIES,
    EffectiveExperimentConfig,
    _emit_experiment_status,
    _status_utc_now,
    _utc_now,
)
from ..preparation.data import _expected_outer_split_rows
from ..prioritization.composition import validate_full_ranking
from ..prioritization.selection import _load_and_verify_freeze


def _expected_outer_test_rows(
    effective_config: EffectiveExperimentConfig,
) -> int:
    return _expected_outer_split_rows(effective_config)[1]


def _assert_unique_keys(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    artifact: str,
) -> None:
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{artifact} is missing key columns {missing}.")
    if frame.duplicated(keys).any():
        raise RuntimeError(f"{artifact} contains duplicate keys {keys}.")


def _assert_finite_numeric(
    frame: pd.DataFrame,
    *,
    artifact: str,
    allowed_nan_columns: set[str] | None = None,
) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    allowed = allowed_nan_columns or set()
    required = numeric.drop(
        columns=[column for column in numeric.columns if column in allowed]
    )
    if not required.empty and not np.isfinite(
        required.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"{artifact} contains non-finite numeric values.")
    for column in allowed & set(numeric.columns):
        values = numeric[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise RuntimeError(f"{artifact} contains infinite numeric values.")


def _validate_selected_gain_numeric_contract(
    frame: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> None:
    artifact = "selected_gain_by_seed_budget.csv"
    fold_prefix = "best_iteration_fold_"
    retained_fold_columns = tuple(f"{fold_prefix}{index}" for index in range(1, 4))
    active_fold_columns = tuple(
        f"{fold_prefix}{index}" for index in range(1, effective_config.inner_folds + 1)
    )
    actual_fold_columns = tuple(
        str(column) for column in frame.columns if str(column).startswith(fold_prefix)
    )

    missing_active_columns = [
        column for column in active_fold_columns if column not in actual_fold_columns
    ]
    if missing_active_columns:
        raise RuntimeError(
            f"{artifact} is missing active fold column {missing_active_columns[0]}."
        )
    if actual_fold_columns != retained_fold_columns:
        raise RuntimeError(f"{artifact} has malformed fold-column numbering.")

    for column in active_fold_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"{artifact} has non-finite active fold values in {column}."
            )
        if (
            (values <= 0).any()
            or (values != np.floor(values)).any()
            or (values > effective_config.ranker_max_estimators).any()
        ):
            raise RuntimeError(
                f"{artifact} has invalid active fold iterations in {column}."
            )

    for column in retained_fold_columns[effective_config.inner_folds :]:
        if frame[column].notna().any():
            raise RuntimeError(
                f"{artifact} has unexpectedly populated inactive fold column {column}."
            )

    _assert_finite_numeric(
        frame.drop(columns=list(retained_fold_columns)), artifact=artifact
    )


def validate_completed_run(
    *,
    output_root: Path,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    freeze_manifest, configs = _load_and_verify_freeze(
        output_root,
        effective_config,
    )
    final_manifest = json.loads(
        (
            output_root / "final_outer_run" / "final_outer_manifest.json"
        ).read_text(encoding="utf-8-sig")
    )
    if final_manifest.get("status") != "PASS":
        raise RuntimeError("Final outer manifest is not PASS.")
    expected_seed_budget_count = len(effective_config.seeds) * len(
        effective_config.target_budgets
    )
    expected_manifest_counts = {
        "outer_seed_count": len(effective_config.seeds),
        "target_budget_count": len(effective_config.target_budgets),
        "selected_amount_gain_model_count": expected_seed_budget_count,
        "p_only_model_count": expected_seed_budget_count,
        "fixed_reference_path_count": expected_seed_budget_count,
        "baseline_matched_path_count": expected_seed_budget_count,
        "ranking_dump_count": len(effective_config.seeds),
    }
    if any(
        final_manifest.get(name) != value
        for name, value in expected_manifest_counts.items()
    ):
        raise RuntimeError("Final outer manifest dimensions are incorrect.")

    selected = pd.read_csv(
        output_root / "tables" / "selected_gain_by_seed_budget.csv"
    )
    all_budget = pd.read_csv(
        output_root / "tables" / "all_budget_matched_results.csv"
    )
    central = pd.read_csv(
        output_root / "tables" / "central_budget_results.csv"
    )
    models = pd.read_csv(
        output_root / "tables" / "selected_model_parameters.csv"
    )
    seed_metrics = pd.read_csv(
        output_root
        / "comparison"
        / "matched_budget_comparison_long.csv"
    )
    inner_nonselected = pd.read_csv(
        output_root
        / "comparison"
        / "selected_vs_nonselected_gain_inner_only.csv"
    )
    if (
        len(selected) != expected_seed_budget_count
        or set(selected["selected_gain"].astype(str))
        - set(effective_config.enabled_gain_profiles)
        or set(selected["target_budget"].astype(int))
        != set(effective_config.target_budgets)
        or set(selected["outer_seed"].astype(int))
        != set(effective_config.seeds)
    ):
        raise RuntimeError("Selected configuration table failed coverage checks.")
    if not inner_nonselected["inner_only"].astype(bool).all():
        raise RuntimeError("Nonselected gain comparison is not inner-only.")
    if set(central["target_budget"].astype(int)) != set(
        effective_config.primary_budgets
    ):
        raise RuntimeError("central_budget_results.csv has incorrect budgets.")
    if set(all_budget["target_budget"].astype(int)) != set(
        effective_config.target_budgets
    ):
        raise RuntimeError("all_budget_matched_results.csv is incomplete.")
    if set(all_budget["method_family"].astype(str)) != set(METHOD_FAMILIES):
        raise RuntimeError("All-budget table has incorrect central paths.")
    if (
        len(models.loc[models["model_type"] == "amount_gain"])
        != expected_seed_budget_count
        or len(models.loc[models["model_type"] == "p_only"])
        != expected_seed_budget_count
    ):
        raise RuntimeError("Selected final model counts are incorrect.")
    if not models["scores_finite"].astype(bool).all():
        raise RuntimeError("A selected final model has non-finite scores.")

    amount_models = models.loc[
        models["model_type"] == "amount_gain"
    ].sort_values(["seed", "target_budget"])
    p_only_models = models.loc[
        models["model_type"] == "p_only"
    ].sort_values(["seed", "target_budget"])
    shared_columns = [
        "seed",
        "target_budget",
        "selected_gain",
        "selection_status",
        "label_gain",
        "truncation_level",
        "eval_at",
        "configured_n_estimators",
        "trained_n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "min_child_weight",
        "reg_lambda",
        "n_jobs",
        "train_candidate_pool_sha256",
        "test_candidate_pool_sha256",
        "config_hash",
    ]
    if not np.array_equal(
        amount_models[shared_columns].to_numpy(),
        p_only_models[shared_columns].to_numpy(),
    ):
        raise RuntimeError("p-only configuration differs from Amount-Gain.")

    for row in selected.itertuples(index=False):
        expected = configs[(int(row.outer_seed), int(row.target_budget))]
        if (
            str(row.selected_gain) != expected["selected_gain"]
            or int(row.truncation) != int(row.target_budget) + 3
            or int(row.eval_at) != int(row.target_budget)
            or int(row.final_n_estimators)
            != int(expected["final_n_estimators"])
        ):
            raise RuntimeError("Selected table differs from frozen config.")

    ranking_path_count = 0
    for seed in effective_config.seeds:
        seed_dir = output_root / "final_outer_run" / f"seed_{seed}"
        bce_diagnostics = pd.read_csv(seed_dir / "bce_fit_diagnostics.csv")
        if (
            len(bce_diagnostics) != effective_config.bce_oof_folds + 1
            or not bce_diagnostics["converged_by_tolerance"].astype(bool).all()
        ):
            raise RuntimeError(
                f"Recreated outer BCE convergence QA failed for seed {seed}."
            )
        train_pool = pd.read_parquet(seed_dir / "candidate_pool_train.parquet")
        test_pool = pd.read_parquet(seed_dir / "candidate_pool_test.parquet")
        if (
            int(train_pool["candidate_flag"].sum())
            != effective_config.candidate_pool_size
            or int(test_pool["candidate_flag"].sum())
            != effective_config.candidate_pool_size
            or not train_pool.loc[train_pool["candidate_flag"], "row_index"].is_unique
            or not test_pool.loc[test_pool["candidate_flag"], "row_index"].is_unique
        ):
            raise RuntimeError(f"Candidate-pool QA failed for seed {seed}.")
        dump = pd.read_parquet(seed_dir / "ranking_dump.parquet")
        expected_test_rows = _expected_outer_test_rows(effective_config)
        if (
            len(dump)
            != len(effective_config.target_budgets)
            * len(METHOD_FAMILIES)
            * expected_test_rows
        ):
            raise RuntimeError(f"Ranking dump row count failed for seed {seed}.")
        for (budget, path), group in dump.groupby(
            ["target_budget", "score_path"],
            sort=False,
        ):
            ranking_path_count += 1
            group = group.reset_index(drop=True)
            if len(group) != expected_test_rows:
                raise RuntimeError("Incomplete ranking path.")
            validate_full_ranking(group)
            flags = group["candidate_flag"].to_numpy(dtype=bool)
            raw = group["raw_ranker_score"].to_numpy(dtype=float)
            if str(group["method_family"].iloc[0]) == METHOD_BASELINE:
                if not np.isfinite(raw).all():
                    raise RuntimeError("BCE raw score vector is non-finite.")
            elif not np.isfinite(raw[flags]).all():
                raise RuntimeError("Candidate raw ranker scores are non-finite.")
            if int(budget) != int(group["truncation_level"].iloc[0]) - 3:
                raise RuntimeError("Ranking truncation does not equal k+3.")
            if group["selected_gain"].nunique() != 1:
                raise RuntimeError("Ranking path contains multiple selected gains.")
        fixed = pd.read_csv(seed_dir / "fixed_reference_budget_paths.csv")
        pool_hash = test_pool["candidate_pool_sha256"].astype(str).iloc[0]
        if not (
            fixed["formula"].astype(str).eq("p_fraud * log1p(Amount)").all()
            and fixed["test_candidate_pool_sha256"].astype(str).eq(pool_hash).all()
        ):
            raise RuntimeError("Fixed reference pool/formula QA failed.")
    expected_ranking_paths = (
        len(effective_config.seeds)
        * len(effective_config.target_budgets)
        * len(METHOD_FAMILIES)
    )
    if ranking_path_count != expected_ranking_paths:
        raise RuntimeError("Ranking path count is incomplete.")

    expected_budget_paths = {
        (budget, method_family)
        for budget in effective_config.target_budgets
        for method_family in METHOD_FAMILIES
    }
    expected_primary_paths = {
        (budget, method_family)
        for budget in effective_config.primary_budgets
        for method_family in METHOD_FAMILIES
    }
    expected_seed_paths = {
        (seed, budget, method_family)
        for seed in effective_config.seeds
        for budget in effective_config.target_budgets
        for method_family in METHOD_FAMILIES
    }
    expected_model_keys = {
        (seed, budget, model_type)
        for seed in effective_config.seeds
        for budget in effective_config.target_budgets
        for model_type in ("amount_gain", "p_only")
    }
    if (
        set(
            all_budget[
                ["target_budget", "method_family"]
            ].itertuples(index=False, name=None)
        )
        != expected_budget_paths
        or set(
            central[
                ["target_budget", "method_family"]
            ].itertuples(index=False, name=None)
        )
        != expected_primary_paths
        or set(
            seed_metrics[
                ["seed", "target_budget", "method_family"]
            ].itertuples(index=False, name=None)
        )
        != expected_seed_paths
        or set(
            models[
                ["seed", "target_budget", "model_type"]
            ].itertuples(index=False, name=None)
        )
        != expected_model_keys
    ):
        raise RuntimeError("Completed result grids are incomplete.")

    standard_deviation_columns = {
        column for column in all_budget.columns if column.endswith("_std")
    }
    if len(effective_config.seeds) == 1:
        if standard_deviation_columns and not all_budget[
            sorted(standard_deviation_columns)
        ].isna().all().all():
            raise RuntimeError("One-seed sample standard deviations must be NaN.")
    elif standard_deviation_columns and all_budget[
        sorted(standard_deviation_columns)
    ].isna().any().any():
        raise RuntimeError("Multi-seed sample standard deviations are missing.")

    key_specs = {
        "central_budget_results.csv": (
            central,
            ["target_budget", "method_family"],
        ),
        "all_budget_matched_results.csv": (
            all_budget,
            ["target_budget", "method_family"],
        ),
        "selected_gain_by_seed_budget.csv": (
            selected,
            ["outer_seed", "target_budget"],
        ),
        "selected_model_parameters.csv": (
            models,
            ["seed", "target_budget", "model_type"],
        ),
        "matched_budget_comparison_long.csv": (
            seed_metrics,
            ["seed", "target_budget", "method_family"],
        ),
    }
    for artifact, (frame, keys) in key_specs.items():
        _assert_unique_keys(frame, keys, artifact=artifact)
        if artifact == "selected_gain_by_seed_budget.csv":
            _validate_selected_gain_numeric_contract(frame, effective_config)
            continue
        allowed_nan_columns = (
            standard_deviation_columns
            if len(effective_config.seeds) == 1
            and artifact
            in {
                "central_budget_results.csv",
                "all_budget_matched_results.csv",
            }
            else set()
        )
        _assert_finite_numeric(
            frame,
            artifact=artifact,
            allowed_nan_columns=allowed_nan_columns,
        )

    central_final = seed_metrics.loc[
        seed_metrics["method_family"] == METHOD_AMOUNT_GAIN
    ]
    expected_selected = selected.rename(columns={"outer_seed": "seed"})[
        ["seed", "target_budget", "selected_gain"]
    ]
    checked = central_final.merge(
        expected_selected,
        on=["seed", "target_budget"],
        suffixes=("_outer", "_frozen"),
        validate="one_to_one",
    )
    if not (
        checked["selected_gain_outer"].astype(str)
        == checked["selected_gain_frozen"].astype(str)
    ).all():
        raise RuntimeError("Final outer results contain a nonselected gain.")

    return {
        "status": "PASS",
        "validated_at_utc": _utc_now(),
        "selection_config_count": len(configs),
        "amount_gain_model_count": len(amount_models),
        "p_only_model_count": len(p_only_models),
        "ranking_path_count": ranking_path_count,
        "candidate_pool_count": len(effective_config.seeds) * 2,
        "outer_test_selection_locked": freeze_manifest[
            "outer_test_selection_locked"
        ],
    }


def write_failure_record(
    *,
    output_root: Path,
    phase: str,
    error: BaseException,
) -> None:
    status_by_phase = {
        "inner": "PARTIAL – INNER VALIDATION FAILED",
        "final": "PARTIAL – FINAL OUTER RUN FAILED",
        "qa": "PARTIAL – FINAL OUTER RUN FAILED",
        "all": "PARTIAL – FINAL OUTER RUN FAILED",
    }
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / "logs" / f"failure_{phase}_{timestamp}.json"
    if not path.exists():
        _write_json(
            path,
            {
                "status": status_by_phase[phase],
                "phase": phase,
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "follow_on_phase_started": False,
            },
        )


def run_validation_phase(
    args: argparse.Namespace,
    effective_config: EffectiveExperimentConfig,
) -> None:
    qa_started = time.perf_counter()
    _emit_experiment_status(
        "INFO",
        "qa-start",
        utc=_status_utc_now(),
    )
    output_root = Path(args.output_dir).resolve()
    _ensure_run_directories(output_root)
    checksum_path = output_root / "comparison" / "checksums.sha256"
    qa = validate_completed_run(
        output_root=output_root,
        effective_config=effective_config,
    )
    final_qa_path = output_root / "comparison" / "final_qa.json"
    _write_json(final_qa_path, qa)

    artifacts = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path != checksum_path
    ]
    _write_checksum_manifest(
        output_root,
        artifacts,
        checksum_path,
    )
    _verify_checksum_manifest(output_root, checksum_path)
    _emit_experiment_status(
        "PASS",
        "artifact-manifest-complete",
        kind="root",
        manifest=checksum_path,
    )
    _emit_experiment_status(
        "PASS",
        "qa-complete",
        status=qa["status"],
        elapsed_seconds=time.perf_counter() - qa_started,
    )
