"""Deterministic synthetic engineering data for the smoke profile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.errors import ProductError

from ..config import BCE_FEATURES, EffectiveExperimentConfig

_GENERATOR_SCHEMA = "fraud_detection.synthetic_engineering.v1"
_EVIDENCE_BOUNDARY = (
    "Synthetic engineering data; not thesis evidence; not comparable to "
    "canonical empirical results."
)


def _synthetic_error(
    code: str,
    summary: str,
    details: dict[str, object] | None = None,
) -> ProductError:
    return ProductError(
        code,
        summary,
        ("Use the unchanged smoke-synthetic profile configuration.",),
        details or {},
    )


def _validated_generation_settings(
    effective_config: EffectiveExperimentConfig,
) -> tuple[int, int]:
    if (
        not isinstance(effective_config, EffectiveExperimentConfig)
        or effective_config.data_source_kind != "synthetic"
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-CONFIG",
            "Synthetic generation requires a synthetic effective configuration.",
        )
    row_target = effective_config.synthetic_row_target
    if (
        isinstance(row_target, bool)
        or not isinstance(row_target, (int, np.integer))
        or int(row_target) <= 0
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-ROW-TARGET",
            "The synthetic row target is missing or invalid.",
            {"synthetic_row_target": row_target},
        )
    generation_seed = effective_config.synthetic_generation_seed
    if isinstance(generation_seed, bool) or not isinstance(
        generation_seed,
        (int, np.integer),
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-SEED",
            "The synthetic generation seed is missing or invalid.",
            {"synthetic_generation_seed": generation_seed},
        )

    row_count = int(row_target)
    outer_test_rows = (row_count + 4) // 5
    outer_train_rows = row_count - outer_test_rows
    smallest_inner_partition = outer_train_rows // effective_config.inner_folds
    if min(outer_test_rows, smallest_inner_partition) < (
        effective_config.candidate_pool_size
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-ROW-TARGET",
            "The synthetic row target cannot support the configured candidate pool.",
            {
                "synthetic_row_target": row_count,
                "candidate_pool_size": effective_config.candidate_pool_size,
            },
        )
    return row_count, int(generation_seed)


def _validate_synthetic_frame(
    dataframe: pd.DataFrame,
    effective_config: EffectiveExperimentConfig,
) -> None:
    expected_columns = ("Time", *BCE_FEATURES, "Amount", "Class")
    if tuple(dataframe.columns) != expected_columns:
        raise _synthetic_error(
            "FD-SYNTHETIC-SCHEMA",
            "The generated synthetic schema is incomplete or incorrectly ordered.",
            {"columns": [str(column) for column in dataframe.columns]},
        )
    if len(dataframe) != effective_config.synthetic_row_target:
        raise _synthetic_error(
            "FD-SYNTHETIC-ROW-COUNT",
            "The generated synthetic row count does not match its target.",
            {
                "expected": effective_config.synthetic_row_target,
                "actual": len(dataframe),
            },
        )
    try:
        numeric_values = dataframe.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise _synthetic_error(
            "FD-SYNTHETIC-VALUES",
            "The generated synthetic data contains non-numeric values.",
        ) from error
    if dataframe.isna().any().any() or not np.isfinite(numeric_values).all():
        raise _synthetic_error(
            "FD-SYNTHETIC-VALUES",
            "The generated synthetic data contains missing or non-finite values.",
        )
    if (
        not pd.api.types.is_integer_dtype(dataframe["Class"])
        or set(dataframe["Class"].unique()) != {0, 1}
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-CLASS",
            "The generated synthetic Class column is not binary integer data.",
        )
    if (dataframe[["Time", "Amount"]] < 0.0).any().any():
        raise _synthetic_error(
            "FD-SYNTHETIC-VALUES",
            "The generated synthetic Time or Amount contains negative values.",
        )

    counts = dataframe["Class"].value_counts().to_dict()
    minimum_class_count = max(
        10,
        2 * effective_config.inner_folds * effective_config.bce_oof_folds,
    )
    if (
        int(counts.get(1, 0)) >= int(counts.get(0, 0))
        or min(int(counts.get(0, 0)), int(counts.get(1, 0)))
        < minimum_class_count
    ):
        raise _synthetic_error(
            "FD-SYNTHETIC-CLASS-SUPPORT",
            "The generated class counts cannot support configured stratified folds.",
            {
                "legitimate_count": int(counts.get(0, 0)),
                "fraud_count": int(counts.get(1, 0)),
                "minimum_class_count": minimum_class_count,
            },
        )
    if dataframe.duplicated().any():
        raise _synthetic_error(
            "FD-SYNTHETIC-DUPLICATES",
            "The generated synthetic data contains complete duplicate rows.",
        )


def generate_synthetic_data(
    effective_config: EffectiveExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate one bounded, deterministic engineering dataset in memory."""

    row_count, generation_seed = _validated_generation_settings(effective_config)
    random = np.random.default_rng(generation_seed)

    fraud_count = max(40, row_count // 50)
    fraud_positions = random.choice(
        row_count,
        size=fraud_count,
        replace=False,
    )
    target = np.zeros(row_count, dtype=np.int64)
    target[fraud_positions] = 1

    feature_values = random.normal(0.0, 1.0, size=(row_count, len(BCE_FEATURES)))
    feature_values[fraud_positions, :4] += random.normal(
        1.25,
        0.45,
        size=(fraud_count, 4),
    )
    feature_values[fraud_positions, 4:8] -= random.normal(
        0.75,
        0.35,
        size=(fraud_count, 4),
    )
    feature_values = np.clip(feature_values, -8.0, 8.0)

    amount = random.lognormal(mean=3.2, sigma=1.1, size=row_count)
    amount_levels = np.array([12.0, 55.0, 260.0, 1_400.0])
    fraud_levels = np.arange(fraud_count) % len(amount_levels)
    random.shuffle(fraud_levels)
    amount[fraud_positions] = random.lognormal(
        mean=np.log(amount_levels[fraud_levels]),
        sigma=0.22,
    )
    amount = np.clip(amount, 0.0, 100_000.0)
    time = np.arange(row_count, dtype=float) * 30.0 + random.uniform(
        0.0,
        15.0,
        size=row_count,
    )

    dataframe = pd.DataFrame(
        {
            "Time": time,
            **{
                feature: feature_values[:, position]
                for position, feature in enumerate(BCE_FEATURES)
            },
            "Amount": amount,
            "Class": target,
        },
        columns=("Time", *BCE_FEATURES, "Amount", "Class"),
    )
    _validate_synthetic_frame(dataframe, effective_config)
    return dataframe, {
        "data_source_kind": "synthetic",
        "synthetic_generator_schema": _GENERATOR_SCHEMA,
        "synthetic_generation_seed": generation_seed,
        "synthetic_row_target": row_count,
        "evidence_classification": effective_config.evidence_classification,
        "evidence_boundary": _EVIDENCE_BOUNDARY,
    }
