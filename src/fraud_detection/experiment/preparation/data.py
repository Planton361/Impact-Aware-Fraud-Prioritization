"""Canonical data-preparation boundary for the frozen experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.artifacts import (
    _ordered_index_sha256,
    _write_csv,
    _write_json,
    require_generated_path,
)
from fraud_detection.errors import ProductError

from ..config import (
    BCE_FEATURES,
    EXPECTED_DEDUPLICATED_SHA256,
    EXPECTED_RAW_SHA256,
    RANKER_SCOPE,
    EffectiveExperimentConfig,
    _utc_now,
)
from .synthetic import generate_synthetic_data

_REAL_DEDUPLICATED_COUNTS = (283726, 283253, 473)
_REAL_OUTER_SPLIT_FRAUD_COUNTS = (378, 95)


def _expected_outer_split_rows(
    effective_config: EffectiveExperimentConfig,
) -> tuple[int, int]:
    row_count = (
        _REAL_DEDUPLICATED_COUNTS[0]
        if effective_config.data_source_kind == "real"
        else effective_config.synthetic_row_target
    )
    if row_count is None:
        raise RuntimeError("The configured data source has no expected row count.")
    test_rows = (row_count + 4) // 5
    return row_count - test_rows, test_rows


def _expected_outer_split_frauds(
    effective_config: EffectiveExperimentConfig,
) -> tuple[int, int] | None:
    if effective_config.data_source_kind == "real":
        return _REAL_OUTER_SPLIT_FRAUD_COUNTS
    return None


def expected_v_columns() -> list[str]:
    return list(BCE_FEATURES)


def required_columns() -> list[str]:
    return ["Time", *expected_v_columns(), "Amount", "Class"]


def validate_creditcard_schema(dataframe: pd.DataFrame) -> None:
    required = required_columns()
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    non_numeric = [
        column
        for column in required
        if not pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric columns in required schema: {non_numeric}")

    required_numeric = dataframe[required]
    if required_numeric.isna().any().any():
        raise ValueError("Required columns contain NaN values.")

    if np.isinf(required_numeric.to_numpy(dtype=float)).any():
        raise ValueError("Required columns contain infinite values.")

    target_values = set(dataframe["Class"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("`Class` must be binary with values {0, 1}.")

    if (dataframe["Amount"] < 0).any():
        raise ValueError("`Amount` must be non-negative.")


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    validate_creditcard_schema(dataframe)
    return dataframe


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha256(dataframe: pd.DataFrame) -> str:
    """Hash ordered full-row content plus column names and dtypes.

    The index is intentionally excluded: the digest identifies the ordered
    dataframe values used by the pipeline, while retained source indices are
    audited separately by the split metadata.
    """
    digest = hashlib.sha256()
    digest.update(b"fraud_detection.dataframe_content_sha256.v1\0")
    for column, dtype in zip(
        dataframe.columns,
        dataframe.dtypes,
        strict=True,
    ):
        digest.update(str(column).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(dtype).encode("ascii"))
        digest.update(b"\0")
    row_hashes = pd.util.hash_pandas_object(
        dataframe,
        index=False,
        categorize=True,
    ).to_numpy(dtype="<u8", copy=False)
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FullRowDeduplicationAudit:
    deduplication_enabled: bool
    rows_before_deduplication: int
    rows_removed_as_duplicate_followups: int
    rows_after_deduplication: int
    class_0_before: int
    class_0_after: int
    class_1_before: int
    class_1_after: int
    source_data_sha256: str
    processed_dataframe_sha256: str
    columns_used_for_deduplication: tuple[str, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "deduplication_enabled": self.deduplication_enabled,
            "duplicate_definition": "exact full-row equality",
            "duplicate_keep": "first",
            "rows_before_deduplication": self.rows_before_deduplication,
            "rows_removed_as_duplicate_followups": (
                self.rows_removed_as_duplicate_followups
            ),
            "rows_after_deduplication": self.rows_after_deduplication,
            "class_0_before": self.class_0_before,
            "class_0_after": self.class_0_after,
            "class_1_before": self.class_1_before,
            "class_1_after": self.class_1_after,
            "source_data_sha256": self.source_data_sha256,
            "deduplicated_dataframe_sha256": (
                self.processed_dataframe_sha256
                if self.deduplication_enabled
                else None
            ),
            "processed_dataframe_sha256": self.processed_dataframe_sha256,
            "dataframe_sha256_method": (
                "SHA-256 over ordered pandas full-row hashes plus ordered "
                "column names and dtypes; dataframe index excluded"
            ),
            "columns_used_for_deduplication": list(
                self.columns_used_for_deduplication
            ),
            "deduplication_pipeline_position": (
                "after CSV schema and basic validation; before outer split, "
                "scaler fit, OOF folds, fraud-amount quantiles, ranking "
                "groups, model training, and evaluation"
            ),
        }


def apply_full_row_deduplication(
    dataframe: pd.DataFrame,
    *,
    source_data_sha256: str,
) -> tuple[pd.DataFrame, FullRowDeduplicationAudit]:
    """Apply the canonical stable full-row deduplication step."""
    validate_creditcard_schema(dataframe)

    rows_before = len(dataframe)
    class_0_before = int((dataframe["Class"] == 0).sum())
    class_1_before = int((dataframe["Class"] == 1).sum())
    processed = dataframe.drop_duplicates(keep="first").copy()

    rows_after = len(processed)
    audit = FullRowDeduplicationAudit(
        deduplication_enabled=True,
        rows_before_deduplication=rows_before,
        rows_removed_as_duplicate_followups=rows_before - rows_after,
        rows_after_deduplication=rows_after,
        class_0_before=class_0_before,
        class_0_after=int((processed["Class"] == 0).sum()),
        class_1_before=class_1_before,
        class_1_after=int((processed["Class"] == 1).sum()),
        source_data_sha256=source_data_sha256,
        processed_dataframe_sha256=dataframe_content_sha256(processed),
        columns_used_for_deduplication=tuple(
            str(column) for column in dataframe.columns
        ),
    )
    return processed, audit


def load_deduplicated_data(
    data_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_hash = file_sha256(data_path)
    if raw_hash != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"Raw data SHA-256 mismatch: {raw_hash} != {EXPECTED_RAW_SHA256}"
        )
    raw = load_creditcard_csv(data_path)
    deduplicated, audit = apply_full_row_deduplication(
        raw,
        source_data_sha256=raw_hash,
    )
    metadata = audit.to_metadata()
    if metadata["deduplicated_dataframe_sha256"] != EXPECTED_DEDUPLICATED_SHA256:
        raise RuntimeError("Deduplicated DataFrame SHA-256 mismatch.")
    actual_counts = (
        len(deduplicated),
        int((deduplicated["Class"] == 0).sum()),
        int((deduplicated["Class"] == 1).sum()),
    )
    if actual_counts != _REAL_DEDUPLICATED_COUNTS:
        raise RuntimeError(f"Deduplicated class/count mismatch: {actual_counts}")
    if deduplicated.duplicated().any():
        raise RuntimeError("Full-row duplicates remain after deduplication.")
    return deduplicated, metadata


def load_experiment_data(
    data_path: Path,
    effective_config: EffectiveExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load or generate data, then return the shared deduplicated contract."""

    if effective_config.data_source_kind == "real":
        return load_deduplicated_data(data_path.resolve())

    generated, generation_metadata = generate_synthetic_data(effective_config)
    generated_identity = dataframe_content_sha256(generated)
    deduplicated, audit = apply_full_row_deduplication(
        generated,
        source_data_sha256=generated_identity,
    )
    if audit.rows_removed_as_duplicate_followups != 0:
        raise ProductError(
            "FD-SYNTHETIC-DUPLICATES",
            "Synthetic generation unexpectedly produced duplicate rows.",
            ("Keep the deterministic synthetic generator unchanged.",),
            {"duplicate_rows": audit.rows_removed_as_duplicate_followups},
        )
    metadata = audit.to_metadata()
    metadata.update(generation_metadata)
    metadata.update(
        {
            "synthetic_requested_row_count": (
                effective_config.synthetic_row_target
            ),
            "synthetic_generated_row_count": len(deduplicated),
            "synthetic_generated_fraud_count": int(
                (deduplicated["Class"] == 1).sum()
            ),
            "synthetic_generated_legitimate_count": int(
                (deduplicated["Class"] == 0).sum()
            ),
            "synthetic_data_identity": metadata[
                "deduplicated_dataframe_sha256"
            ],
        }
    )
    return deduplicated, metadata


def outer_split_indices(
    dataframe: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = dataframe.index.to_numpy()
    train_index, test_index = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=int(seed),
        stratify=dataframe["Class"].to_numpy(dtype=int),
    )
    return np.asarray(train_index, dtype=int), np.asarray(test_index, dtype=int)


def preflight_split_identity(
    output_root: Path,
    seed: int,
) -> pd.Series:
    identity = pd.read_csv(
        output_root / "preflight" / "outer_split_identity.csv"
    )
    rows = identity.loc[identity["seed"].astype(int) == int(seed)]
    if len(rows) != 1 or not bool(rows.iloc[0]["passed"]):
        raise RuntimeError(f"Missing passing preflight identity for seed {seed}.")
    return rows.iloc[0]


def verify_outer_split_without_test_labels(
    dataframe: pd.DataFrame,
    output_root: Path,
    seed: int,
    effective_config: EffectiveExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    train_index, test_index = outer_split_indices(dataframe, seed)
    identity = preflight_split_identity(output_root, seed)
    if _ordered_index_sha256(train_index) != identity["train_index_sha256"]:
        raise RuntimeError(f"Outer train split mismatch for seed {seed}.")
    if _ordered_index_sha256(test_index) != identity["test_index_sha256"]:
        raise RuntimeError(f"Outer test split mismatch for seed {seed}.")
    expected_train_rows, expected_test_rows = _expected_outer_split_rows(
        effective_config
    )
    if (
        len(train_index) != expected_train_rows
        or len(test_index) != expected_test_rows
    ):
        raise RuntimeError(f"Outer split row-count mismatch for seed {seed}.")
    return train_index, test_index


def _load_preflight(
    output_root: Path,
    effective_config: EffectiveExperimentConfig,
) -> dict[str, Any]:
    path = output_root / "preflight" / "preflight_validation.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing preflight manifest: {path}")
    preflight = json.loads(path.read_text(encoding="utf-8-sig"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("Preflight status is not PASS.")
    checked = preflight.get("stop_conditions_checked", {})
    if not checked or not all(bool(value) for value in checked.values()):
        raise RuntimeError("Not all preflight stop conditions passed.")
    expected_definitions = {
        "outer_seeds": list(effective_config.seeds),
        "target_budgets": list(effective_config.target_budgets),
        "ranker_scope": RANKER_SCOPE,
        "candidate_pool_size": effective_config.candidate_pool_size,
    }
    if preflight.get("locked_definitions") != expected_definitions:
        raise RuntimeError("Preflight configuration does not match the profile.")
    return preflight


def _initialize_preflight(
    *,
    repository_root: Path,
    output_root: Path,
    data_path: Path,
    effective_config: EffectiveExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create the identity preflight for a new generated run root.

    This function verifies only data and split identity.  It does not fit a
    model or create a score.  The empirical computation remains in the
    inner/final phase functions.
    """

    require_generated_path(repository_root, output_root)
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to initialize an existing output root: {output_root}"
        )
    dataframe, data_metadata = load_experiment_data(
        data_path,
        effective_config,
    )

    identity_rows: list[dict[str, Any]] = []
    expected_train_rows, expected_test_rows = _expected_outer_split_rows(
        effective_config
    )
    expected_fraud_counts = _expected_outer_split_frauds(effective_config)
    for seed in effective_config.seeds:
        train_index, test_index = outer_split_indices(dataframe, seed)
        train_labels = dataframe.loc[train_index, "Class"].to_numpy(dtype=int)
        test_labels = dataframe.loc[test_index, "Class"].to_numpy(dtype=int)
        if (
            len(train_index) != expected_train_rows
            or len(test_index) != expected_test_rows
            or (
                expected_fraud_counts is not None
                and (
                    int(train_labels.sum()),
                    int(test_labels.sum()),
                )
                != expected_fraud_counts
            )
        ):
            raise RuntimeError(f"Outer split identity mismatch for seed {seed}.")
        identity_rows.append(
            {
                "seed": seed,
                "train_rows": len(train_index),
                "test_rows": len(test_index),
                "train_fraud": int(train_labels.sum()),
                "test_fraud": int(test_labels.sum()),
                "train_index_sha256": _ordered_index_sha256(train_index),
                "test_index_sha256": _ordered_index_sha256(test_index),
                "passed": True,
            }
        )

    output_root.mkdir(parents=True, exist_ok=False)
    preflight_dir = output_root / "preflight"
    preflight_dir.mkdir()
    _write_csv(
        preflight_dir / "outer_split_identity.csv",
        pd.DataFrame(identity_rows),
    )
    if effective_config.data_source_kind == "real":
        preflight_data = {
            "raw_path": str(data_path.resolve()),
            "expected_raw_sha256": EXPECTED_RAW_SHA256,
            "actual_raw_sha256": data_metadata["source_data_sha256"],
            "deduplication_keep": "first",
            "deduplication_before_split": True,
            "rows": len(dataframe),
            "class_0": int((dataframe["Class"] == 0).sum()),
            "class_1": int((dataframe["Class"] == 1).sum()),
            "expected_deduplicated_dataframe_sha256": (
                EXPECTED_DEDUPLICATED_SHA256
            ),
            "actual_deduplicated_dataframe_sha256": data_metadata[
                "deduplicated_dataframe_sha256"
            ],
        }
        stop_conditions = {
            "raw_data_hash": True,
            "deduplicated_data_hash": True,
            "outer_split_identity": True,
            "output_root_is_new_and_ignored": True,
            "no_model_fit_during_preflight": True,
            "no_model_scoring_during_preflight": True,
        }
    else:
        preflight_data = {
            "data_source_kind": data_metadata["data_source_kind"],
            "synthetic_generator_schema": data_metadata[
                "synthetic_generator_schema"
            ],
            "synthetic_generation_seed": data_metadata[
                "synthetic_generation_seed"
            ],
            "synthetic_requested_row_count": data_metadata[
                "synthetic_requested_row_count"
            ],
            "synthetic_generated_row_count": data_metadata[
                "synthetic_generated_row_count"
            ],
            "synthetic_generated_fraud_count": data_metadata[
                "synthetic_generated_fraud_count"
            ],
            "synthetic_generated_legitimate_count": data_metadata[
                "synthetic_generated_legitimate_count"
            ],
            "deterministic_data_identity": data_metadata[
                "synthetic_data_identity"
            ],
            "evidence_classification": data_metadata[
                "evidence_classification"
            ],
            "evidence_boundary": data_metadata["evidence_boundary"],
            "deduplication_keep": "first",
            "deduplication_before_split": True,
        }
        stop_conditions = {
            "synthetic_data_identity": True,
            "deduplicated_data_hash": True,
            "outer_split_identity": True,
            "output_root_is_new_and_ignored": True,
            "no_model_fit_during_preflight": True,
            "no_model_scoring_during_preflight": True,
        }

    _write_json(
        preflight_dir / "preflight_validation.json",
        {
            "schema": "ranker_gain_validation.preflight.v2",
            "status": "PASS",
            "created_at_utc": _utc_now(),
            "data": preflight_data,
            "seed_split_identity_path": (
                "preflight/outer_split_identity.csv"
            ),
            "locked_definitions": {
                "outer_seeds": list(effective_config.seeds),
                "target_budgets": list(effective_config.target_budgets),
                "ranker_scope": RANKER_SCOPE,
                "candidate_pool_size": effective_config.candidate_pool_size,
            },
            "stop_conditions_checked": stop_conditions,
            "model_fit_performed": False,
            "model_scoring_performed": False,
        },
    )
    return dataframe, data_metadata


def _prepare_output_root(
    args: argparse.Namespace,
    effective_config: EffectiveExperimentConfig,
    repository_root: Path,
) -> None:
    output_root = require_generated_path(
        repository_root,
        Path(args.output_dir),
    )
    args.output_dir = str(output_root)
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"Output root is not a directory: {output_root}")
        if not (output_root / "preflight" / "preflight_validation.json").is_file():
            raise FileExistsError(
                "Refusing an existing output root without a completed preflight."
            )
        return
    if args.phase not in {"inner", "all"}:
        raise FileNotFoundError(
            "A new run must begin with phase 'inner' or 'all'."
        )
    _initialize_preflight(
        repository_root=repository_root,
        output_root=output_root,
        data_path=Path(args.data_path),
        effective_config=effective_config,
    )
