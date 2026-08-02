"""Shared, computation-free helpers for generated thesis presentation files.

The empirical experiment writes to ``outputs/``.  Presentation builders write
to ``generated/`` or ``thesis_build/``.  All three roots are ignored by Git.
Keeping the path policy here prevents a command-line typo from overwriting
source files or a frozen evidence worktree.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Final, Iterable

import pandas as pd

from fraud_detection.artifacts import require_generated_path

METHOD_BCE: Final[str] = "baseline_bce_probability"
METHOD_P_ONLY: Final[str] = "selected_candidate_p_only"
METHOD_AMOUNT_GAIN: Final[str] = "selected_candidate_amount_gain"
METHOD_FIXED: Final[str] = "candidate_postprocessing_p_times_log_amount"
METHOD_ORDER: Final[tuple[str, ...]] = (
    METHOD_BCE,
    METHOD_P_ONLY,
    METHOD_AMOUNT_GAIN,
    METHOD_FIXED,
)
PATH_IDS: Final[dict[str, str]] = {
    METHOD_BCE: "BCE",
    METHOD_P_ONLY: "p_only",
    METHOD_AMOUNT_GAIN: "amount_gain",
    METHOD_FIXED: "fixed_reference",
}
GERMAN_PATH_LABELS: Final[dict[str, str]] = {
    METHOD_BCE: "BCE",
    METHOD_P_ONLY: "p-only",
    METHOD_AMOUNT_GAIN: "Amount-Gain",
    METHOD_FIXED: "feste Referenz",
}
def prepare_output_directory(
    repository_root: Path,
    output: Path,
    *,
    force: bool = False,
) -> Path:
    """Create a safe output directory and reject accidental overwrites.

    ``force`` is explicit authorization to replace only the selected directory
    below one of the three generated roots.  The resolved target is validated
    before any recursive operation.
    """

    target = require_generated_path(repository_root, output)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"Output path is not a directory: {target}")
        if any(target.iterdir()):
            if not force:
                raise FileExistsError(
                    f"Refusing to overwrite non-empty output directory: {target}"
                )
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def require_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write stable UTF-8/LF CSV output."""

    require_new_file(path)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(path: Path, payload: object) -> None:
    """Write stable, timestamp-free JSON output."""

    require_new_file(path)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


from .build import (  # noqa: E402 - public facade follows helper initialization
    PresentationConfig,
    PresentationError,
    PresentationResult,
    PresentationStepResult,
    build_presentation,
)

__all__ = [
    "PresentationConfig",
    "PresentationStepResult",
    "PresentationResult",
    "PresentationError",
    "build_presentation",
]
