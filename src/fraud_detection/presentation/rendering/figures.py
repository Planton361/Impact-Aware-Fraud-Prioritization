"""Render the final Chapter-5 figures from manifested presentation data only."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "fraud_detection_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter

from .. import (
    METHOD_AMOUNT_GAIN,
    METHOD_BCE,
    METHOD_FIXED,
    METHOD_ORDER,
    METHOD_P_ONLY,
    file_inventory,
    prepare_output_directory,
    require_generated_path,
    sha256_file,
    write_json,
)
from ..catalog import CANONICAL_ARTIFACT_IDS, ENGINEERING_ARTIFACT_IDS
from .style import (
    CENTRAL_BAND_COLOUR,
    COMPARISON_METHODS,
    DEFAULT_WIDTH_MM,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    PANEL_LETTERS,
    PATH_STYLES,
    PNG_DPI,
    configure_presentation_style,
    decimal_comma,
    format_german,
    method_legend_handles,
    method_marker_legend_handles,
    mm_to_inches,
    panel_label,
    style_axis,
)

_EventSink = Callable[[str, Mapping[str, object]], None]
EXPECTED_DATA_SCHEMA = "fraud_detection.chapter5_presentation_data.r6.v1"
MAIN_STEMS = (
    "ch5_f1_paired_plr_fraud_tradeoff",
    "ch5_f2_budget_policy_profile",
    "ch5_f3_within_model_depth_k50",
    "ch5_f4_global_roc_pr_k50",
    "ch5_f6_replacement_case_map_k50",
)
OPTIONAL_STEMS = ("ch5_f5_hard_impact_profile_k50",)
APPENDIX_STEMS = (
    "app_f1_seed_budget_delta_heatmap",
    "app_f2_exact_tie_bound_intervals",
    "app_f3_candidate_pool_ceiling_utilization",
)
FIXED_PDF_DATE = datetime(2026, 7, 27, tzinfo=timezone.utc)
EXPECTED_FIGURE_COUNT = 9
EXPECTED_FIGURE_FILE_COUNT = 27


def _status_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _emit_status(
    event_sink: _EventSink | None,
    level: str,
    event: str,
    **fields: object,
) -> None:
    if event_sink is not None:
        event_sink(
            "status",
            {
                "level": level,
                "event": event,
                "fields": fields,
            },
        )


def _record_figure_completion(
    files: list[Path],
    rendered: list[Path],
    *,
    completed: int,
    stem: str,
    event_sink: _EventSink | None,
) -> None:
    files.extend(rendered)
    _emit_status(
        event_sink,
        "INFO",
        "render-figure",
        completed=completed,
        total=EXPECTED_FIGURE_COUNT,
        stem=stem,
        files_completed=completed * 3,
        files_total=EXPECTED_FIGURE_FILE_COUNT,
    )


class FigureStore:
    """Checksum-gated reader for prepared figure data."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "PRESENTATION_DATA_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing presentation-data manifest: {manifest_path}"
            )
        try:
            self.manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Presentation-data manifest is not valid UTF-8 JSON."
            ) from exc
        if self.manifest.get("schema") != EXPECTED_DATA_SCHEMA:
            raise RuntimeError(
                "Presentation-data manifest schema is not the approved R6 schema."
            )
        if self.manifest.get("status") != "PASS":
            raise RuntimeError("Presentation-data manifest is not PASS.")
        rows = self.manifest.get("outputs", [])
        if not isinstance(rows, list):
            raise RuntimeError("Presentation-data output inventory is invalid.")
        hashes: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    "Presentation-data output inventory is invalid."
                )
            relative = row.get("path")
            digest = row.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or ":" in PurePosixPath(relative).parts[0]
            ):
                raise RuntimeError(
                    "Presentation-data manifest contains an unsafe output path."
                )
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError(
                    "Presentation-data output checksum is invalid."
                )
            hashes[relative] = digest
        self.hashes = hashes
        if len(self.hashes) != len(rows):
            raise RuntimeError("Presentation-data manifest contains duplicate outputs.")
        self.presentation_role = self.manifest.get("presentation_role")
        self.profile = self.manifest.get("profile")
        self.evidence_classification = self.manifest.get(
            "evidence_classification"
        )
        self.data_source_kind = self.manifest.get("data_source_kind")
        selected = self.manifest.get("selected_catalog_artifact_ids")
        if not isinstance(selected, list) or not all(
            isinstance(value, str) for value in selected
        ):
            raise RuntimeError(
                "Presentation-data catalog registration is invalid."
            )
        self.selected_catalog_artifact_ids = tuple(selected)
        if self.presentation_role == "canonical":
            if (
                self.profile != "canonical"
                or self.evidence_classification != "thesis-evidentiary"
                or self.data_source_kind != "real"
                or self.selected_catalog_artifact_ids
                != CANONICAL_ARTIFACT_IDS
            ):
                raise RuntimeError(
                    "Canonical presentation-data role or catalog is invalid."
                )
            self.evidence_statement = None
            self.comparability_boundary = None
        elif self.presentation_role == "engineering":
            if (
                self.profile not in {"mini-real", "smoke-synthetic"}
                or self.selected_catalog_artifact_ids
                != ENGINEERING_ARTIFACT_IDS
            ):
                raise RuntimeError(
                    "Engineering presentation-data role or catalog is invalid."
                )
            self.evidence_statement = self.manifest.get(
                "evidence_statement"
            )
            self.comparability_boundary = self.manifest.get(
                "comparability_boundary"
            )
            if (
                not isinstance(self.evidence_statement, str)
                or "not thesis evidence" not in self.evidence_statement
                or not isinstance(self.comparability_boundary, str)
                or self.comparability_boundary
                not in self.evidence_statement
            ):
                raise RuntimeError(
                    "Engineering presentation evidence metadata is missing."
                )
        else:
            raise RuntimeError("Presentation-data role is unsupported.")
        self.seeds = tuple(int(seed) for seed in self.manifest.get("seeds", []))
        self.budgets = tuple(
            int(budget) for budget in self.manifest.get("budgets", [])
        )
        if not self.seeds or not self.budgets:
            raise RuntimeError("Presentation-data seed/budget order is missing.")
        self.read_paths: set[str] = set()

    def csv(
        self,
        name: str,
        required_columns: Iterable[str],
    ) -> pd.DataFrame:
        filename = name if name.endswith(".csv") else f"{name}.csv"
        relative = filename if "/" in filename else f"figures/{filename}"
        if relative not in self.hashes:
            raise RuntimeError(f"Unregistered figure data rejected: {relative}")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise RuntimeError(f"Unsafe figure data path rejected: {relative}")
        if not path.is_file() or sha256_file(path) != self.hashes[relative]:
            raise RuntimeError(f"Figure-data checksum mismatch: {relative}")
        frame = pd.read_csv(path)
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            raise RuntimeError(
                f"Figure data {relative} is missing required columns: {missing}"
            )
        self.read_paths.add(relative)
        return frame


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes"}
    )


def _single_value(frame: pd.DataFrame, column: str, context: str) -> float:
    if len(frame) != 1:
        raise RuntimeError(f"Expected one row for {context}, observed {len(frame)}.")
    value = float(frame[column].iloc[0])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value for {context}/{column}.")
    return value


def _figure_size(width_mm: float, base_height_mm: float) -> tuple[float, float]:
    scale = width_mm / DEFAULT_WIDTH_MM
    return mm_to_inches(width_mm), mm_to_inches(base_height_mm * scale)


def _visible_text_artists(fig: Figure) -> list[Text]:
    return [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and bool(artist.get_text().strip())
    ]


def _validate_figure_canvas(fig: Figure, stem: str) -> None:
    """Fail if visible text or legends escape the fixed physical canvas."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tolerance = 2.0
    issues: list[str] = []
    for artist in _visible_text_artists(fig):
        bbox = artist.get_window_extent(renderer=renderer)
        if not np.isfinite([bbox.x0, bbox.y0, bbox.x1, bbox.y1]).all():
            issues.append(f"non-finite text bbox: {artist.get_text()!r}")
            continue
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            issues.append(f"text outside canvas: {artist.get_text()!r}")
        if artist.get_fontsize() < 6.0:
            issues.append(f"unreadable text size: {artist.get_text()!r}")
    legends = list(fig.findobj(match=Legend))
    for legend in legends:
        bbox = legend.get_window_extent(renderer=renderer)
        if (
            bbox.x0 < canvas.x0 - tolerance
            or bbox.y0 < canvas.y0 - tolerance
            or bbox.x1 > canvas.x1 + tolerance
            or bbox.y1 > canvas.y1 + tolerance
        ):
            issues.append("legend outside canvas")
    if issues:
        preview = "; ".join(issues[:8])
        raise RuntimeError(f"Layout validation failed for {stem}: {preview}")


def _save_figure(
    fig: Figure,
    output: Path,
    stem: str,
    *,
    subject: str = "Deterministic Chapter-5 presentation render",
    description: str = "Deterministic Chapter-5 presentation render",
) -> list[Path]:
    _validate_figure_canvas(fig, stem)
    output.mkdir(parents=True, exist_ok=True)
    targets = {
        "pdf": output / f"{stem}.pdf",
        "svg": output / f"{stem}.svg",
        "png": output / f"{stem}.png",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite figure stem: {stem}")

    pdf_metadata = {
        "Title": stem,
        "Author": "Fraud Detection Reproducibility Pipeline",
        "Subject": subject,
        "Keywords": "fraud detection, ranking, reproducibility",
        "Creator": "fraud_detection.presentation.figures",
        "Producer": "Matplotlib",
        "CreationDate": FIXED_PDF_DATE,
        "ModDate": FIXED_PDF_DATE,
    }
    svg_metadata = {
        "Title": stem,
        "Date": "2026-07-27",
        "Creator": "fraud_detection.presentation.figures",
        "Description": description,
    }
    png_metadata = {"Software": "fraud_detection.presentation.figures"}
    fig.savefig(
        targets["pdf"],
        metadata=pdf_metadata,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    fig.savefig(targets["svg"], metadata=svg_metadata)
    fig.savefig(targets["png"], dpi=PNG_DPI, metadata=png_metadata)
    plt.close(fig)
    return [targets["pdf"], targets["svg"], targets["png"]]


def _render_tradeoff(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    seedwise = store.csv(
        "ch5_tradeoff_seedwise",
        [
            "seed",
            "target_budget",
            "method_family",
            "delta_plr_vs_bce",
            "delta_fraud_at_k_vs_bce",
        ],
    )
    summary = store.csv(
        "ch5_tradeoff_summary",
        [
            "target_budget",
            "method_family",
            "delta_plr_vs_bce_mean",
            "delta_fraud_at_k_vs_bce_mean",
        ],
    )
    budgets = (20, 50, 100)
    observed = set(
        seedwise[["seed", "target_budget", "method_family"]].itertuples(
            index=False, name=None
        )
    )
    expected = {
        (seed, budget, method)
        for seed in store.seeds
        for budget in budgets
        for method in COMPARISON_METHODS
    }
    if observed != expected:
        raise RuntimeError("Trade-off seed/budget/path grid is incomplete.")

    x_values = np.concatenate(
        [
            seedwise["delta_fraud_at_k_vs_bce"].to_numpy(float),
            summary["delta_fraud_at_k_vs_bce_mean"].to_numpy(float),
        ]
    )
    x_limit = max(1.0, float(np.nanmax(np.abs(x_values))) * 1.10)
    y_values = np.concatenate(
        [
            seedwise["delta_plr_vs_bce"].to_numpy(float),
            summary["delta_plr_vs_bce_mean"].to_numpy(float),
        ]
    )
    y_limits = (-0.20, 0.65)
    if float(np.nanmin(y_values)) < y_limits[0] or float(
        np.nanmax(y_values)
    ) > y_limits[1]:
        raise RuntimeError("Trade-off PLR values exceed the approved R7B y-range.")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=_figure_size(width_mm, 64.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for index, (axis, budget) in enumerate(zip(axes, budgets, strict=True)):
        style_axis(axis)
        axis.axvline(0.0, color=NEUTRAL_MID, linewidth=0.65, zorder=0)
        axis.axhline(0.0, color=NEUTRAL_MID, linewidth=0.65, zorder=0)
        for method in COMPARISON_METHODS:
            path_style = PATH_STYLES[method]
            rows = seedwise.loc[
                (seedwise["target_budget"].astype(int) == budget)
                & (seedwise["method_family"] == method)
            ]
            axis.scatter(
                rows["delta_fraud_at_k_vs_bce"],
                rows["delta_plr_vs_bce"],
                s=22,
                marker=path_style.marker,
                facecolors=path_style.colour,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.72,
                zorder=3,
            )
            mean_row = summary.loc[
                (summary["target_budget"].astype(int) == budget)
                & (summary["method_family"] == method)
            ]
            mean_x = _single_value(
                mean_row,
                "delta_fraud_at_k_vs_bce_mean",
                f"trade-off mean k={budget}/{method}",
            )
            mean_y = _single_value(
                mean_row,
                "delta_plr_vs_bce_mean",
                f"trade-off mean k={budget}/{method}",
            )
            axis.scatter(
                [mean_x],
                [mean_y],
                s=40,
                marker="D",
                facecolors="white",
                edgecolors=path_style.colour,
                linewidths=1.15,
                zorder=5,
            )

        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(*y_limits)
        axis.set_yticks([-0.20, 0.00, 0.20, 0.40, 0.60])
        inner_tick_limit = max(1.0, math.floor(x_limit))
        axis.set_xticks(
            np.linspace(-inner_tick_limit, inner_tick_limit, 5)
        )
        axis.xaxis.set_major_formatter(decimal_comma(0))
        axis.yaxis.set_major_formatter(decimal_comma(2))
        axis.set_title(f"k = {budget}", pad=4.0)
        panel_label(axis, PANEL_LETTERS[index])
    axes[0].set_ylabel("Δ PLR@k gegenüber BCE")
    fig.supxlabel("Δ Fraud@k gegenüber BCE", fontsize=8.0)
    fig.legend(
        handles=method_marker_legend_handles(COMPARISON_METHODS),
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=0.8,
        columnspacing=1.3,
    )
    return _save_figure(
        fig,
        output,
        "ch5_f1_paired_plr_fraud_tradeoff",
    )


def _render_budget_policy(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    store.csv(
        "ch5_budget_policy_seedwise",
        [
            "seed",
            "target_budget",
            "budget_position",
            "method_family",
            "prevented_loss_ratio_at_k",
            "delta_fraud_at_k_vs_bce",
            "central_budget",
            "separate_budget_conditioned_model",
        ],
    )
    summary = store.csv(
        "ch5_budget_policy_summary",
        [
            "target_budget",
            "budget_position",
            "central_budget",
            "method_family",
            "separate_budget_conditioned_model",
            "prevented_loss_ratio_at_k_mean",
            "delta_fraud_at_k_vs_bce_mean",
        ],
    )
    budgets = tuple(
        summary[["target_budget", "budget_position"]]
        .drop_duplicates()
        .sort_values("budget_position", kind="mergesort")["target_budget"]
        .astype(int)
    )
    if budgets != store.budgets:
        raise RuntimeError("Budget-policy order differs from the manifested order.")
    if len(summary) != len(budgets) * len(METHOD_ORDER):
        raise RuntimeError("Budget-policy summary grid is incomplete.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=_figure_size(width_mm, 101.0),
        sharex=True,
        layout="constrained",
    )
    for axis in axes:
        style_axis(axis)
        central_positions = sorted(
            summary.loc[_as_bool(summary["central_budget"]), "budget_position"]
            .astype(int)
            .unique()
        )
        for position in central_positions:
            axis.axvspan(
                position - 0.19,
                position + 0.19,
                color=CENTRAL_BAND_COLOUR,
                alpha=0.65,
                linewidth=0,
                zorder=-2,
            )

    metric_specs = (
        (
            axes[0],
            "prevented_loss_ratio_at_k_mean",
            METHOD_ORDER,
        ),
        (
            axes[1],
            "delta_fraud_at_k_vs_bce_mean",
            COMPARISON_METHODS,
        ),
    )
    for axis, mean_column, methods in metric_specs:
        for method in methods:
            rows = summary.loc[summary["method_family"] == method].sort_values(
                "budget_position", kind="mergesort"
            )
            path_style = PATH_STYLES[method]
            axis.plot(
                rows["budget_position"],
                rows[mean_column],
                color=path_style.colour,
                marker=path_style.marker,
                linestyle=path_style.linestyle,
                markerfacecolor="white",
                markeredgecolor=path_style.colour,
                markeredgewidth=0.75,
                linewidth=0.9,
                zorder=3,
            )
    axes[1].axhline(0.0, color=NEUTRAL_DARK, linewidth=0.8, zorder=0)
    upper_limits = (0.00, 0.85)
    axes[0].set_ylim(*upper_limits)
    axes[0].set_yticks([0.00, 0.20, 0.40, 0.60, 0.80])
    lower_limits = (-10.0, 5.0)
    axes[1].set_ylim(*lower_limits)
    axes[1].set_yticks([-10.0, -5.0, 0.0, 5.0])
    axes[0].set_ylabel("PLR@k")
    axes[1].set_ylabel("Δ Fraud@k gegenüber BCE")
    axes[1].set_xlabel("Untersuchungsbudget k")
    axes[0].yaxis.set_major_formatter(decimal_comma(2))
    axes[1].yaxis.set_major_formatter(decimal_comma(0))
    positions = np.arange(len(budgets))
    axes[1].set_xticks(positions, [str(budget) for budget in budgets])
    axes[1].set_xlim(-0.4, len(budgets) - 0.6)
    for index, axis in enumerate(axes):
        panel_label(axis, PANEL_LETTERS[index])
    fig.legend(
        handles=method_legend_handles(METHOD_ORDER),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )
    return _save_figure(fig, output, "ch5_f2_budget_policy_profile")


def _render_depth(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    seedwise = store.csv(
        "ch5_depth_seedwise",
        [
            "seed",
            "target_budget",
            "method_family",
            "rank_depth",
            "cumulative_fraud_count",
            "cumulative_plr",
            "fixed_model_depth_profile",
            "cross_budget_subtraction",
        ],
    )
    summary = store.csv(
        "ch5_depth_summary",
        [
            "target_budget",
            "method_family",
            "rank_depth",
            "fixed_model_depth_profile",
            "cross_budget_subtraction",
            "cumulative_fraud_count_mean",
            "cumulative_plr_mean",
        ],
    )
    if _as_bool(seedwise["cross_budget_subtraction"]).any():
        raise RuntimeError("Depth input contains forbidden cross-budget subtraction.")
    if not _as_bool(seedwise["fixed_model_depth_profile"]).all():
        raise RuntimeError("Depth input is not marked as a fixed-model profile.")
    budget = 50
    checkpoints = (20, 50, 100)
    selected = summary.loc[
        summary["target_budget"].astype(int) == budget
    ].copy()
    if set(selected["method_family"]) != set(METHOD_ORDER):
        raise RuntimeError("The k=50 depth path grid is incomplete.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=_figure_size(width_mm, 70.0),
        sharex=True,
        layout="constrained",
    )
    metric_specs = (
        (
            axes[0],
            "cumulative_plr_mean",
            "Kumulatives PLR@r",
            2,
        ),
        (
            axes[1],
            "cumulative_fraud_count_mean",
            "Kumulative Fraud-Fälle bis Rang r",
            0,
        ),
    )
    for index, (axis, mean_column, y_label, decimals) in enumerate(
        metric_specs
    ):
        style_axis(axis)
        for checkpoint in checkpoints:
            is_target = checkpoint == budget
            axis.axvline(
                checkpoint,
                color=NEUTRAL_MID if is_target else "#BFBFBF",
                linestyle="--",
                linewidth=0.85 if is_target else 0.5,
                alpha=0.95 if is_target else 0.7,
                zorder=0,
                label=(
                    "target-budget-checkpoint"
                    if is_target
                    else "diagnostic-checkpoint"
                ),
            )
        for method in METHOD_ORDER:
            rows = selected.loc[
                selected["method_family"] == method
            ].sort_values("rank_depth", kind="mergesort")
            if len(rows) != 100 or rows["rank_depth"].astype(int).tolist() != list(
                range(1, 101)
            ):
                raise RuntimeError(
                    f"Depth summary is incomplete for k=50/{method}."
                )
            path_style = PATH_STYLES[method]
            axis.plot(
                rows["rank_depth"],
                rows[mean_column],
                color=path_style.colour,
                linestyle=path_style.linestyle,
                linewidth=1.05,
                zorder=3,
                label=f"mean:{method}",
            )
            checkpoint_rows = rows.loc[
                rows["rank_depth"].astype(int).isin(checkpoints)
            ].sort_values("rank_depth", kind="mergesort")
            if checkpoint_rows["rank_depth"].astype(int).tolist() != list(
                checkpoints
            ):
                raise RuntimeError(
                    f"Depth checkpoints are incomplete for k=50/{method}."
                )
            axis.scatter(
                checkpoint_rows["rank_depth"],
                checkpoint_rows[mean_column],
                s=15,
                marker=path_style.marker,
                facecolors="white",
                edgecolors=path_style.colour,
                linewidths=0.65,
                zorder=4,
            )
        axis.set_xlim(1, 101)
        axis.set_xticks([1, 20, 50, 100])
        axis.yaxis.set_major_formatter(decimal_comma(decimals))
        axis.set_ylabel(y_label)
        panel_label(axis, PANEL_LETTERS[index])

    plr_maximum = float(selected["cumulative_plr_mean"].max())
    plr_tick_upper = max(0.1, math.ceil(plr_maximum * 10.0) / 10.0)
    fraud_maximum = float(selected["cumulative_fraud_count_mean"].max())
    fraud_tick_upper = max(
        10.0,
        math.ceil(fraud_maximum / 10.0) * 10.0,
    )
    axes[0].set_ylim(0.0, plr_tick_upper * 1.02)
    axes[0].set_yticks(np.linspace(0.0, plr_tick_upper, 5))
    axes[1].set_ylim(0.0, fraud_tick_upper * 1.02)
    axes[1].set_yticks(np.linspace(0.0, fraud_tick_upper, 5))
    fig.supxlabel(
        "Rangtiefe r im Budgetmodell k = 50",
        fontsize=8.0,
    )
    fig.legend(
        handles=method_legend_handles(METHOD_ORDER),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )
    return _save_figure(
        fig,
        output,
        "ch5_f3_within_model_depth_k50",
    )


def _render_global_pool_curves(
    store: FigureStore,
    output: Path,
    width_mm: float,
    *,
    scope: str = "full_order",
) -> list[Path]:
    store.csv(
        "ch5_global_pool_curves_seedwise",
        [
            "seed",
            "target_budget",
            "scope",
            "curve_type",
            "method_family",
            "grid_index",
            "x",
            "y",
        ],
    )
    curves = store.csv(
        "ch5_global_pool_curves_summary",
        [
            "target_budget",
            "scope",
            "scope_label",
            "curve_type",
            "method_family",
            "grid_index",
            "x_name",
            "y_name",
            "x",
            "y_mean",
        ],
    )
    store.csv(
        "ch5_global_metrics_seedwise",
        [
            "seed",
            "target_budget",
            "scope",
            "method_family",
            "metric",
            "value",
            "score_interpretation",
        ],
    )
    metrics = store.csv(
        "ch5_global_metrics_summary",
        [
            "target_budget",
            "scope",
            "method_family",
            "metric",
            "value_mean",
            "value_sd",
            "score_interpretation",
        ],
    )
    brier = metrics.loc[metrics["metric"] == "brier"]
    if (
        brier.empty
        or set(brier["method_family"]) != {METHOD_BCE}
        or set(brier["scope"]) != {"full_order"}
    ):
        raise RuntimeError("Brier scope is not restricted to BCE full order.")

    if scope not in {"full_order", "candidate_pool"}:
        raise ValueError(f"Unsupported curve scope: {scope}")
    scope_label = (
        "vollständige Testordnung"
        if scope == "full_order"
        else "BCE-Top-1000-Kandidatenpool"
    )
    panels = (("roc", "ROC"), ("pr", "Precision–Recall"))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=_figure_size(width_mm, 70.0),
        layout="constrained",
    )
    for index, (axis, panel) in enumerate(zip(axes.flat, panels, strict=True)):
        curve_type, curve_label = panel
        style_axis(axis)
        for method in METHOD_ORDER:
            rows = curves.loc[
                (curves["target_budget"].astype(int) == 50)
                & (curves["scope"] == scope)
                & (curves["curve_type"] == curve_type)
                & (curves["method_family"] == method)
            ].sort_values("grid_index", kind="mergesort")
            if len(rows) != 1001:
                raise RuntimeError(
                    f"Curve grid is incomplete for {scope}/{curve_type}/{method}."
                )
            path_style = PATH_STYLES[method]
            axis.plot(
                rows["x"],
                rows["y_mean"],
                color=path_style.colour,
                linestyle=path_style.linestyle,
                linewidth=0.9,
                zorder=3,
                label=f"{scope}:{curve_type}:{method}",
            )
        if curve_type == "roc":
            axis.plot(
                [0, 1],
                [0, 1],
                color="#A6A6A6",
                linestyle="--",
                linewidth=0.65,
                zorder=0,
                label=f"{scope}:random-order",
            )
            if scope == "full_order":
                axis.text(
                    0.57,
                    0.53,
                    "Zufallsordnung",
                    rotation=45,
                    rotation_mode="anchor",
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    color=NEUTRAL_MID,
                    zorder=1,
                )
            axis.set_xlabel("Falsch-Positiv-Rate")
            axis.set_ylabel("Richtig-Positiv-Rate")
        else:
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.xaxis.set_major_formatter(decimal_comma(1))
        axis.yaxis.set_major_formatter(decimal_comma(1))
        axis.set_box_aspect(0.82)
        axis.set_title(
            f"{curve_label} –\n{scope_label}",
            pad=4.0,
        )
        panel_label(axis, PANEL_LETTERS[index], x=-0.09)
    fig.legend(
        handles=method_legend_handles(METHOD_ORDER),
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )
    return _save_figure(
        fig,
        output,
        (
            "ch5_f4_global_roc_pr_k50"
            if scope == "full_order"
            else "app_f4_candidate_pool_roc_pr_k50"
        ),
    )


def _render_hard_impact(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    seedwise = store.csv(
        "ch5_hard_impact_seedwise",
        [
            "seed",
            "target_budget",
            "q",
            "method_family",
            "q90_captured_ratio",
            "baseline_miss_recovery",
            "amount_ndcg_at_50",
        ],
    )
    summary = store.csv(
        "ch5_hard_impact_summary",
        [
            "target_budget",
            "method_family",
            "q90_captured_ratio_mean",
            "baseline_miss_recovery_mean",
            "amount_ndcg_at_50_mean",
        ],
    )
    bce_recovery = seedwise.loc[
        seedwise["method_family"] == METHOD_BCE,
        "baseline_miss_recovery",
    ]
    if bce_recovery.notna().any():
        raise RuntimeError("BCE baseline-miss recovery must remain missing.")
    if len(seedwise) != len(store.seeds) * len(METHOD_ORDER):
        raise RuntimeError("Hard-impact seed/path grid is incomplete.")

    metrics = (
        (
            "q90_captured_ratio",
            "q90_captured_ratio_mean",
            "q90-Fraud-Abdeckung",
        ),
        (
            "baseline_miss_recovery",
            "baseline_miss_recovery_mean",
            "BCE-Miss-Recovery",
        ),
        ("amount_ndcg_at_50", "amount_ndcg_at_50_mean", "Amount-nDCG@50"),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=_figure_size(width_mm, 49.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    y_positions = np.arange(len(METHOD_ORDER), dtype=float) * 0.68
    for index, (axis, metric) in enumerate(zip(axes, metrics, strict=True)):
        value_column, mean_column, title = metric
        style_axis(axis)
        for y_position, method in zip(
            y_positions, METHOD_ORDER, strict=True
        ):
            path_style = PATH_STYLES[method]
            rows = seedwise.loc[seedwise["method_family"] == method]
            finite = rows[value_column].dropna()
            if len(finite):
                axis.scatter(
                    finite,
                    np.full(len(finite), y_position),
                    s=20,
                    marker=path_style.marker,
                    facecolors=path_style.colour,
                    edgecolors="white",
                    linewidths=0.4,
                    alpha=0.72,
                    zorder=3,
                )
                mean_row = summary.loc[summary["method_family"] == method]
                mean_value = _single_value(
                    mean_row,
                    mean_column,
                    f"hard-impact mean {method}/{value_column}",
                )
                axis.scatter(
                    [mean_value],
                    [y_position],
                    s=34,
                    marker="D",
                    facecolors="white",
                    edgecolors=path_style.colour,
                    linewidths=1.1,
                    zorder=4,
                )
            elif method == METHOD_BCE and value_column == "baseline_miss_recovery":
                axis.text(
                    0.5,
                    y_position,
                    "n. a.",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=NEUTRAL_DARK,
                )
            else:
                raise RuntimeError(
                    f"Unexpected missing hard-impact values: {method}/{value_column}"
                )
        axis.set_xlim(0, 1)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.xaxis.set_major_formatter(decimal_comma(2))
        axis.set_yticks(
            y_positions,
            [PATH_STYLES[method].label for method in METHOD_ORDER],
        )
        axis.tick_params(axis="y", labelleft=index == 0)
        axis.tick_params(axis="y", pad=1.5)
        axis.set_title(title, pad=2.5)
        panel_label(axis, PANEL_LETTERS[index])
    axes[0].set_ylim(y_positions[-1] + 0.24, y_positions[0] - 0.24)
    return _save_figure(
        fig,
        output,
        "ch5_f5_hard_impact_profile_k50",
    )


def _render_replacement_map(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    events = store.csv(
        "ch5_replacement_events",
        [
            "seed",
            "row_index",
            "direction",
            "Class",
            "Amount",
            "log1p_amount",
            "p_fraud",
            "q90_fraud_flag",
            "bce_rank",
            "amount_gain_rank",
        ],
    )
    direction_specs = {
        "added_by_amount_gain": (
            "In Amount-Gain-Top-50 aufgenommen",
            PATH_STYLES[METHOD_AMOUNT_GAIN].colour,
        ),
        "removed_from_bce": (
            "Aus BCE-Top-50-Auswahl entfernt",
            PATH_STYLES[METHOD_BCE].colour,
        ),
    }
    if set(events["direction"]) != set(direction_specs):
        raise RuntimeError("Replacement-event directions are incomplete.")
    if set(events["Class"].astype(int)) != {0, 1}:
        raise RuntimeError("Replacement-event class encoding is incomplete.")
    q90_mask = _as_bool(events["q90_fraud_flag"])
    if (q90_mask & (events["Class"].astype(int) != 1)).any():
        raise RuntimeError("q90 replacement events must be Fraud cases.")
    counts = events.groupby(["seed", "direction"], sort=False).size().unstack()
    if not (
        counts["added_by_amount_gain"] == counts["removed_from_bce"]
    ).all():
        raise RuntimeError("Replacement-event directions are not balanced by seed.")

    y_values = events["log1p_amount"].to_numpy(float)
    y_pad = max(0.10, float(np.ptp(y_values)) * 0.055)
    x_limits = (-0.01, 1.02)
    y_limits = (
        float(np.min(y_values)) - y_pad,
        float(np.max(y_values)) + y_pad,
    )
    y_tick_step = max(1.0, float(math.ceil(y_limits[1] / 6.0)))
    y_tick_start = math.ceil(y_limits[0] / y_tick_step) * y_tick_step
    y_ticks = np.arange(
        y_tick_start,
        y_limits[1] + np.finfo(float).eps,
        y_tick_step,
    )

    fig, axis = plt.subplots(
        1,
        1,
        figsize=_figure_size(width_mm, 82.0),
    )
    fig.subplots_adjust(left=0.105, right=0.64, bottom=0.15, top=0.96)
    style_axis(axis)
    draw_order = ("removed_from_bce", "added_by_amount_gain")
    for direction in draw_order:
        _, colour = direction_specs[direction]
        rows = events.loc[events["direction"] == direction]
        legitimate = rows.loc[rows["Class"].astype(int) == 0]
        fraud = rows.loc[
            (rows["Class"].astype(int) == 1)
            & ~_as_bool(rows["q90_fraud_flag"])
        ]
        axis.scatter(
            legitimate["p_fraud"],
            legitimate["log1p_amount"],
            s=11,
            marker="o",
            facecolors="none",
            edgecolors=colour,
            linewidths=0.65,
            alpha=0.68,
            zorder=2,
            label=f"{direction}:legitimate",
        )
        axis.scatter(
            fraud["p_fraud"],
            fraud["log1p_amount"],
            s=11,
            marker="o",
            facecolors=colour,
            edgecolors=colour,
            linewidths=0.4,
            alpha=0.64,
            zorder=3,
            label=f"{direction}:fraud-non-q90",
        )

    for direction in draw_order:
        _, colour = direction_specs[direction]
        rows = events.loc[events["direction"] == direction]
        q90 = rows.loc[
            (rows["Class"].astype(int) == 1)
            & _as_bool(rows["q90_fraud_flag"])
        ]
        axis.scatter(
            q90["p_fraud"],
            q90["log1p_amount"],
            s=11,
            marker="o",
            facecolors=colour,
            edgecolors=colour,
            linewidths=0.4,
            alpha=0.70,
            zorder=4,
            label=f"{direction}:q90",
        )
        axis.scatter(
            q90["p_fraud"],
            q90["log1p_amount"],
            s=21,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.85,
            alpha=0.95,
            zorder=5,
            label=f"{direction}:q90-ring",
        )

    direction_counts = events["direction"].value_counts()
    direction_labels = {
        "added_by_amount_gain": "In Amount-Gain-Top-50\naufgenommen",
        "removed_from_bce": "Aus BCE-Top-50-Auswahl\nentfernt",
    }
    direction_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=direction_specs[direction][1],
            markeredgecolor=direction_specs[direction][1],
            markersize=4.3,
            label=(
                f"{direction_labels[direction]} "
                f"(n={int(direction_counts[direction])})"
            ),
        )
        for direction in ("added_by_amount_gain", "removed_from_bce")
    ]
    direction_legend = axis.legend(
        handles=direction_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        handletextpad=0.5,
        borderaxespad=0.0,
        title="Auswahlveränderung",
    )
    direction_legend.get_title().set_fontsize(7.2)
    axis.add_artist(direction_legend)

    neutral_fill = "#8C8C8C"
    type_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=neutral_fill,
            markeredgecolor=neutral_fill,
            markersize=4.3,
            label="Fraud",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=neutral_fill,
            markersize=4.3,
            label="legitim",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=neutral_fill,
            markeredgecolor="black",
            markeredgewidth=1.15,
            markersize=5.2,
            label="q90-Fraud",
        ),
    ]
    type_legend = axis.legend(
        handles=type_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.43),
        frameon=False,
        handletextpad=0.5,
        borderaxespad=0.0,
        title="Falltyp",
    )
    type_legend.get_title().set_fontsize(7.2)

    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticks(y_ticks)
    axis.xaxis.set_major_formatter(decimal_comma(2))
    axis.yaxis.set_major_formatter(decimal_comma(1))
    axis.set_xlabel("BCE-Fraud-Basisscore")
    axis.set_ylabel("log(1 + Amount)")
    return _save_figure(
        fig,
        output,
        "ch5_f6_replacement_case_map_k50",
    )


def _render_heatmap(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    values = store.csv(
        "app_seed_budget_delta_heatmap",
        [
            "seed",
            "target_budget",
            "delta_plr_vs_bce",
            "delta_fraud_at_k_vs_bce",
            "seed_order",
            "budget_order",
        ],
    )
    if len(values) != len(store.seeds) * len(store.budgets):
        raise RuntimeError("Heatmap input is not the exact seed/budget grid.")
    panels = (
        (
            "delta_plr_vs_bce",
            "Amount-Gain: Δ PLR gegenüber BCE",
            3,
            "Δ PLR",
        ),
        (
            "delta_fraud_at_k_vs_bce",
            "Amount-Gain: Δ Fraud@k gegenüber BCE",
            0,
            "Δ Fraud@k",
        ),
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=_figure_size(width_mm, 101.0),
        layout="constrained",
    )
    for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
        metric, title, decimals, colourbar_label = panel
        matrix = (
            values.pivot(
                index="seed",
                columns="target_budget",
                values=metric,
            )
            .reindex(index=store.seeds, columns=store.budgets)
        )
        if matrix.shape != (5, 7) or matrix.isna().any().any():
            raise RuntimeError(f"Heatmap matrix for {metric} is not exactly 5 x 7.")
        numeric = matrix.to_numpy(float)
        limit = max(float(np.max(np.abs(numeric))), 1e-12)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = axis.imshow(
            numeric,
            cmap="coolwarm",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(
            np.arange(len(store.budgets)),
            [str(budget) for budget in store.budgets],
        )
        axis.set_yticks(
            np.arange(len(store.seeds)),
            [str(seed) for seed in store.seeds],
        )
        axis.set_xlabel("Untersuchungsbudget k")
        axis.set_ylabel("Seed")
        axis.set_title(title, pad=4.0)
        axis.tick_params(width=0.65, length=2.5)
        for row in range(numeric.shape[0]):
            for column in range(numeric.shape[1]):
                value = numeric[row, column]
                relative = abs(value) / limit
                text_colour = "white" if relative >= 0.62 else "black"
                label = format_german(value, decimals)
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=text_colour,
                )
        colourbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
        colourbar.set_ticks(np.linspace(-limit, limit, 5))
        colourbar.ax.yaxis.set_major_formatter(decimal_comma(decimals))
        colourbar.ax.tick_params(labelsize=7.0, width=0.6, length=2.5)
        colourbar.set_label(colourbar_label, fontsize=7.5)
        panel_label(axis, PANEL_LETTERS[index])
    return _save_figure(fig, output, "app_f1_seed_budget_delta_heatmap")


def _render_tie_intervals(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    intervals = store.csv(
        "app_exact_tie_intervals",
        [
            "seed",
            "target_budget",
            "method_family",
            "fraud_at_k_min",
            "fraud_at_k_actual",
            "fraud_at_k_max",
            "plr_at_k_min",
            "plr_at_k_actual",
            "plr_at_k_max",
            "interval_interpretation",
        ],
    )
    budgets = (20, 50, 100)
    methods = (METHOD_P_ONLY, METHOD_AMOUNT_GAIN)
    expected = {
        (seed, budget, method)
        for seed in store.seeds
        for budget in budgets
        for method in methods
    }
    observed = set(
        intervals[["seed", "target_budget", "method_family"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected:
        raise RuntimeError("Exact-tie interval grid is incomplete.")
    categories = [(budget, method) for budget in budgets for method in methods]
    category_labels = [
        f"k={budget} · {PATH_STYLES[method].label}"
        for budget, method in categories
    ]
    offsets = np.linspace(-0.16, 0.16, len(store.seeds))
    seed_offsets = dict(zip(store.seeds, offsets, strict=True))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=_figure_size(width_mm, 94.0),
        sharey=True,
        layout="constrained",
    )
    panels = (
        (
            axes[0],
            "plr_at_k_min",
            "plr_at_k_actual",
            "plr_at_k_max",
            "PLR@k (Fraud-Amount-Proxy-Abdeckung)",
        ),
        (
            axes[1],
            "fraud_at_k_min",
            "fraud_at_k_actual",
            "fraud_at_k_max",
            "Fraud@k",
        ),
    )
    for panel_index, panel in enumerate(panels):
        axis, minimum_column, actual_column, maximum_column, title = panel
        style_axis(axis)
        for category_position, (budget, method) in enumerate(categories):
            path_style = PATH_STYLES[method]
            rows = intervals.loc[
                (intervals["target_budget"].astype(int) == budget)
                & (intervals["method_family"] == method)
            ]
            for row in rows.itertuples(index=False):
                y = category_position + seed_offsets[int(row.seed)]
                minimum = float(getattr(row, minimum_column))
                actual = float(getattr(row, actual_column))
                maximum = float(getattr(row, maximum_column))
                if actual < minimum - 1e-12 or actual > maximum + 1e-12:
                    raise RuntimeError("Actual value lies outside exact tie interval.")
                axis.hlines(
                    y,
                    minimum,
                    maximum,
                    color=path_style.colour,
                    linewidth=0.75,
                    alpha=0.68,
                    zorder=2,
                )
                axis.vlines(
                    [minimum, maximum],
                    y - 0.022,
                    y + 0.022,
                    color=path_style.colour,
                    linewidth=0.55,
                    alpha=0.68,
                    zorder=2,
                )
                axis.plot(
                    [actual],
                    [y],
                    marker=path_style.marker,
                    markersize=3.2,
                    markerfacecolor="white",
                    markeredgecolor=path_style.colour,
                    markeredgewidth=0.7,
                    linestyle="none",
                    zorder=3,
                )
        axis.set_yticks(np.arange(len(categories)), category_labels)
        if minimum_column.startswith("plr"):
            axis.set_title(
                "PLR@k\n(Fraud-Amount-Proxy-Abdeckung)",
                pad=4.0,
            )
        else:
            axis.set_title(title, pad=4.0)
        if minimum_column.startswith("plr"):
            axis.set_xlim(0, 1)
            axis.xaxis.set_major_formatter(decimal_comma(2))
        else:
            maximum = float(intervals[maximum_column].max())
            tick_upper = max(10.0, math.ceil(maximum / 10.0) * 10.0)
            axis.set_xlim(0, tick_upper * 1.04)
            axis.set_xticks(np.linspace(0.0, tick_upper, 5))
            axis.xaxis.set_major_formatter(decimal_comma(0))
        panel_label(axis, PANEL_LETTERS[panel_index])
    axes[0].invert_yaxis()
    for separator in (1.5, 3.5):
        for axis in axes:
            axis.axhline(
                separator,
                color="#BFBFBF",
                linewidth=0.5,
                zorder=0,
            )
    fig.legend(
        handles=method_legend_handles(methods),
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    return _save_figure(fig, output, "app_f2_exact_tie_bound_intervals")


_SEED_LABEL_OFFSETS = {
    (20, 42): (-22, 8),
    (20, 7): (-8, -12),
    (20, 13): (-4, -13),
    (20, 123): (-23, 3),
    (20, 202): (-23, 7),
    (50, 42): (-28, 8),
    (50, 7): (-8, 10),
    (50, 13): (6, 5),
    (50, 123): (-24, -4),
    (50, 202): (-25, -13),
    (100, 42): (5, 8),
    (100, 7): (5, 7),
    (100, 13): (5, -12),
    (100, 123): (-25, -5),
    (100, 202): (-32, -8),
}


def _resolve_seed_label_overlaps(
    fig: Figure,
    seed_labels: dict[int, list[Text]],
) -> None:
    """Deterministically move dense seed labels to an unoccupied nearby slot."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fallback_offsets = (
        (7, 7),
        (-7, 7),
        (7, -8),
        (-7, -8),
        (8, 17),
        (-8, 17),
        (8, -18),
        (-8, -18),
        (0, 25),
        (0, -25),
        (21, 0),
        (-21, 0),
        (23, 18),
        (-23, 18),
        (23, -19),
        (-23, -19),
        (36, 0),
        (-36, 0),
    )
    for budget, labels in seed_labels.items():
        accepted = []
        for label in labels:
            preferred = tuple(float(value) for value in label.get_position())
            candidates = (preferred, *fallback_offsets)
            chosen = None
            for candidate_index, offset in enumerate(dict.fromkeys(candidates)):
                label.set_position(offset)
                if candidate_index:
                    label.set_horizontalalignment(
                        "left"
                        if offset[0] > 0
                        else "right"
                        if offset[0] < 0
                        else "center"
                    )
                box = label.get_window_extent(renderer=renderer).expanded(
                    1.03,
                    1.08,
                )
                within_canvas = (
                    box.x0 >= fig.bbox.x0
                    and box.y0 >= fig.bbox.y0
                    and box.x1 <= fig.bbox.x1
                    and box.y1 <= fig.bbox.y1
                )
                if within_canvas and not any(
                    box.overlaps(existing) for existing in accepted
                ):
                    chosen = box
                    break
            if chosen is None:
                raise RuntimeError(
                    f"Unable to place seed label without overlap in "
                    f"candidate-pool panel k={budget}."
                )
            accepted.append(chosen)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for budget, labels in seed_labels.items():
        boxes = [
            label.get_window_extent(renderer=renderer).expanded(1.03, 1.08)
            for label in labels
        ]
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                if boxes[left].overlaps(boxes[right]):
                    raise RuntimeError(
                        f"Seed-label overlap in candidate-pool panel k={budget}."
                    )


def _render_pool_ceiling(
    store: FigureStore,
    output: Path,
    width_mm: float,
) -> list[Path]:
    values = store.csv(
        "app_candidate_pool_ceiling",
        [
            "seed",
            "target_budget",
            "method_family",
            "fraud_ceiling_utilization",
            "plr_ceiling_utilization",
            "q90_ceiling_utilization",
            "candidate_pool_fraud_case_coverage",
            "candidate_pool_fraud_amount_coverage",
            "ceiling_interpretation",
        ],
    )
    budgets = (20, 50, 100)
    expected = {
        (seed, budget) for seed in store.seeds for budget in budgets
    }
    observed = set(
        values[["seed", "target_budget"]].itertuples(index=False, name=None)
    )
    if observed != expected or set(values["method_family"]) != {
        METHOD_AMOUNT_GAIN
    }:
        raise RuntimeError("Candidate-pool ceiling grid is incomplete.")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=_figure_size(width_mm, 65.0),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    path_style = PATH_STYLES[METHOD_AMOUNT_GAIN]
    seed_labels: dict[int, list[Text]] = {}
    for index, (axis, budget) in enumerate(zip(axes, budgets, strict=True)):
        style_axis(axis)
        rows = values.loc[
            values["target_budget"].astype(int) == budget
        ].copy()
        axis.scatter(
            rows["fraud_ceiling_utilization"],
            rows["plr_ceiling_utilization"],
            s=27,
            marker=path_style.marker,
            facecolors=path_style.colour,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.82,
            zorder=3,
        )
        for row in rows.itertuples(index=False):
            seed = int(row.seed)
            offset = _SEED_LABEL_OFFSETS.get(
                (budget, seed),
                (5 if seed % 2 else -18, 6 if seed % 3 else -8),
            )
            annotation = axis.annotate(
                str(seed),
                (
                    float(row.fraud_ceiling_utilization),
                    float(row.plr_ceiling_utilization),
                ),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.8,
                ha="left",
                va="center",
                color="black",
                zorder=5,
            )
            seed_labels.setdefault(budget, []).append(annotation)
        if budget == 50:
            highlighted = rows.loc[rows["seed"].astype(int) == 13]
            ring_x = _single_value(
                highlighted,
                "fraud_ceiling_utilization",
                "Seed 13 candidate-pool ceiling",
            )
            ring_y = _single_value(
                highlighted,
                "plr_ceiling_utilization",
                "Seed 13 candidate-pool ceiling",
            )
            axis.scatter(
                [ring_x],
                [ring_y],
                s=48,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )
        axis.set_xlim(0, 1.04)
        axis.set_ylim(0, 1.04)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.xaxis.set_major_formatter(decimal_comma(2))
        axis.yaxis.set_major_formatter(decimal_comma(2))
        axis.set_xlabel("Fraud-Ceiling-Nutzung")
        axis.set_title(f"k = {budget}", pad=4.0)
        axis.set_aspect("equal", adjustable="box")
        panel_label(axis, PANEL_LETTERS[index])
    axes[0].set_ylabel("PLR-Ceiling-Nutzung")
    _resolve_seed_label_overlaps(fig, seed_labels)
    return _save_figure(
        fig,
        output,
        "app_f3_candidate_pool_ceiling_utilization",
    )


def _global_metric_caption_metadata(
    store: FigureStore,
) -> list[dict[str, object]]:
    """Select full-order aggregate k=50 metrics for F4 caption metadata."""

    metrics = store.csv(
        "ch5_global_metrics_summary",
        [
            "target_budget",
            "scope",
            "scope_label",
            "method_family",
            "path_id",
            "path_label",
            "metric",
            "score_interpretation",
            "value_n",
            "value_mean",
            "value_sd",
        ],
    )
    selected = metrics.loc[
        (metrics["target_budget"].astype(int) == 50)
        & (metrics["scope"] == "full_order")
    ].copy()
    selected = selected.loc[
        selected["metric"].isin({"roc_auc", "average_precision", "brier"})
    ]
    brier = selected.loc[selected["metric"] == "brier"]
    if (
        len(brier) != 1
        or set(brier["method_family"]) != {METHOD_BCE}
        or set(brier["scope"]) != {"full_order"}
    ):
        raise RuntimeError("B4 caption metadata has invalid Brier scope.")
    method_order = {
        method: index for index, method in enumerate(METHOD_ORDER)
    }
    metric_order = {"roc_auc": 0, "average_precision": 1, "brier": 2}
    selected["__method"] = selected["method_family"].map(method_order)
    selected["__metric"] = selected["metric"].map(metric_order)
    selected = selected.sort_values(
        ["__method", "__metric"], kind="mergesort"
    )
    records: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        records.append(
            {
                "target_budget": 50,
                "scope": str(row.scope),
                "scope_label": str(row.scope_label),
                "method_family": str(row.method_family),
                "path_id": str(row.path_id),
                "path_label": str(row.path_label),
                "metric": str(row.metric),
                "mean": float(row.value_mean),
                "sample_sd": float(row.value_sd),
                "n": int(row.value_n),
                "score_interpretation": str(row.score_interpretation),
            }
        )
    return records


def _configure_engineering_colourbar_ticks(
    figure: Figure,
    colourbar: Any,
    heatmap_axes: tuple[Any, Any],
    *,
    limit: float,
    minimum_decimals: int,
    metric_panel: str,
) -> None:
    """Install and validate finite, domain-contained engineering ticks."""

    if not math.isfinite(limit) or limit <= 0.0:
        raise RuntimeError(
            f"Engineering {metric_panel} colour-bar limit is invalid: {limit}."
        )
    scale_decimals = max(
        0,
        int(math.ceil(-math.log10(limit))) + 1,
    )
    decimals = min(max(minimum_decimals, scale_decimals), 6)
    candidates = (-limit, -limit / 2.0, 0.0, limit / 2.0, limit)
    tick_by_rounded_value: dict[float, float] = {}
    for candidate in candidates:
        rounded = round(float(candidate), decimals)
        if rounded == 0.0:
            tick_by_rounded_value[0.0] = 0.0
        else:
            tick_by_rounded_value.setdefault(rounded, float(candidate))
    ticks = tuple(sorted(tick_by_rounded_value.values()))
    if not ticks or 0.0 not in ticks:
        raise RuntimeError(
            f"Engineering {metric_panel} colour-bar ticks omit zero."
        )
    if not all(
        math.isfinite(tick) and -limit <= tick <= limit for tick in ticks
    ):
        raise RuntimeError(
            f"Engineering {metric_panel} colour-bar ticks leave their domain."
        )

    def format_tick(value: float, _position: float) -> str:
        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"Engineering {metric_panel} colour-bar tick is non-finite."
            )
        return format_german(float(value), decimals)

    colourbar.locator = FixedLocator(ticks)
    colourbar.formatter = FuncFormatter(format_tick)
    colourbar.update_ticks()
    colourbar.ax.yaxis.get_offset_text().set_visible(False)
    figure.canvas.draw()

    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    current_ticks = colourbar.ax.get_yticks()
    labels = [
        label
        for label in colourbar.ax.get_yticklabels()
        if label.get_visible()
    ]
    if len(labels) != len(current_ticks):
        raise RuntimeError(
            f"Engineering {metric_panel} colour-bar tick inventory differs."
        )
    adjacent_boxes = [
        axis.get_window_extent(renderer=renderer) for axis in heatmap_axes
    ]
    panel_title_boxes = [
        axis.title.get_window_extent(renderer=renderer)
        for axis in heatmap_axes
        if axis.title.get_visible() and axis.title.get_text().strip()
    ]
    for tick, label in zip(current_ticks, labels, strict=True):
        text = label.get_text()
        display = label.get_transform().transform(label.get_position())
        bbox = label.get_window_extent(renderer=renderer)
        coordinates = (
            float(display[0]),
            float(display[1]),
            bbox.x0,
            bbox.y0,
            bbox.x1,
            bbox.y1,
        )
        normalized_tick = round(float(tick), decimals)
        has_minus_sign = text.startswith(("-", "−"))
        invalid_sign = (
            (normalized_tick == 0.0 and has_minus_sign)
            or (normalized_tick < 0.0 and not has_minus_sign)
            or (normalized_tick > 0.0 and has_minus_sign)
        )
        invalid = (
            not math.isfinite(float(tick))
            or not -limit <= float(tick) <= limit
            or not text
            or invalid_sign
            or not all(math.isfinite(value) for value in coordinates)
            or bbox.x0 < canvas.x0 - 2.0
            or bbox.y0 < canvas.y0 - 2.0
            or bbox.x1 > canvas.x1 + 2.0
            or bbox.y1 > canvas.y1 + 2.0
            or any(bbox.overlaps(adjacent) for adjacent in adjacent_boxes)
            or any(bbox.overlaps(title) for title in panel_title_boxes)
        )
        if invalid:
            raise RuntimeError(
                f"Engineering {metric_panel} colour-bar tick is invalid: "
                f"value={float(tick)!r}, text={text!r}, "
                f"transform={type(label.get_transform()).__name__}, "
                f"bbox={coordinates[2:]!r}."
            )


def _engineering_heatmap_figure(
    store: FigureStore,
    width_mm: float,
) -> Figure:
    """Build the shared engineering delta heatmap from prepared values."""

    source = (
        "engineering/figures/engineering_seed_budget_delta_heatmap.csv"
    )
    required = (
        "profile",
        "evidence_classification",
        "data_source_kind",
        "evidence_statement",
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
    )
    values = store.csv(source, required)
    metadata = {
        "profile": store.profile,
        "evidence_classification": store.evidence_classification,
        "data_source_kind": store.data_source_kind,
        "evidence_statement": store.evidence_statement,
    }
    for column, expected_value in metadata.items():
        observed = values[column].dropna().astype(str).unique().tolist()
        if observed != [str(expected_value)]:
            raise RuntimeError(
                f"Engineering heatmap {column} does not match its manifest."
            )

    methods = (METHOD_P_ONLY, METHOD_AMOUNT_GAIN, METHOD_FIXED)
    path_ids = {
        METHOD_P_ONLY: "p_only",
        METHOD_AMOUNT_GAIN: "amount_gain",
        METHOD_FIXED: "fixed_reference",
    }
    try:
        values = values.copy()
        values["seed"] = values["seed"].astype(int)
        values["target_budget"] = values["target_budget"].astype(int)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Engineering heatmap seed/budget values are invalid."
        ) from exc
    expected_grid = {
        (seed, budget, method)
        for seed in store.seeds
        for budget in store.budgets
        for method in methods
    }
    keys = ["seed", "target_budget", "method_family"]
    if values.duplicated(keys).any():
        raise RuntimeError("Engineering heatmap has duplicate grid cells.")
    observed_grid = set(
        values.loc[:, keys].itertuples(index=False, name=None)
    )
    if observed_grid != expected_grid:
        raise RuntimeError(
            "Engineering heatmap grid is incomplete or contains unsupported values."
        )
    expected_path_ids = values["method_family"].map(path_ids)
    if expected_path_ids.isna().any() or not expected_path_ids.equals(
        values["path_id"].astype(str)
    ):
        raise RuntimeError("Engineering heatmap contains an unknown path.")
    numeric_columns = (
        "plr_at_k",
        "fraud_at_k",
        "bce_plr_at_k",
        "bce_fraud_at_k",
        "delta_plr_vs_bce",
        "delta_fraud_at_k_vs_bce",
    )
    try:
        numeric_values = values.loc[:, numeric_columns].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Engineering heatmap contains invalid numeric values."
        ) from exc
    if not np.isfinite(numeric_values).all():
        raise RuntimeError(
            "Engineering heatmap contains non-finite numeric values."
        )

    row_keys = [(seed, method) for seed in store.seeds for method in methods]
    row_labels = {
        METHOD_P_ONLY: "p-only",
        METHOD_AMOUNT_GAIN: "Amount-Gain",
        METHOD_FIXED: "fixed reference",
    }
    panels = (
        ("delta_plr_vs_bce", "(a) PLR-Delta", 3),
        ("delta_fraud_at_k_vs_bce", "(b) Fraud@k-Delta", 1),
    )
    base_height = 96.0 if len(store.seeds) == 1 else 126.0
    fig = plt.figure(
        figsize=_figure_size(width_mm, base_height),
    )
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=(1.0, 0.045, 0.13, 1.0, 0.045),
        height_ratios=(5.6, 1.0),
        left=0.245,
        right=0.89,
        bottom=0.04,
        top=0.79,
        hspace=0.3,
        wspace=0.12,
    )
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 3]))
    colourbar_axes = (
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 4]),
    )
    footer_axis = fig.add_subplot(grid[1, :])
    footer_axis.set_axis_off()
    footer_axis.set_frame_on(False)
    footer_axis.set_xticks([])
    footer_axis.set_yticks([])
    for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
        metric, title, decimals = panel
        matrix = np.empty((len(row_keys), len(store.budgets)), dtype=float)
        for row_index, (seed, method) in enumerate(row_keys):
            selected = values.loc[
                (values["seed"] == seed)
                & (values["method_family"] == method)
            ].set_index("target_budget")
            matrix[row_index, :] = selected.loc[
                list(store.budgets), metric
            ].to_numpy(float)
        limit = max(float(np.max(np.abs(matrix))), 1e-12)
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_xticks(
            np.arange(len(store.budgets)),
            [str(budget) for budget in store.budgets],
        )
        axis.set_yticks(
            np.arange(len(row_keys)),
            (
                [
                    f"{seed} · {row_labels[method]}"
                    for seed, method in row_keys
                ]
                if index == 0
                else []
            ),
        )
        axis.set_xlabel("Target budget k")
        axis.set_title(title, pad=4.0)
        axis.tick_params(width=0.65, length=2.5)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    format_german(value, decimals, signed=True),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=(
                        "white"
                        if abs(value) / limit >= 0.62
                        else "black"
                    ),
                )
        colourbar = fig.colorbar(
            image,
            cax=colourbar_axes[index],
        )
        colourbar.ax.tick_params(
            labelsize=6.4,
            width=0.6,
            length=2.5,
            pad=1.0,
        )
        _configure_engineering_colourbar_ticks(
            fig,
            colourbar,
            axes,
            limit=limit,
            minimum_decimals=decimals,
            metric_panel=("PLR" if index == 0 else "Fraud@k"),
        )

    fig.suptitle(
        "Engineering-Heatmap der Seed-Budget-Deltas gegenüber BCE",
        fontsize=9.0,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.875,
        (
            f"Profile: {store.profile} | Evidence: "
            f"{store.evidence_classification}"
        ),
        ha="center",
        va="center",
        fontsize=7.2,
    )
    statement = str(store.evidence_statement)
    boundary = str(store.comparability_boundary)
    suffix = f"; {boundary}"
    if not statement.endswith(suffix):
        raise RuntimeError(
            "Engineering figure evidence statement is incomplete."
        )
    classification = statement.removesuffix(suffix).replace("; ", " · ")
    footer_axis.text(
        0.5,
        0.68,
        f"{store.profile} · {classification}",
        transform=footer_axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=NEUTRAL_DARK,
    )
    footer_axis.text(
        0.5,
        0.25,
        boundary,
        transform=footer_axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.4,
        color=NEUTRAL_DARK,
    )
    return fig


def _render_engineering_figures(
    store: FigureStore,
    repository_root: Path,
    requested_output: Path,
    width_mm: float,
    force: bool,
    event_sink: _EventSink | None,
    started: float,
) -> dict[str, Any]:
    output = prepare_output_directory(
        repository_root,
        requested_output,
        force=force,
    )
    role_output = output / "engineering"
    role_output.mkdir(parents=True, exist_ok=False)
    stem = "engineering_f1_seed_budget_delta_heatmap"
    figure = _engineering_heatmap_figure(store, width_mm)
    files = _save_figure(
        figure,
        role_output,
        stem,
        subject="Engineering presentation render; not thesis evidence",
        description=str(store.evidence_statement),
    )
    expected_files = {
        role_output / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")
    }
    if set(files) != expected_files:
        raise RuntimeError("Engineering figure output inventory drift.")
    manifest_path = output / "FIGURE_RENDER_MANIFEST.json"
    manifest = {
        "schema": "fraud_detection.engineering_figure_render.v1",
        "status": "PASS",
        "presentation_role": "engineering",
        "profile": store.profile,
        "evidence_classification": store.evidence_classification,
        "data_source_kind": store.data_source_kind,
        "evidence_statement": store.evidence_statement,
        "comparability_boundary": store.comparability_boundary,
        "selected_catalog_artifact_ids": list(
            store.selected_catalog_artifact_ids
        ),
        "inputs": sorted(store.read_paths),
        "outputs": file_inventory(output, files),
        "rendered_stems": [stem],
        "logical_figure_count": 1,
        "rendered_file_count": 3,
        "formats": ["pdf", "svg", "png"],
        "width_mm": float(width_mm),
        "png_dpi": PNG_DPI,
        "layout_validation": "PASS",
        "significance_annotation_rendered": False,
        "winner_ranking_rendered": False,
        "model_fit_performed": False,
        "model_scoring_performed": False,
        "data_derivation_performed": False,
    }
    write_json(manifest_path, manifest)
    _emit_status(
        event_sink,
        "PASS",
        "render-figures-complete",
        figures=1,
        files=3,
        manifest=manifest_path,
        elapsed_seconds=time.perf_counter() - started,
    )
    return manifest


def render(
    repository_root: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    width_mm: float = DEFAULT_WIDTH_MM,
    force: bool = False,
    event_sink: _EventSink | None = None,
) -> dict[str, Any]:
    """Render all selected R7 figures without empirical recomputation."""

    started = time.perf_counter()
    _emit_status(
        event_sink,
        "INFO",
        "render-figures-start",
        data_dir=data_dir,
        output_dir=output_dir,
        width_mm=width_mm,
        utc=_status_utc_now(),
    )
    if not math.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("--width-mm must be a positive finite number.")
    requested_output = require_generated_path(repository_root, output_dir)
    resolved_data = (
        data_dir if data_dir.is_absolute() else repository_root / data_dir
    ).resolve()
    if (
        requested_output == resolved_data
        or requested_output in resolved_data.parents
        or resolved_data in requested_output.parents
    ):
        raise ValueError(
            "Figure output and presentation-data directories must be disjoint."
        )
    configure_presentation_style()
    store = FigureStore(resolved_data)
    if store.presentation_role == "engineering":
        _emit_status(
            event_sink,
            "PASS",
            "render-figures-input-validation",
            figures_expected=1,
            formats_per_figure=3,
            files_expected=3,
        )
        return _render_engineering_figures(
            store,
            repository_root,
            requested_output,
            width_mm,
            force,
            event_sink,
            started,
        )
    _emit_status(
        event_sink,
        "PASS",
        "render-figures-input-validation",
        figures_expected=EXPECTED_FIGURE_COUNT,
        formats_per_figure=3,
        files_expected=EXPECTED_FIGURE_FILE_COUNT,
    )
    output = prepare_output_directory(
        repository_root,
        requested_output,
        force=force,
    )
    main_output = output / "main"
    optional_output = output / "optional"
    appendix_output = output / "appendix"
    main_output.mkdir(parents=True, exist_ok=False)
    optional_output.mkdir(parents=True, exist_ok=False)
    appendix_output.mkdir(parents=True, exist_ok=False)

    files: list[Path] = []
    _record_figure_completion(
        files,
        _render_tradeoff(store, main_output, width_mm),
        completed=1,
        stem=MAIN_STEMS[0],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_budget_policy(store, main_output, width_mm),
        completed=2,
        stem=MAIN_STEMS[1],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_depth(store, main_output, width_mm),
        completed=3,
        stem=MAIN_STEMS[2],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_global_pool_curves(store, main_output, width_mm),
        completed=4,
        stem=MAIN_STEMS[3],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_hard_impact(store, optional_output, width_mm),
        completed=5,
        stem=OPTIONAL_STEMS[0],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_replacement_map(store, main_output, width_mm),
        completed=6,
        stem=MAIN_STEMS[4],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_heatmap(store, appendix_output, width_mm),
        completed=7,
        stem=APPENDIX_STEMS[0],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_tie_intervals(store, appendix_output, width_mm),
        completed=8,
        stem=APPENDIX_STEMS[1],
        event_sink=event_sink,
    )
    _record_figure_completion(
        files,
        _render_pool_ceiling(store, appendix_output, width_mm),
        completed=9,
        stem=APPENDIX_STEMS[2],
        event_sink=event_sink,
    )
    b4_caption_metrics = _global_metric_caption_metadata(store)

    rendered_stems = sorted({path.stem for path in files})
    expected_stems = sorted(
        (*MAIN_STEMS, *OPTIONAL_STEMS, *APPENDIX_STEMS)
    )
    if rendered_stems != expected_stems:
        raise RuntimeError("Rendered figure selection differs from R7B.")
    metric_means = {
        metric: {
            str(row["path_label"]): float(row["mean"])
            for row in b4_caption_metrics
            if row["metric"] == metric
        }
        for metric in ("roc_auc", "average_precision")
    }
    bce_brier = next(
        float(row["mean"])
        for row in b4_caption_metrics
        if row["metric"] == "brier"
    )
    manifest_path = output / "FIGURE_RENDER_MANIFEST.json"
    manifest = {
        "schema": "fraud_detection.chapter5_figure_render.r7b.v1",
        "status": "PASS",
        "inputs": sorted(store.read_paths),
        "outputs": file_inventory(output, files),
        "rendered_stems": rendered_stems,
        "main_text_stems": list(MAIN_STEMS),
        "optional_main_text_stems": list(OPTIONAL_STEMS),
        "appendix_stems": list(APPENDIX_STEMS),
        "formats": ["pdf", "svg", "png"],
        "width_mm": float(width_mm),
        "png_dpi": PNG_DPI,
        "font_embedding": {
            "pdf_fonttype": 42,
            "svg_text_as_paths": True,
        },
        "layout_validation": "PASS",
        "unresolved_layout_warnings": [],
        "caption_metadata": {
            "ch5_f2_budget_policy_profile": {
                "caption_note": (
                    "Trainierte Punkte stammen aus separaten "
                    "budgetkonditionierten Modellen; die Linien verbinden "
                    "ein geordnetes Policyprofil und sind keine "
                    "Präfixauswertung eines einzelnen Rankers."
                ),
            },
            "ch5_f3_within_model_depth_k50": {
                "caption_note": (
                    "k=50 ist das Modellselektions-/Trainingsbudget; r ist "
                    "die diagnostische Auslesetiefe innerhalb der festen "
                    "abgeschlossenen Rangfolge dieses Modells."
                ),
                "budget_model": 50,
                "rank_depth_range": [1, 100],
            },
            "ch5_f4_global_roc_pr_k50": {
                "exact_aggregate_metrics": b4_caption_metrics,
                "mean_roc_auc_by_path": metric_means["roc_auc"],
                "mean_average_precision_by_path": metric_means[
                    "average_precision"
                ],
                "bce_brier": bce_brier,
                "aggregation_basis": "arithmetic mean over five seeds",
                "scope": "complete test order",
                "score_interpretation": (
                    "Ranking scores are ordinal and not Fraud probabilities."
                ),
                "metric_table_reference": "app_t2_global_metrics_by_budget",
                "legend_contains_metric_strings": False,
                "brier_scope": "BCE full order only",
            },
            "ch5_f6_replacement_case_map_k50": {
                "title": (
                    "Zusammensetzung der Top-50-Replacement-Ereignisse"
                ),
                "caption_note": (
                    "Ereignisse sind seed-spezifisch; dieselbe Transaktion "
                    "kann in mehreren Outer-Splits auftreten. Ereigniszahlen "
                    "sind keine Anzahlen eindeutiger Transaktionen. Amount "
                    "ist ein Proxy. Die Abbildung zeigt die beobachtete "
                    "Selektionszusammensetzung; sie ist weder eine gelernte "
                    "Entscheidungsgrenze noch kausal zu interpretieren."
                ),
            },
        },
        "model_fit_performed": False,
        "model_scoring_performed": False,
        "data_derivation_performed": False,
        "ranking_modified": False,
        "significance_annotation_rendered": False,
        "causal_annotation_rendered": False,
        "historical_near_tie_rendered": False,
        "cross_budget_rank_segments_rendered": False,
    }
    write_json(manifest_path, manifest)
    _emit_status(
        event_sink,
        "PASS",
        "render-figures-complete",
        figures=EXPECTED_FIGURE_COUNT,
        files=EXPECTED_FIGURE_FILE_COUNT,
        manifest=manifest_path,
        elapsed_seconds=time.perf_counter() - started,
    )
    return manifest
