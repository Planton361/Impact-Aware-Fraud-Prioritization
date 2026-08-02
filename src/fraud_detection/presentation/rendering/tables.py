"""Render the final Chapter-5 R6 tables from deterministic presentation data.

This module is deliberately a rendering layer.  It checksum-gates the exact
table frames prepared by the active presentation-data pipeline, validates their
display schemas, and writes human-readable CSV plus LaTeX fragments.  It does
not derive empirical quantities, fit models, score rows, or alter rankings.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import pandas as pd

from .. import (
    METHOD_AMOUNT_GAIN,
    METHOD_BCE,
    METHOD_FIXED,
    METHOD_P_ONLY,
    file_inventory,
    prepare_output_directory,
    require_generated_path,
    require_new_file,
    sha256_file,
    write_json,
)
from ..catalog import CANONICAL_ARTIFACT_IDS, ENGINEERING_ARTIFACT_IDS

_EventSink = Callable[[str, Mapping[str, object]], None]
CENTRAL_BUDGETS = (20, 50, 100)
PATH_ORDER = ("BCE", "p_only", "amount_gain", "fixed_reference")
TRAINED_PATH_ORDER = ("p_only", "amount_gain")
DIRECTION_ORDER = ("added_by_amount_gain", "removed_from_bce")
NA_DISPLAY = "–"
NA_TEX = r"\multicolumn{1}{c}{\textendash}"
EXPECTED_TABLE_COUNT = 9
EXPECTED_TABLE_FILE_COUNT = 18
EXPECTED_DATA_SCHEMA = "fraud_detection.chapter5_presentation_data.r6.v1"


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


@dataclass(frozen=True)
class RenderedTable:
    stem: str
    role: str
    sources: tuple[str, ...]
    display_rows: list[list[str]]
    display_header: tuple[str, ...]
    latex: str


class TableStore:
    """Checksum-gated access to deterministic presentation table frames."""

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
            raise RuntimeError(
                "Presentation-data manifest contains duplicate outputs."
            )
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
        self.read_paths: set[str] = set()
        self.seeds = tuple(int(value) for value in self.manifest.get("seeds", []))
        self.budgets = tuple(
            int(value) for value in self.manifest.get("budgets", [])
        )
        if not self.seeds or not self.budgets:
            raise RuntimeError(
                "Presentation-data manifest lacks the seed/budget grid."
            )

    def csv(self, name: str) -> pd.DataFrame:
        relative = name if "/" in name else f"tables/{name}"
        if relative not in self.hashes:
            raise RuntimeError(f"Unregistered table data rejected: {relative}")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise RuntimeError(f"Unsafe table data path rejected: {relative}")
        if not path.is_file() or sha256_file(path) != self.hashes[relative]:
            raise RuntimeError(f"Table-data checksum mismatch: {relative}")
        self.read_paths.add(relative)
        return pd.read_csv(path)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    source: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{source} lacks display columns: {missing}")


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "−": "--",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _decimal(value: object, decimals: int) -> Decimal | None:
    if pd.isna(value):
        return None
    result = Decimal(str(value)).quantize(
        Decimal(1).scaleb(-decimals),
        rounding=ROUND_HALF_UP,
    )
    return abs(result) if result == 0 else result


def _display_number(
    value: object,
    decimals: int,
    *,
    grouped: bool = False,
) -> str:
    number = _decimal(value, decimals)
    if number is None:
        return NA_DISPLAY
    rendered = (
        f"{number:,.{decimals}f}" if grouped else f"{number:.{decimals}f}"
    )
    separator_placeholder = "\uE000"
    return (
        rendered.replace(",", separator_placeholder)
        .replace(".", ",")
        .replace(separator_placeholder, ".")
    )


def _display_integer(value: object) -> str:
    if pd.isna(value):
        return NA_DISPLAY
    return f"{int(round(float(value))):d}"


def _display_mean_sd(
    mean: object,
    sd: object,
    decimals: int,
    *,
    grouped: bool = False,
) -> str:
    if pd.isna(mean):
        return NA_DISPLAY
    if pd.isna(sd):
        return _display_number(mean, decimals, grouped=grouped)
    return (
        f"{_display_number(mean, decimals, grouped=grouped)} "
        f"± {_display_number(sd, decimals, grouped=grouped)}"
    )


def _tex_number(value: object, decimals: int) -> str:
    number = _decimal(value, decimals)
    if number is None:
        return NA_TEX
    return f"{number:.{decimals}f}"


def _tex_integer(value: object) -> str:
    if pd.isna(value):
        return NA_TEX
    return f"{int(round(float(value))):d}"


def _tex_preamble(stem: str) -> list[str]:
    return [
        f"% Generated by fraud_detection.presentation.tables ({stem}); do not edit.",
        r"\begingroup",
        r"\fontsize{7.8}{9.4}\selectfont",
        r"\sisetup{",
        r"  output-decimal-marker = {,},",
        r"  group-separator = {.},",
        r"  group-minimum-digits = 4,",
        r"  detect-all",
        r"}",
    ]


def _tex_note(note: str) -> list[str]:
    return [
        r"\begin{tablenotes}[flushleft]",
        rf"\item {_latex_escape(note)}",
        r"\end{tablenotes}",
    ]


def _tex_finish() -> list[str]:
    return [r"\endgroup", ""]


def _ordered(
    frame: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    path_ids: tuple[str, ...] = PATH_ORDER,
    keys: tuple[str, ...],
) -> pd.DataFrame:
    work = frame.copy()
    helpers: list[str] = []
    translated: list[str] = []
    mappings: dict[str, dict[object, int]] = {
        "seed": {value: index for index, value in enumerate(seeds)},
        "target_budget": {
            value: index for index, value in enumerate(budgets)
        },
        "path_id": {value: index for index, value in enumerate(path_ids)},
        "direction": {
            value: index for index, value in enumerate(DIRECTION_ORDER)
        },
    }
    for key in keys:
        if key not in mappings:
            translated.append(key)
            continue
        helper = f"__{key}_order"
        values = work[key].astype(int) if key in {"seed", "target_budget"} else work[key]
        work[helper] = values.map(mappings[key])
        if work[helper].isna().any():
            unknown = sorted(work.loc[work[helper].isna(), key].astype(str).unique())
            raise RuntimeError(f"Unexpected {key} values: {unknown}")
        helpers.append(helper)
        translated.append(helper)
    return (
        work.sort_values(translated, kind="mergesort")
        .drop(columns=helpers)
        .reset_index(drop=True)
    )


def _validate_grid(
    frame: pd.DataFrame,
    *,
    source: str,
    keys: tuple[str, ...],
    expected: set[tuple[object, ...]],
) -> None:
    if frame.duplicated(list(keys)).any():
        raise RuntimeError(f"{source} has duplicate table keys: {keys}")
    observed = set(frame.loc[:, list(keys)].itertuples(index=False, name=None))
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise RuntimeError(
            f"{source} table grid mismatch: missing={missing}, extra={extra}"
        )


def _render_ch5_t1(store: TableStore) -> RenderedTable:
    source = "ch5_t1_central_topk_results.csv"
    frame = store.csv(source)
    metrics = (
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "precision_at_k",
        "recall_at_k",
    )
    required = [
        "target_budget",
        "path_id",
        "path_label",
        "row_count",
        *[
            f"{metric}_{suffix}"
            for metric in metrics
            for suffix in ("n", "mean", "sd")
        ],
    ]
    _require_columns(frame, required, source)
    budgets = tuple(value for value in CENTRAL_BUDGETS if value in store.budgets)
    expected = {
        (budget, path_id) for budget in budgets for path_id in PATH_ORDER
    }
    _validate_grid(
        frame,
        source=source,
        keys=("target_budget", "path_id"),
        expected=expected,
    )
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("target_budget", "path_id"),
    )
    for row in frame.itertuples(index=False):
        if int(row.row_count) != len(store.seeds):
            raise RuntimeError(f"{source} does not aggregate every seed.")
        for metric in metrics:
            if int(getattr(row, f"{metric}_n")) != len(store.seeds):
                raise RuntimeError(f"{source} has missing values for {metric}.")

    display_header = (
        "k",
        "Pfad",
        "PLR@k (M ± SD)",
        "Fraud@k (M ± SD)",
        "Precision@k (M ± SD)",
        "Recall@k (M ± SD)",
    )
    display_rows: list[list[str]] = []
    for row in frame.itertuples(index=False):
        display_rows.append(
            [
                str(int(row.target_budget)),
                str(row.path_label),
                _display_mean_sd(
                    row.prevented_loss_ratio_at_k_mean,
                    row.prevented_loss_ratio_at_k_sd,
                    3,
                ),
                _display_mean_sd(
                    row.frauds_at_k_mean,
                    row.frauds_at_k_sd,
                    1,
                ),
                _display_mean_sd(
                    row.precision_at_k_mean,
                    row.precision_at_k_sd,
                    3,
                ),
                _display_mean_sd(
                    row.recall_at_k_mean,
                    row.recall_at_k_sd,
                    3,
                ),
            ]
        )

    note = (
        f"M = arithmetisches Mittel, SD = Stichprobenstandardabweichung, "
        f"n={len(store.seeds)}. PLR@k bezeichnet die "
        "Fraud-Amount-Proxy-Abdeckung, nicht realisierten oder verhinderten "
        "finanziellen Verlust. p-only und Amount-Gain sind je k separate "
        "budgetkonditionierte Modelle."
    )
    lines = _tex_preamble("ch5_t1_central_topk_results")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Zentrale Top-\(k\)-Ergebnisse}",
            r"\label{tab:ch5-central-topk-results}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{2.6pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=3.1] S[table-format=2.1] "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=1.3] S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"Pfad & \multicolumn{2}{c}{PLR@\(k\)} "
                r"& \multicolumn{2}{c}{Fraud@\(k\)} "
                r"& \multicolumn{2}{c}{Precision@\(k\)} "
                r"& \multicolumn{2}{c}{Recall@\(k\)} \\"
            ),
            (
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
                r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
            ),
            r" & {M} & {SD} & {M} & {SD} & {M} & {SD} & {M} & {SD} \\",
            r"\midrule",
        ]
    )
    for budget in budgets:
        panel = frame.loc[frame["target_budget"].astype(int) == budget]
        lines.append(
            r"\multicolumn{9}{l}{\textit{\(k="
            + str(budget)
            + r"\) (\(n="
            + str(len(store.seeds))
            + r"\))}} \\"
        )
        for row in panel.itertuples(index=False):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(row.path_label),
                        _tex_number(row.prevented_loss_ratio_at_k_mean, 3),
                        _tex_number(row.prevented_loss_ratio_at_k_sd, 3),
                        _tex_number(row.frauds_at_k_mean, 1),
                        _tex_number(row.frauds_at_k_sd, 1),
                        _tex_number(row.precision_at_k_mean, 3),
                        _tex_number(row.precision_at_k_sd, 3),
                        _tex_number(row.recall_at_k_mean, 3),
                        _tex_number(row.recall_at_k_sd, 3),
                    ]
                )
                + r" \\"
            )
        if budget != budgets[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="ch5_t1_central_topk_results",
        role="main",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _tie_class_label(value: object) -> str:
    mapping = {
        "NO_CUTOFF_TIE_EFFECT": "kein Cutoff-Tie-Effekt",
        "TIE_ROBUST_BOTH": "Tie-robust",
        "TIE_SENSITIVE": "Tie-sensitiv",
    }
    text = str(value)
    if text not in mapping:
        raise RuntimeError(f"Unknown technical tie classification: {text}")
    return mapping[text]


def _render_ch5_t2(store: TableStore) -> RenderedTable:
    source = "ch5_t2_seedwise_k50_diagnostic.csv"
    frame = store.csv(source)
    required = [
        "seed",
        "target_budget",
        "delta_plr_vs_bce",
        "delta_fraud_at_k_vs_bce",
        "fraud_ceiling_utilization",
        "plr_ceiling_utilization",
        "fraud_at_k_min",
        "fraud_at_k_max",
        "technical_tie_classification",
    ]
    _require_columns(frame, required, source)
    _validate_grid(
        frame,
        source=source,
        keys=("seed",),
        expected={(seed,) for seed in store.seeds},
    )
    if not frame["target_budget"].astype(int).eq(50).all():
        raise RuntimeError(f"{source} contains a budget other than k=50.")
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("seed",),
    )
    display_header = (
        "Seed",
        "Δ PLR vs. BCE",
        "Δ Fraud@50 vs. BCE",
        "Fraud-Ceiling-Nutzung",
        "PLR-Ceiling-Nutzung",
        "exakter Tie-Bereich Fraud@50",
        "Tie-Klassifikation",
    )
    display_rows: list[list[str]] = []
    for row in frame.itertuples(index=False):
        display_rows.append(
            [
                str(int(row.seed)),
                _display_number(row.delta_plr_vs_bce, 3),
                _display_number(row.delta_fraud_at_k_vs_bce, 0),
                _display_number(row.fraud_ceiling_utilization, 3),
                _display_number(row.plr_ceiling_utilization, 3),
                (
                    f"{_display_integer(row.fraud_at_k_min)}"
                    f"–{_display_integer(row.fraud_at_k_max)}"
                ),
                _tie_class_label(row.technical_tie_classification),
            ]
        )
    note = (
        "Differenzen sind seedweise mit BCE gepaart. Ceiling-Nutzungen beziehen "
        "sich auf Verfügbarkeitsgrenzen im Kandidatenpool; sie sind keine "
        "erwartete Modellleistung. Tie-Bereiche sind exakte technische "
        "Permutationsgrenzen und keine Konfidenzintervalle."
    )
    lines = _tex_preamble("ch5_t2_seedwise_k50_diagnostic")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Seedweises \(k=50\)-Diagnostikum: Amount-Gain versus BCE}",
            r"\label{tab:ch5-seedwise-k50-diagnostic}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{2.4pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r"S[table-format=3.0] "
                r"S[table-format=-1.3] "
                r"S[table-format=-2.0] "
                r"S[table-format=1.3] "
                r"S[table-format=1.3] "
                r"c >{\raggedright\arraybackslash}X}"
            ),
            r"\toprule",
            (
                r"{Seed} & {\(\Delta\) PLR} & {\(\Delta\) Fraud@50} "
                r"& {\shortstack{Fraud-\\Ceiling}} "
                r"& {\shortstack{PLR-\\Ceiling}} "
                r"& {Fraud-Tie} & Tie-Klasse \\"
            ),
            r"\midrule",
        ]
    )
    for row in frame.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _tex_integer(row.seed),
                    _tex_number(row.delta_plr_vs_bce, 3),
                    _tex_number(row.delta_fraud_at_k_vs_bce, 0),
                    _tex_number(row.fraud_ceiling_utilization, 3),
                    _tex_number(row.plr_ceiling_utilization, 3),
                    (
                        f"{_tex_integer(row.fraud_at_k_min)}"
                        f"--{_tex_integer(row.fraud_at_k_max)}"
                    ),
                    _latex_escape(
                        _tie_class_label(row.technical_tie_classification)
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="ch5_t2_seedwise_k50_diagnostic",
        role="main",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _direction_label(value: object) -> str:
    mapping = {
        "added_by_amount_gain": "durch Amount-Gain hinzugefügt",
        "removed_from_bce": "aus BCE Top 50 entfernt",
    }
    text = str(value)
    if text not in mapping:
        raise RuntimeError(f"Unknown replacement direction: {text}")
    return mapping[text]


def _render_ch5_t3(store: TableStore) -> RenderedTable:
    seed_source = "ch5_t3_replacement_seedwise.csv"
    summary_source = "ch5_t3_replacement_summary.csv"
    boundary_source = "ch5_t3_boundary_pooled.csv"
    seedwise = store.csv(seed_source)
    summary = store.csv(summary_source)
    boundary = store.csv(boundary_source)
    seed_metrics = (
        "case_count",
        "fraud_count",
        "q90_fraud_count",
        "fraud_amount_sum",
        "mean_bce_base_score",
    )
    _require_columns(
        seedwise,
        [
            "seed",
            "target_budget",
            "direction",
            *seed_metrics,
        ],
        seed_source,
    )
    _require_columns(
        summary,
        [
            "target_budget",
            "direction",
            "row_count",
            *[
                f"{metric}_{suffix}"
                for metric in seed_metrics
                for suffix in ("n", "mean", "sd")
            ],
        ],
        summary_source,
    )
    _require_columns(
        boundary,
        [
            "target_budget",
            "window_lower_rank",
            "window_upper_rank",
            "direction",
            "seed_count",
            "pooled_case_count",
            "pooled_fraud_count",
            "pooled_q90_fraud_count",
            "pooled_mean_amount",
            "pooled_mean_bce_base_score",
            "aggregation_basis",
        ],
        boundary_source,
    )
    _validate_grid(
        seedwise,
        source=seed_source,
        keys=("seed", "direction"),
        expected={
            (seed, direction)
            for direction in DIRECTION_ORDER
            for seed in store.seeds
        },
    )
    for current, source in (
        (summary, summary_source),
        (boundary, boundary_source),
    ):
        _validate_grid(
            current,
            source=source,
            keys=("direction",),
            expected={(direction,) for direction in DIRECTION_ORDER},
        )
    if not (
        seedwise["target_budget"].astype(int).eq(50).all()
        and summary["target_budget"].astype(int).eq(50).all()
        and boundary["target_budget"].astype(int).eq(50).all()
    ):
        raise RuntimeError("Replacement table inputs contain a non-k=50 row.")
    summary = _ordered(
        summary,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("direction",),
    )
    boundary = _ordered(
        boundary,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("direction",),
    )
    for row in summary.itertuples(index=False):
        if int(row.row_count) != len(store.seeds):
            raise RuntimeError(
                f"{summary_source} does not aggregate all replacement sets."
            )
        for metric in seed_metrics:
            if int(getattr(row, f"{metric}_n")) != len(store.seeds):
                raise RuntimeError(
                    f"{summary_source} has missing seed values for {metric}."
                )
    if not boundary["seed_count"].astype(int).eq(len(store.seeds)).all():
        raise RuntimeError(
            f"{boundary_source} does not document all outer seeds."
        )
    if not (
        boundary["window_lower_rank"].astype(int).eq(30).all()
        and boundary["window_upper_rank"].astype(int).eq(70).all()
    ):
        raise RuntimeError(f"{boundary_source} is not the 30--70 boundary frame.")

    display_header = (
        "Panel",
        "Aggregationsbasis",
        "Richtung",
        "Fälle",
        "Fraud",
        "q90-Fraud",
        "Amount-Kennzahl",
        "mittlerer BCE-Basisscore",
    )
    display_rows: list[list[str]] = []
    panel_a_basis = (
        f"M ± SD über {len(store.seeds)} vollständige seedweise Ersatzmengen"
    )
    for row in summary.itertuples(index=False):
        display_rows.append(
            [
                "A",
                panel_a_basis,
                _direction_label(row.direction),
                _display_mean_sd(row.case_count_mean, row.case_count_sd, 1),
                _display_mean_sd(row.fraud_count_mean, row.fraud_count_sd, 1),
                _display_mean_sd(
                    row.q90_fraud_count_mean,
                    row.q90_fraud_count_sd,
                    1,
                ),
                (
                    "Fraud-Amount-Summe: "
                    + _display_mean_sd(
                        row.fraud_amount_sum_mean,
                        row.fraud_amount_sum_sd,
                        2,
                        grouped=True,
                    )
                ),
                _display_mean_sd(
                    row.mean_bce_base_score_mean,
                    row.mean_bce_base_score_sd,
                    3,
                ),
            ]
        )
    panel_b_basis = (
        "gepoolte Ersatzereignisse; Rang 30–70 in mindestens einer Ordnung"
    )
    for row in boundary.itertuples(index=False):
        display_rows.append(
            [
                "B",
                panel_b_basis,
                _direction_label(row.direction),
                _display_integer(row.pooled_case_count),
                _display_integer(row.pooled_fraud_count),
                _display_integer(row.pooled_q90_fraud_count),
                (
                    "mittlerer Amount: "
                    + _display_number(row.pooled_mean_amount, 2, grouped=True)
                ),
                _display_number(row.pooled_mean_bce_base_score, 3),
            ]
        )

    note = (
        "Panel A aggregiert zuerst vollständige Ersatzmengen je Seed und zeigt "
        "danach M und Stichproben-SD. Panel B poolt einzelne Boundary-Events, "
        "wenn BCE- oder Amount-Gain-Rang im inklusiven Fenster 30--70 liegt. "
        "Diese Aggregationsbasen werden nicht vermischt. Amount ist ein Proxy."
    )
    lines = _tex_preamble("ch5_t3_replacement_boundary_k50")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Ersetzungen an der Top-50-Grenze}",
            r"\label{tab:ch5-replacement-boundary-k50}",
            r"\begin{threeparttable}",
            (
                rf"\noindent\textbf{{Panel A: vollständige Ersatzmengen; "
                rf"M \(\pm\) SD über \(n={len(store.seeds)}\) Seeds}}"
                r"\par\smallskip"
            ),
            r"\setlength{\tabcolsep}{2.0pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=2.1] S[table-format=2.1] "
                r"S[table-format=2.1] S[table-format=2.1] "
                r"S[table-format=2.1] S[table-format=2.1] "
                r"S[table-format=4.2] S[table-format=4.2] "
                r"S[table-format=1.3] S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"Richtung & \multicolumn{2}{c}{Fälle} "
                r"& \multicolumn{2}{c}{Fraud} "
                r"& \multicolumn{2}{c}{q90-Fraud} "
                r"& \multicolumn{2}{c}{Fraud-Amount-Summe} "
                r"& \multicolumn{2}{c}{BCE-Basisscore} \\"
            ),
            (
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
                r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
                r"\cmidrule(lr){10-11}"
            ),
            (
                r" & {M} & {SD} & {M} & {SD} & {M} & {SD} "
                r"& {M} & {SD} & {M} & {SD} \\"
            ),
            r"\midrule",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _latex_escape(_direction_label(row.direction)),
                    _tex_number(row.case_count_mean, 1),
                    _tex_number(row.case_count_sd, 1),
                    _tex_number(row.fraud_count_mean, 1),
                    _tex_number(row.fraud_count_sd, 1),
                    _tex_number(row.q90_fraud_count_mean, 1),
                    _tex_number(row.q90_fraud_count_sd, 1),
                    _tex_number(row.fraud_amount_sum_mean, 2),
                    _tex_number(row.fraud_amount_sum_sd, 2),
                    _tex_number(row.mean_bce_base_score_mean, 3),
                    _tex_number(row.mean_bce_base_score_sd, 3),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\medskip",
            (
                r"\noindent\textbf{Panel B: gepoolte Boundary-Events; "
                r"Rangfenster 30--70}\par\smallskip"
            ),
            r"\setlength{\tabcolsep}{3.2pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=3.0] S[table-format=3.0] "
                r"S[table-format=2.0] S[table-format=4.2] "
                r"S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"Richtung & {Fälle} & {Fraud} & {q90-Fraud} "
                r"& {\shortstack{Amount\\(M, gepoolt)}} "
                r"& {\shortstack{BCE-Basisscore\\(M, gepoolt)}} \\"
            ),
            r"\midrule",
        ]
    )
    for row in boundary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _latex_escape(_direction_label(row.direction)),
                    _tex_integer(row.pooled_case_count),
                    _tex_integer(row.pooled_fraud_count),
                    _tex_integer(row.pooled_q90_fraud_count),
                    _tex_number(row.pooled_mean_amount, 2),
                    _tex_number(row.pooled_mean_bce_base_score, 3),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="ch5_t3_replacement_boundary_k50",
        role="main",
        sources=(seed_source, summary_source, boundary_source),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_ch5_t4(store: TableStore) -> RenderedTable:
    seed_source = "ch5_t4_high_amount_legit_seedwise.csv"
    summary_source = "ch5_t4_high_amount_legit_summary.csv"
    seedwise = store.csv(seed_source)
    summary = store.csv(summary_source)
    metrics = (
        "legitimate_count",
        "q90_high_amount_legitimate_count",
        "high_amount_legitimate_share",
        "mean_legitimate_amount",
        "mean_bce_base_score",
    )
    _require_columns(
        seedwise,
        ["seed", "target_budget", "path_id", *metrics],
        seed_source,
    )
    _require_columns(
        summary,
        [
            "target_budget",
            "path_id",
            "row_count",
            *[
                f"{metric}_{suffix}"
                for metric in metrics
                for suffix in ("n", "mean", "sd")
            ],
        ],
        summary_source,
    )
    _validate_grid(
        seedwise,
        source=seed_source,
        keys=("seed",),
        expected={(seed,) for seed in store.seeds},
    )
    if len(summary) != 1:
        raise RuntimeError(f"{summary_source} must contain exactly one row.")
    if not (
        seedwise["target_budget"].astype(int).eq(50).all()
        and summary["target_budget"].astype(int).eq(50).all()
        and seedwise["path_id"].eq("amount_gain").all()
        and summary["path_id"].eq("amount_gain").all()
    ):
        raise RuntimeError("High-Amount legitimate table has the wrong scope.")
    row = summary.iloc[0]
    if int(row["row_count"]) != len(store.seeds):
        raise RuntimeError(f"{summary_source} does not aggregate every seed.")
    for metric in metrics:
        if int(row[f"{metric}_n"]) != len(store.seeds):
            raise RuntimeError(f"{summary_source} has missing values for {metric}.")
    entries = [
        (
            "Legitime Fälle in Amount-Gain Top 50",
            "legitimate_count",
            1,
            False,
        ),
        (
            "q90-High-Amount-Legitime",
            "q90_high_amount_legitimate_count",
            1,
            False,
        ),
        (
            "High-Amount-Anteil legitimer Fälle",
            "high_amount_legitimate_share",
            3,
            False,
        ),
        (
            "Mittlerer legitimer Amount",
            "mean_legitimate_amount",
            2,
            True,
        ),
        (
            "Mittlerer BCE-Basisscore",
            "mean_bce_base_score",
            3,
            False,
        ),
    ]
    display_header = ("Kennzahl", "M", "SD", "n")
    display_rows = [
        [
            label,
            _display_number(row[f"{metric}_mean"], decimals, grouped=grouped),
            _display_number(row[f"{metric}_sd"], decimals, grouped=grouped),
            str(len(store.seeds)),
        ]
        for label, metric, decimals, grouped in entries
    ]
    note = (
        f"M und Stichproben-SD werden über n={len(store.seeds)} Seeds gebildet. "
        "q90 bezeichnet je Seed das 90-%-Quantil des Amount unter legitimen "
        "Outer-Test-Fällen. Amount ist ein Proxy."
    )
    lines = _tex_preamble("ch5_t4_high_amount_legit_k50")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Legitimer High-Amount-Guardrail für Amount-Gain@50}",
            r"\label{tab:ch5-high-amount-legit-k50}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=4.3] S[table-format=4.3] "
                r"S[table-format=1.0]}"
            ),
            r"\toprule",
            r"Kennzahl & {M} & {SD} & {\(n\)} \\",
            r"\midrule",
        ]
    )
    for label, metric, decimals, _ in entries:
        lines.append(
            " & ".join(
                [
                    _latex_escape(label),
                    _tex_number(row[f"{metric}_mean"], decimals),
                    _tex_number(row[f"{metric}_sd"], decimals),
                    _tex_integer(row[f"{metric}_n"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="ch5_t4_high_amount_legit_k50",
        role="main",
        sources=(seed_source, summary_source),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_app_t1(store: TableStore) -> RenderedTable:
    source = "app_t1_hard_impact_exact_values.csv"
    frame = store.csv(source)
    required = [
        "seed",
        "target_budget",
        "path_id",
        "path_label",
        "q90_captured_ratio",
        "baseline_miss_recovery",
        "amount_ndcg_at_50",
    ]
    _require_columns(frame, required, source)
    expected = {
        (path_id, seed) for path_id in PATH_ORDER for seed in store.seeds
    }
    _validate_grid(
        frame,
        source=source,
        keys=("path_id", "seed"),
        expected=expected,
    )
    if not frame["target_budget"].astype(int).eq(50).all():
        raise RuntimeError(f"{source} contains a budget other than k=50.")
    bce = frame.loc[frame["path_id"] == "BCE", "baseline_miss_recovery"]
    if bce.notna().any():
        raise RuntimeError("BCE miss recovery must remain missing.")
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("path_id", "seed"),
    )
    display_header = (
        "Pfad",
        "Seed",
        "q90 Capture",
        "Baseline Miss Recovery",
        "Amount-nDCG@50",
    )
    display_rows = [
        [
            str(row.path_label),
            str(int(row.seed)),
            _display_number(row.q90_captured_ratio, 3),
            _display_number(row.baseline_miss_recovery, 3),
            _display_number(row.amount_ndcg_at_50, 3),
        ]
        for row in frame.itertuples(index=False)
    ]
    note = (
        "Alle Werte sind seedweise Einzelwerte bei q=0,90 und k=50. Ein "
        "Gedankenstrich kennzeichnet nicht anwendbar; für BCE ist Baseline Miss "
        "Recovery definitionsgemäß nicht anwendbar. Amount ist ein Proxy."
    )
    lines = _tex_preamble("app_t1_hard_impact_exact_values")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Exakte seedweise Hard-Impact-Werte bei \(k=50\)}",
            r"\label{tab:app-hard-impact-exact}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=3.0] S[table-format=1.3] "
                r"S[table-format=1.3] S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"Pfad & {Seed} & {q90 Capture} "
                r"& {Baseline Miss Recovery} & {Amount-nDCG@50} \\"
            ),
            r"\midrule",
        ]
    )
    previous_path: str | None = None
    for row in frame.itertuples(index=False):
        if previous_path is not None and row.path_id != previous_path:
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.path_label),
                    _tex_integer(row.seed),
                    _tex_number(row.q90_captured_ratio, 3),
                    _tex_number(row.baseline_miss_recovery, 3),
                    _tex_number(row.amount_ndcg_at_50, 3),
                ]
            )
            + r" \\"
        )
        previous_path = str(row.path_id)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="app_t1_hard_impact_exact_values",
        role="appendix",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    budget: int,
    path_id: str,
    metric: str,
    source: str,
) -> pd.Series:
    selected = frame.loc[
        (frame["scope"] == scope)
        & (frame["target_budget"].astype(int) == budget)
        & (frame["path_id"] == path_id)
        & (frame["metric"] == metric)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"{source} lacks one unique {(scope, budget, path_id, metric)} row."
        )
    return selected.iloc[0]


def _score_interpretation(scope: str, path_id: str) -> str:
    if scope == "full_order":
        if path_id == "BCE":
            return "vollständige, durch p_fraud induzierte Testordnung"
        return "vollständige zusammengesetzte Ordnung; Rankerscore ordinal"
    if path_id == "BCE":
        return "BCE-Reihenfolge innerhalb des Top-1000-Pools"
    return "Ordnung im Top-1000-Pool; Rankerscore ordinal"


def _render_app_t2(store: TableStore) -> RenderedTable:
    source = "app_t2_global_metrics_by_budget.csv"
    frame = store.csv(source)
    _require_columns(
        frame,
        [
            "target_budget",
            "scope",
            "scope_label",
            "path_id",
            "path_label",
            "metric",
            "score_interpretation",
            "row_count",
            "value_n",
            "value_mean",
            "value_sd",
        ],
        source,
    )
    if frame.duplicated(
        ["target_budget", "scope", "path_id", "metric"]
    ).any():
        raise RuntimeError(f"{source} has duplicate metric rows.")
    if not frame.loc[frame["metric"] == "brier", "path_id"].eq("BCE").all():
        raise RuntimeError("Brier rows must be restricted to BCE.")
    budgets = tuple(value for value in CENTRAL_BUDGETS if value in store.budgets)
    full_rows: list[dict[str, object]] = []
    for budget in budgets:
        for path_id in PATH_ORDER:
            auc = _metric_row(
                frame,
                scope="full_order",
                budget=budget,
                path_id=path_id,
                metric="roc_auc",
                source=source,
            )
            ap = _metric_row(
                frame,
                scope="full_order",
                budget=budget,
                path_id=path_id,
                metric="average_precision",
                source=source,
            )
            if int(auc["value_n"]) != len(store.seeds) or int(
                ap["value_n"]
            ) != len(store.seeds):
                raise RuntimeError(f"{source} global metric omits a seed.")
            full_rows.append(
                {
                    "scope": "full_order",
                    "scope_label": auc["scope_label"],
                    "target_budget": budget,
                    "path_id": path_id,
                    "path_label": auc["path_label"],
                    "auc_mean": auc["value_mean"],
                    "auc_sd": auc["value_sd"],
                    "ap_mean": ap["value_mean"],
                    "ap_sd": ap["value_sd"],
                    "interpretation": _score_interpretation(
                        "full_order", path_id
                    ),
                }
            )
    pool_rows: list[dict[str, object]] = []
    if 50 in budgets:
        for path_id in PATH_ORDER:
            auc = _metric_row(
                frame,
                scope="candidate_pool",
                budget=50,
                path_id=path_id,
                metric="roc_auc",
                source=source,
            )
            ap = _metric_row(
                frame,
                scope="candidate_pool",
                budget=50,
                path_id=path_id,
                metric="average_precision",
                source=source,
            )
            pool_rows.append(
                {
                    "scope": "candidate_pool",
                    "scope_label": auc["scope_label"],
                    "target_budget": 50,
                    "path_id": path_id,
                    "path_label": auc["path_label"],
                    "auc_mean": auc["value_mean"],
                    "auc_sd": auc["value_sd"],
                    "ap_mean": ap["value_mean"],
                    "ap_sd": ap["value_sd"],
                    "interpretation": _score_interpretation(
                        "candidate_pool", path_id
                    ),
                }
            )
    brier = frame.loc[
        (frame["scope"] == "full_order")
        & (frame["path_id"] == "BCE")
        & (frame["metric"] == "brier")
        & frame["target_budget"].astype(int).isin(budgets)
    ].copy()
    if len(brier) != len(budgets):
        raise RuntimeError(f"{source} lacks BCE Brier rows for central budgets.")
    if not (
        brier["value_mean"].map(float).map(math.isfinite).all()
        and brier["value_sd"].map(float).map(math.isfinite).all()
    ):
        raise RuntimeError(f"{source} has a non-finite BCE Brier aggregate.")
    first_brier = brier.iloc[0]
    if not all(
        math.isclose(
            float(value),
            float(first_brier["value_mean"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in brier["value_mean"]
    ) or not all(
        math.isclose(
            float(value),
            float(first_brier["value_sd"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in brier["value_sd"]
    ):
        raise RuntimeError("BCE Brier is not identical across budget copies.")

    display_header = (
        "Scope",
        "k",
        "Pfad",
        "ROC-AUC (M ± SD)",
        "AP (M ± SD)",
        "Brier (M ± SD)",
        "Score-Interpretation",
    )
    display_rows: list[list[str]] = []
    for row in [*full_rows, *pool_rows]:
        display_rows.append(
            [
                str(row["scope_label"]),
                str(int(row["target_budget"])),
                str(row["path_label"]),
                _display_mean_sd(row["auc_mean"], row["auc_sd"], 4),
                _display_mean_sd(row["ap_mean"], row["ap_sd"], 4),
                NA_DISPLAY,
                str(row["interpretation"]),
            ]
        )
    display_rows.append(
        [
            str(first_brier["scope_label"]),
            "identisch über k",
            "BCE",
            NA_DISPLAY,
            NA_DISPLAY,
            _display_mean_sd(
                first_brier["value_mean"],
                first_brier["value_sd"],
                6,
            ),
            "BCE-Wahrscheinlichkeit p_fraud; einzig zulässiger Brier-Score",
        ]
    )

    note = (
        f"M und Stichproben-SD basieren auf n={len(store.seeds)} Seeds. "
        "ROC-AUC und AP bewerten die jeweils explizit genannte Ordnung; ordinale "
        "Rankerscores sind keine Wahrscheinlichkeiten. Brier wird einmalig und "
        "nur für die identische BCE-Wahrscheinlichkeit p_fraud ausgewiesen. "
        "Vollständige Testordnung und Top-1000-Kandidatenpool sind getrennte "
        "Scopes."
    )
    lines = _tex_preamble("app_t2_global_metrics_by_budget")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Globale Ordnungsmetriken nach Budgetmodell und Scope}",
            r"\label{tab:app-global-metrics-by-budget}",
            r"\begin{threeparttable}",
        ]
    )

    def add_metric_panel(
        title: str,
        rows: list[dict[str, object]],
    ) -> None:
        lines.extend(
            [
                rf"\textbf{{{_latex_escape(title)}}}\par\smallskip",
                r"\setlength{\tabcolsep}{2.4pt}",
                (
                    r"\begin{tabularx}{\linewidth}{"
                    r"S[table-format=3.0] "
                    r">{\raggedright\arraybackslash}p{18mm} "
                    r"S[table-format=1.4] S[table-format=1.4] "
                    r"S[table-format=1.4] S[table-format=1.4] "
                    r">{\raggedright\arraybackslash}X}"
                ),
                r"\toprule",
                (
                    r"{\(k\)} & Pfad & \multicolumn{2}{c}{ROC-AUC} "
                    r"& \multicolumn{2}{c}{AP} & Score-Interpretation \\"
                ),
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
                r" & & {M} & {SD} & {M} & {SD} & \\",
                r"\midrule",
            ]
        )
        for row in rows:
            lines.append(
                " & ".join(
                    [
                        _tex_integer(row["target_budget"]),
                        _latex_escape(row["path_label"]),
                        _tex_number(row["auc_mean"], 4),
                        _tex_number(row["auc_sd"], 4),
                        _tex_number(row["ap_mean"], 4),
                        _tex_number(row["ap_sd"], 4),
                        _latex_escape(row["interpretation"]),
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabularx}", r"\medskip"])

    add_metric_panel("Panel A: vollständige Testordnung", full_rows)
    add_metric_panel("Panel B: BCE-Top-1000-Kandidatenpool", pool_rows)
    lines.extend(
        [
            r"\textbf{Panel C: BCE-Brier (budgetidentisch)}\par\smallskip",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=1.6] S[table-format=1.6]}"
            ),
            r"\toprule",
            r"Score & {M} & {SD} \\",
            r"\midrule",
            (
                r"BCE-Wahrscheinlichkeit \(p_{\mathrm{fraud}}\) & "
                f"{_tex_number(first_brier['value_mean'], 6)} & "
                f"{_tex_number(first_brier['value_sd'], 6)} "
                r"\\"
            ),
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="app_t2_global_metrics_by_budget",
        role="appendix",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_app_t3(store: TableStore) -> RenderedTable:
    source = "app_t3_exact_tie_bounds.csv"
    frame = store.csv(source)
    required = [
        "seed",
        "target_budget",
        "path_id",
        "path_label",
        "fraud_at_k_min",
        "fraud_at_k_actual",
        "fraud_at_k_max",
        "plr_at_k_min",
        "plr_at_k_actual",
        "plr_at_k_max",
        "interval_interpretation",
    ]
    _require_columns(frame, required, source)
    budgets = tuple(value for value in CENTRAL_BUDGETS if value in store.budgets)
    expected = {
        (budget, path_id, seed)
        for budget in budgets
        for path_id in TRAINED_PATH_ORDER
        for seed in store.seeds
    }
    _validate_grid(
        frame,
        source=source,
        keys=("target_budget", "path_id", "seed"),
        expected=expected,
    )
    for row in frame.itertuples(index=False):
        if not (
            float(row.fraud_at_k_min)
            <= float(row.fraud_at_k_actual)
            <= float(row.fraud_at_k_max)
            and float(row.plr_at_k_min) - 1e-12
            <= float(row.plr_at_k_actual)
            <= float(row.plr_at_k_max) + 1e-12
        ):
            raise RuntimeError(f"{source} has an actual value outside its bounds.")
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        path_ids=TRAINED_PATH_ORDER,
        keys=("target_budget", "path_id", "seed"),
    )
    display_header = (
        "k",
        "Pfad",
        "Seed",
        "Fraud@k min",
        "Fraud@k tatsächlich",
        "Fraud@k max",
        "PLR@k min",
        "PLR@k tatsächlich",
        "PLR@k max",
    )
    display_rows = [
        [
            str(int(row.target_budget)),
            str(row.path_label),
            str(int(row.seed)),
            _display_integer(row.fraud_at_k_min),
            _display_integer(row.fraud_at_k_actual),
            _display_integer(row.fraud_at_k_max),
            _display_number(row.plr_at_k_min, 6),
            _display_number(row.plr_at_k_actual, 6),
            _display_number(row.plr_at_k_max, 6),
        ]
        for row in frame.itertuples(index=False)
    ]
    note = (
        "Minimum und Maximum sind exakte technische Permutationsgrenzen des "
        "Cutoff-Tie-Blocks und keine Konfidenzintervalle. PLR@k bezeichnet die "
        "Fraud-Amount-Proxy-Abdeckung."
    )
    lines = _tex_preamble("app_t3_exact_tie_bounds")
    lines.extend(
        [
            r"\setlength{\LTleft}{0pt}",
            r"\setlength{\LTright}{0pt}",
            r"\setlength{\tabcolsep}{2.5pt}",
            (
                r"\begin{longtable}{"
                r"S[table-format=3.0] l S[table-format=3.0] "
                r"S[table-format=3.0] S[table-format=3.0] "
                r"S[table-format=3.0] S[table-format=1.6] "
                r"S[table-format=1.6] S[table-format=1.6]}"
            ),
            (
                r"\caption{Exakte technische Tie-Bounds der trainierten Pfade}"
                r"\label{tab:app-exact-tie-bounds}\\"
            ),
            r"\toprule",
            (
                r"{\(k\)} & Pfad & {Seed} "
                r"& \multicolumn{3}{c}{Fraud@\(k\)} "
                r"& \multicolumn{3}{c}{PLR@\(k\)} \\"
            ),
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
            (
                r" & & & {min.} & {tats.} & {max.} "
                r"& {min.} & {tats.} & {max.} \\"
            ),
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            (
                r"{\(k\)} & Pfad & {Seed} "
                r"& \multicolumn{3}{c}{Fraud@\(k\)} "
                r"& \multicolumn{3}{c}{PLR@\(k\)} \\"
            ),
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
            (
                r" & & & {min.} & {tats.} & {max.} "
                r"& {min.} & {tats.} & {max.} \\"
            ),
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{9}{r}{Fortsetzung auf der nächsten Seite}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    previous_budget: int | None = None
    previous_path: str | None = None
    for row in frame.itertuples(index=False):
        budget = int(row.target_budget)
        path_id = str(row.path_id)
        if previous_budget is not None and budget != previous_budget:
            lines.append(r"\addlinespace[3pt]")
        elif previous_path is not None and path_id != previous_path:
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            " & ".join(
                [
                    _tex_integer(row.target_budget),
                    _latex_escape(row.path_label),
                    _tex_integer(row.seed),
                    _tex_integer(row.fraud_at_k_min),
                    _tex_integer(row.fraud_at_k_actual),
                    _tex_integer(row.fraud_at_k_max),
                    _tex_number(row.plr_at_k_min, 6),
                    _tex_number(row.plr_at_k_actual, 6),
                    _tex_number(row.plr_at_k_max, 6),
                ]
            )
            + r" \\"
        )
        previous_budget = budget
        previous_path = path_id
    lines.extend(
        [
            r"\end{longtable}",
            rf"\noindent\parbox{{\linewidth}}{{\fontsize{{7.5}}{{9}}\selectfont "
            rf"{_latex_escape(note)}}}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="app_t3_exact_tie_bounds",
        role="appendix",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_app_t4(store: TableStore) -> RenderedTable:
    source = "app_t4_candidate_pool_coverage.csv"
    frame = store.csv(source)
    required = [
        "seed",
        "target_budget",
        "path_id",
        "fraud_ceiling_utilization",
        "plr_ceiling_utilization",
        "q90_ceiling_utilization",
        "candidate_pool_fraud_case_coverage",
        "candidate_pool_fraud_amount_coverage",
        "ceiling_interpretation",
    ]
    _require_columns(frame, required, source)
    budgets = tuple(value for value in CENTRAL_BUDGETS if value in store.budgets)
    expected = {
        (budget, seed) for budget in budgets for seed in store.seeds
    }
    _validate_grid(
        frame,
        source=source,
        keys=("target_budget", "seed"),
        expected=expected,
    )
    if not frame["path_id"].eq("amount_gain").all():
        raise RuntimeError(f"{source} must contain Amount-Gain only.")
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("target_budget", "seed"),
    )
    display_header = (
        "k",
        "Seed",
        "Fraud-Ceiling-Nutzung",
        "PLR-Ceiling-Nutzung",
        "q90-Ceiling-Nutzung",
        "Kandidatenpool-Fraudfall-Abdeckung",
        "Kandidatenpool-Fraud-Amount-Abdeckung",
    )
    display_rows = [
        [
            str(int(row.target_budget)),
            str(int(row.seed)),
            _display_number(row.fraud_ceiling_utilization, 3),
            _display_number(row.plr_ceiling_utilization, 3),
            _display_number(row.q90_ceiling_utilization, 3),
            _display_number(row.candidate_pool_fraud_case_coverage, 3),
            _display_number(row.candidate_pool_fraud_amount_coverage, 3),
        ]
        for row in frame.itertuples(index=False)
    ]
    note = (
        "Ceilings sind Verfügbarkeitsgrenzen und keine erwartete "
        "Modellleistung. Sie verwenden Outer-Test-Labels ausschließlich als "
        "Post-hoc-Diagnostik. Amount und PLR sind Proxygrößen."
    )
    lines = _tex_preamble("app_t4_candidate_pool_coverage")
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Kandidatenpool-Abdeckung und Ceiling-Nutzung}",
            r"\label{tab:app-candidate-pool-coverage}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{3.1pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r"S[table-format=3.0] S[table-format=3.0] "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"{\(k\)} & {Seed} & {\shortstack{Fraud-\\Ceiling}} "
                r"& {\shortstack{PLR-\\Ceiling}} "
                r"& {\shortstack{q90-\\Ceiling}} "
                r"& {\shortstack{Fraudfall-\\Abdeckung}} "
                r"& {\shortstack{Fraud-Amount-\\Abdeckung}} \\"
            ),
            r"\midrule",
        ]
    )
    previous_budget: int | None = None
    for row in frame.itertuples(index=False):
        budget = int(row.target_budget)
        if previous_budget is not None and budget != previous_budget:
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            " & ".join(
                [
                    _tex_integer(row.target_budget),
                    _tex_integer(row.seed),
                    _tex_number(row.fraud_ceiling_utilization, 3),
                    _tex_number(row.plr_ceiling_utilization, 3),
                    _tex_number(row.q90_ceiling_utilization, 3),
                    _tex_number(row.candidate_pool_fraud_case_coverage, 3),
                    _tex_number(row.candidate_pool_fraud_amount_coverage, 3),
                ]
            )
            + r" \\"
        )
        previous_budget = budget
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="app_t4_candidate_pool_coverage",
        role="appendix",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_app_t5(store: TableStore) -> RenderedTable:
    source = "app_t5_seedwise_central_results.csv"
    frame = store.csv(source)
    required = [
        "seed",
        "target_budget",
        "path_id",
        "path_label",
        "prevented_loss_ratio_at_k",
        "frauds_at_k",
        "precision_at_k",
        "recall_at_k",
    ]
    _require_columns(frame, required, source)
    budgets = tuple(value for value in CENTRAL_BUDGETS if value in store.budgets)
    expected = {
        (budget, path_id, seed)
        for budget in budgets
        for path_id in PATH_ORDER
        for seed in store.seeds
    }
    _validate_grid(
        frame,
        source=source,
        keys=("target_budget", "path_id", "seed"),
        expected=expected,
    )
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("target_budget", "path_id", "seed"),
    )
    display_header = (
        "k",
        "Pfad",
        "Seed",
        "PLR@k",
        "Fraud@k",
        "Precision@k",
        "Recall@k",
    )
    display_rows = [
        [
            str(int(row.target_budget)),
            str(row.path_label),
            str(int(row.seed)),
            _display_number(row.prevented_loss_ratio_at_k, 6),
            _display_integer(row.frauds_at_k),
            _display_number(row.precision_at_k, 6),
            _display_number(row.recall_at_k, 6),
        ]
        for row in frame.itertuples(index=False)
    ]
    note = (
        "Seedweise Einzelwerte. PLR@k bezeichnet die "
        "Fraud-Amount-Proxy-Abdeckung, nicht realisierten "
        "oder verhinderten finanziellen Verlust. p-only und Amount-Gain sind "
        "separate budgetkonditionierte Modelle je k."
    )
    lines = _tex_preamble("app_t5_seedwise_central_results")
    lines.extend(
        [
            r"\setlength{\LTleft}{0pt}",
            r"\setlength{\LTright}{0pt}",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\begin{longtable}{"
                r"S[table-format=3.0] l S[table-format=3.0] "
                r"S[table-format=1.6] S[table-format=3.0] "
                r"S[table-format=1.6] S[table-format=1.6]}"
            ),
            (
                r"\caption{Seedweise Ergebnisse der zentralen Budgets}"
                r"\label{tab:app-seedwise-central-results}\\"
            ),
            r"\toprule",
            (
                r"{\(k\)} & Pfad & {Seed} & {PLR@\(k\)} "
                r"& {Fraud@\(k\)} & {Precision@\(k\)} & {Recall@\(k\)} \\"
            ),
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            (
                r"{\(k\)} & Pfad & {Seed} & {PLR@\(k\)} "
                r"& {Fraud@\(k\)} & {Precision@\(k\)} & {Recall@\(k\)} \\"
            ),
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{7}{r}{Fortsetzung auf der nächsten Seite}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )
    previous_budget: int | None = None
    previous_path: str | None = None
    for row in frame.itertuples(index=False):
        budget = int(row.target_budget)
        path_id = str(row.path_id)
        if previous_budget is not None and budget != previous_budget:
            lines.append(r"\addlinespace[3pt]")
        elif previous_path is not None and path_id != previous_path:
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            " & ".join(
                [
                    _tex_integer(row.target_budget),
                    _latex_escape(row.path_label),
                    _tex_integer(row.seed),
                    _tex_number(row.prevented_loss_ratio_at_k, 6),
                    _tex_integer(row.frauds_at_k),
                    _tex_number(row.precision_at_k, 6),
                    _tex_number(row.recall_at_k, 6),
                ]
            )
            + r" \\"
        )
        previous_budget = budget
        previous_path = path_id
    lines.extend(
        [
            r"\end{longtable}",
            rf"\noindent\parbox{{\linewidth}}{{\fontsize{{7.5}}{{9}}\selectfont "
            rf"{_latex_escape(note)}}}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem="app_t5_seedwise_central_results",
        role="appendix",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


def _render_engineering_t1(store: TableStore) -> RenderedTable:
    """Render the shared engineering central-budget acceptance table."""

    source = "engineering/tables/engineering_central_topk_summary.csv"
    frame = store.csv(source)
    metrics = (
        ("plr", "PLR@k", 3),
        ("fraud_at_k", "Fraud@k", 1),
        ("precision_at_k", "Precision@k", 3),
        ("recall_at_k", "Recall@k", 3),
    )
    required = [
        "profile",
        "evidence_classification",
        "data_source_kind",
        "evidence_statement",
        "target_budget",
        "method_family",
        "path_id",
        "seed_count",
        *[
            f"{metric}_{suffix}"
            for metric, _label, _decimals in metrics
            for suffix in ("mean", "sample_sd")
        ],
    ]
    _require_columns(frame, required, source)
    metadata = {
        "profile": store.profile,
        "evidence_classification": store.evidence_classification,
        "data_source_kind": store.data_source_kind,
        "evidence_statement": store.evidence_statement,
    }
    for column, expected_value in metadata.items():
        observed = frame[column].dropna().astype(str).unique().tolist()
        if observed != [str(expected_value)]:
            raise RuntimeError(
                f"Engineering table {column} does not match its manifest."
            )
    expected_grid = {
        (budget, path_id)
        for budget in store.budgets
        for path_id in PATH_ORDER
    }
    _validate_grid(
        frame,
        source=source,
        keys=("target_budget", "path_id"),
        expected=expected_grid,
    )
    method_by_path = {
        "BCE": METHOD_BCE,
        "p_only": METHOD_P_ONLY,
        "amount_gain": METHOD_AMOUNT_GAIN,
        "fixed_reference": METHOD_FIXED,
    }
    expected_methods = frame["path_id"].map(method_by_path)
    if expected_methods.isna().any() or not expected_methods.equals(
        frame["method_family"].astype(str)
    ):
        raise RuntimeError("Engineering table contains an unknown path.")
    frame = _ordered(
        frame,
        seeds=store.seeds,
        budgets=store.budgets,
        keys=("target_budget", "path_id"),
    )
    sample_sd_columns = [
        f"{metric}_sample_sd" for metric, _label, _decimals in metrics
    ]
    mean_columns = [
        f"{metric}_mean" for metric, _label, _decimals in metrics
    ]
    if not frame["seed_count"].eq(len(store.seeds)).all():
        raise RuntimeError("Engineering table seed count is incomplete.")
    try:
        means = frame.loc[:, mean_columns].to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Engineering table contains invalid means.") from exc
    if not pd.notna(means).all() or not all(
        math.isfinite(float(value)) for value in means.ravel()
    ):
        raise RuntimeError("Engineering table contains non-finite means.")
    if len(store.seeds) == 1:
        if not frame.loc[:, sample_sd_columns].isna().all().all():
            raise RuntimeError(
                "One-seed engineering sample SD must remain missing, not zero."
            )
    else:
        try:
            standard_deviations = frame.loc[
                :, sample_sd_columns
            ].to_numpy(float)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Engineering table contains invalid sample SD values."
            ) from exc
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in standard_deviations.ravel()
        ):
            raise RuntimeError(
                "Engineering table sample SD values must be finite and non-negative."
            )

    path_labels = {
        "BCE": "BCE",
        "p_only": "p-only",
        "amount_gain": "Amount-Gain",
        "fixed_reference": "fixed reference",
    }
    display_header = (
        "Profile",
        "Evidence",
        "k",
        "Path",
        "Seeds",
        "PLR@k mean",
        "PLR@k sample SD",
        "Fraud@k mean",
        "Fraud@k sample SD",
        "Precision@k mean",
        "Precision@k sample SD",
        "Recall@k mean",
        "Recall@k sample SD",
    )
    display_rows: list[list[str]] = []
    for row in frame.itertuples(index=False):
        display_rows.append(
            [
                str(store.profile),
                str(store.evidence_statement),
                str(int(row.target_budget)),
                path_labels[str(row.path_id)],
                str(int(row.seed_count)),
                _display_number(row.plr_mean, 3),
                _display_number(row.plr_sample_sd, 3),
                _display_number(row.fraud_at_k_mean, 1),
                _display_number(row.fraud_at_k_sample_sd, 1),
                _display_number(row.precision_at_k_mean, 3),
                _display_number(row.precision_at_k_sample_sd, 3),
                _display_number(row.recall_at_k_mean, 3),
                _display_number(row.recall_at_k_sample_sd, 3),
            ]
        )

    stem = "engineering_t1_central_topk_summary"
    note = (
        f"Profile: {store.profile}. Evidence classification: "
        f"{store.evidence_classification}. {store.evidence_statement}. "
        "M is the arithmetic seed mean; SD is the sample standard "
        "deviation (ddof=1). A dash marks mathematically undefined SD "
        "for a one-seed run. Technical acceptance only; no winner or "
        "significance interpretation."
    )
    lines = _tex_preamble(stem)
    lines.extend(
        [
            r"\begin{table}[tbp]",
            r"\centering",
            rf"\caption{{Engineering Top-k summary --- {_latex_escape(store.profile)}}}",
            r"\begin{threeparttable}",
            r"\setlength{\tabcolsep}{2.2pt}",
            (
                r"\begin{tabularx}{\linewidth}{"
                r">{\raggedright\arraybackslash}X "
                r"S[table-format=1.0] "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=3.1] S[table-format=2.1] "
                r"S[table-format=1.3] S[table-format=1.3] "
                r"S[table-format=1.3] S[table-format=1.3]}"
            ),
            r"\toprule",
            (
                r"Path & {n} & \multicolumn{2}{c}{PLR@\(k\)} "
                r"& \multicolumn{2}{c}{Fraud@\(k\)} "
                r"& \multicolumn{2}{c}{Precision@\(k\)} "
                r"& \multicolumn{2}{c}{Recall@\(k\)} \\"
            ),
            (
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"
                r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}"
            ),
            r" & & {M} & {SD} & {M} & {SD} & {M} & {SD} & {M} & {SD} \\",
            r"\midrule",
        ]
    )
    for budget in store.budgets:
        panel = frame.loc[frame["target_budget"].astype(int) == budget]
        lines.append(
            r"\multicolumn{10}{l}{\textit{\(k="
            + str(budget)
            + r"\)}} \\"
        )
        for row in panel.itertuples(index=False):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(path_labels[str(row.path_id)]),
                        str(int(row.seed_count)),
                        _tex_number(row.plr_mean, 3),
                        _tex_number(row.plr_sample_sd, 3),
                        _tex_number(row.fraud_at_k_mean, 1),
                        _tex_number(row.fraud_at_k_sample_sd, 1),
                        _tex_number(row.precision_at_k_mean, 3),
                        _tex_number(row.precision_at_k_sample_sd, 3),
                        _tex_number(row.recall_at_k_mean, 3),
                        _tex_number(row.recall_at_k_sample_sd, 3),
                    ]
                )
                + r" \\"
            )
        if budget != store.budgets[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            *_tex_note(note),
            r"\end{threeparttable}",
            r"\end{table}",
            *_tex_finish(),
        ]
    )
    return RenderedTable(
        stem=stem,
        role="engineering",
        sources=(source,),
        display_rows=display_rows,
        display_header=display_header,
        latex="\n".join(lines),
    )


TABLE_RENDERERS: tuple[Callable[[TableStore], RenderedTable], ...] = (
    _render_ch5_t1,
    _render_ch5_t2,
    _render_ch5_t3,
    _render_ch5_t4,
    _render_app_t1,
    _render_app_t2,
    _render_app_t3,
    _render_app_t4,
    _render_app_t5,
)


def _write_display_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[list[str]],
) -> None:
    require_new_file(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_latex(path: Path, value: str) -> None:
    require_new_file(path)
    path.write_text(value, encoding="utf-8", newline="\n")


def _default_preview_dir(repository_root: Path, output_dir: Path) -> Path:
    output = (
        output_dir
        if output_dir.is_absolute()
        else repository_root.resolve() / output_dir
    ).resolve()
    name = "preview" if output.name == "tables" else f"{output.name}-preview"
    return output.parent / name


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def _preflight_output_directories(
    repository_root: Path,
    output_dir: Path,
    preview_dir: Path,
    *,
    force: bool,
) -> tuple[Path, Path]:
    output = require_generated_path(repository_root, output_dir)
    preview = require_generated_path(repository_root, preview_dir)
    if output == preview:
        raise ValueError("Table and preview output directories must differ.")
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output}"
        )
    if preview.exists() and not preview.is_dir():
        raise FileExistsError(f"Output path is not a directory: {preview}")
    owned_preview_names = (
        "ch5_tables_preview.tex",
        "ch5_tables_preview.pdf",
        "ch5_tables_preview.aux",
        "ch5_tables_preview.log",
        "ch5_tables_preview.out",
        "engineering_tables_preview.tex",
        "engineering_tables_preview.pdf",
        "engineering_tables_preview.aux",
        "engineering_tables_preview.log",
        "engineering_tables_preview.out",
        "LATEX_PREVIEW_STATUS.json",
    )
    owned_existing = [
        preview / name
        for name in owned_preview_names
        if (preview / name).exists()
    ]
    owned_directories = [path for path in owned_existing if path.is_dir()]
    if owned_directories:
        raise FileExistsError(
            "Owned preview path is unexpectedly a directory: "
            + ", ".join(str(path) for path in owned_directories)
        )
    if owned_existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing Chapter-5 table preview files: "
            + ", ".join(path.name for path in owned_existing)
        )
    output = prepare_output_directory(repository_root, output, force=force)
    preview.mkdir(parents=True, exist_ok=True)
    if force:
        for path in owned_existing:
            path.unlink()
    return output, preview


def _preview_harness(
    preview_dir: Path,
    output_dir: Path,
    tables: list[RenderedTable],
    width_mm: float,
    *,
    presentation_role: str = "canonical",
) -> str:
    relative_tables = Path(os.path.relpath(output_dir, preview_dir)).as_posix()
    heading = (
        "% Minimal Chapter-5 R6 table preview harness."
        if presentation_role == "canonical"
        else "% Minimal engineering table preview harness; not thesis evidence."
    )
    lines = [
        heading,
        r"\documentclass[a4paper]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[ngerman]{babel}",
        rf"\usepackage[a4paper,textwidth={width_mm:g}mm,top=15mm,bottom=15mm]{{geometry}}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{threeparttable}",
        r"\usepackage{tabularx}",
        r"\usepackage{longtable}",
        r"\usepackage{siunitx}",
        r"\sisetup{output-decimal-marker={,},group-separator={.}}",
        r"\begin{document}",
    ]
    for table in tables:
        lines.append(
            rf"\input{{{relative_tables}/{table.role}/{table.stem}.tex}}"
        )
        lines.append(r"\clearpage")
    lines.extend([r"\end{document}", ""])
    return "\n".join(lines)


def _compile_preview(
    preview_dir: Path,
    harness: Path,
    *,
    presentation_role: str = "canonical",
) -> dict[str, object]:
    schema = (
        "fraud_detection.chapter5_latex_preview.r6.v1"
        if presentation_role == "canonical"
        else "fraud_detection.engineering_latex_preview.v1"
    )
    engine: str | None = None
    for candidate in ("lualatex", "xelatex", "pdflatex", "tectonic"):
        if shutil.which(candidate):
            engine = candidate
            break
    if engine is None:
        return {
            "schema": schema,
            "status": "SKIPPED_NO_ENGINE",
            "engine": None,
            "harness": harness.name,
            "pdf": None,
            "required_packages": [
                "array",
                "booktabs",
                "threeparttable",
                "tabularx",
                "longtable",
                "siunitx",
            ],
        }
    if engine == "tectonic":
        command = [engine, "--outdir", str(preview_dir), harness.name]
    else:
        command = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            harness.name,
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "SOURCE_DATE_EPOCH": "0",
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
        }
    )
    result = subprocess.run(
        command,
        cwd=preview_dir,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    pdf = preview_dir / f"{harness.stem}.pdf"
    status = "COMPILED" if result.returncode == 0 and pdf.is_file() else "FAILED"
    payload: dict[str, object] = {
        "schema": schema,
        "status": status,
        "engine": engine,
        "harness": harness.name,
        "pdf": pdf.name if pdf.is_file() else None,
        "required_packages": [
            "array",
            "booktabs",
            "threeparttable",
            "tabularx",
            "longtable",
            "siunitx",
        ],
    }
    for suffix in (".aux", ".log", ".out"):
        auxiliary = preview_dir / f"{harness.stem}{suffix}"
        if auxiliary.is_file():
            auxiliary.unlink()
    if status == "FAILED":
        payload["return_code"] = result.returncode
    return payload


def render(
    repository_root: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    preview_dir: Path | None = None,
    width_mm: float = 160.0,
    force: bool = False,
    event_sink: _EventSink | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not 80.0 <= float(width_mm) <= 300.0:
        raise ValueError("--width-mm must be between 80 and 300.")
    selected_preview = (
        preview_dir
        if preview_dir is not None
        else _default_preview_dir(repository_root, output_dir)
    )
    _emit_status(
        event_sink,
        "INFO",
        "render-tables-start",
        data_dir=data_dir,
        output_dir=output_dir,
        preview_dir=selected_preview,
        width_mm=width_mm,
        utc=_status_utc_now(),
    )
    resolved_data = data_dir.resolve()
    requested_output = require_generated_path(repository_root, output_dir)
    requested_preview = require_generated_path(
        repository_root, selected_preview
    )
    if _paths_overlap(resolved_data, requested_output) or _paths_overlap(
        resolved_data, requested_preview
    ):
        raise ValueError(
            "Table and preview outputs must be disjoint from presentation data."
        )
    if _paths_overlap(requested_output, requested_preview):
        raise ValueError(
            "Table and preview output directories must be disjoint."
        )
    store = TableStore(data_dir)
    output, preview = _preflight_output_directories(
        repository_root,
        output_dir,
        selected_preview,
        force=force,
    )
    engineering = store.presentation_role == "engineering"
    tables = (
        [_render_engineering_t1(store)]
        if engineering
        else [renderer(store) for renderer in TABLE_RENDERERS]
    )
    expected_table_count = 1 if engineering else EXPECTED_TABLE_COUNT
    expected_file_count = 2 if engineering else EXPECTED_TABLE_FILE_COUNT
    _emit_status(
        event_sink,
        "PASS",
        "render-tables-input-validation",
        tables_expected=expected_table_count,
        files_expected=expected_file_count,
    )
    generated: list[Path] = []
    rendered: list[dict[str, object]] = []
    for completed, table in enumerate(tables, start=1):
        role_dir = output / table.role
        csv_path = role_dir / f"{table.stem}.csv"
        tex_path = role_dir / f"{table.stem}.tex"
        _write_display_csv(csv_path, table.display_header, table.display_rows)
        _write_latex(tex_path, table.latex)
        generated.extend([csv_path, tex_path])
        _emit_status(
            event_sink,
            "INFO",
            "render-table",
            completed=completed,
            total=expected_table_count,
            name=table.stem,
            files_completed=completed * 2,
            files_total=expected_file_count,
        )
        rendered.append(
            {
                "artifact_id": table.stem,
                "role": table.role,
                "sources": [
                    source if "/" in source else f"tables/{source}"
                    for source in table.sources
                ],
                "display_row_count": len(table.display_rows),
            }
        )

    harness_name = (
        "engineering_tables_preview.tex"
        if engineering
        else "ch5_tables_preview.tex"
    )
    harness_path = preview / harness_name
    _write_latex(
        harness_path,
        _preview_harness(
            preview,
            output,
            tables,
            float(width_mm),
            presentation_role=str(store.presentation_role),
        ),
    )
    preview_status = _compile_preview(
        preview,
        harness_path,
        presentation_role=str(store.presentation_role),
    )
    preview_status_path = preview / "LATEX_PREVIEW_STATUS.json"
    write_json(preview_status_path, preview_status)
    if preview_status["status"] == "COMPILED":
        _emit_status(
            event_sink,
            "PASS",
            "table-preview",
            status="COMPILED",
            engine=preview_status["engine"],
            output=preview / str(preview_status["pdf"]),
        )
    elif preview_status["status"] == "SKIPPED_NO_ENGINE":
        _emit_status(
            event_sink,
            "WARN",
            "table-preview",
            status="SKIPPED_NO_ENGINE",
        )
    else:
        _emit_status(
            event_sink,
            "FAIL",
            "table-preview",
            status="FAILED",
        )
    if preview_status["status"] == "FAILED":
        raise RuntimeError(
            "LaTeX preview compilation failed; inspect the generated harness."
        )

    manifest_path = output / "TABLE_RENDER_MANIFEST.json"
    if engineering:
        manifest = {
            "schema": "fraud_detection.engineering_table_render.v1",
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
            "tables": rendered,
            "outputs": file_inventory(output, generated),
            "logical_table_count": 1,
            "rendered_file_count": 2,
            "width_mm": float(width_mm),
            "latex_preview": {
                "status": preview_status["status"],
                "engine": preview_status["engine"],
            },
            "undefined_sample_sd_marker": "en dash",
            "winner_ranking_rendered": False,
            "significance_columns_rendered": False,
            "model_fit_performed": False,
            "model_scoring_performed": False,
            "data_derivation_performed": False,
        }
    else:
        manifest = {
            "schema": "fraud_detection.chapter5_table_render.r6.v1",
            "status": "PASS",
            "inputs": sorted(store.read_paths),
            "tables": rendered,
            "outputs": file_inventory(output, generated),
            "main_table_count": sum(
                table.role == "main" for table in tables
            ),
            "appendix_table_count": sum(
                table.role == "appendix" for table in tables
            ),
            "width_mm": float(width_mm),
            "latex_preview": {
                "status": preview_status["status"],
                "engine": preview_status["engine"],
            },
            "rank_segment_rendered": False,
            "historical_near_tie_rendered": False,
            "practical_profile_rendered": False,
            "winner_ranking_rendered": False,
            "significance_columns_rendered": False,
            "model_fit_performed": False,
            "model_scoring_performed": False,
            "data_derivation_performed": False,
        }
    write_json(manifest_path, manifest)
    _emit_status(
        event_sink,
        "PASS",
        "render-tables-complete",
        tables=expected_table_count,
        files=expected_file_count,
        manifest=manifest_path,
        elapsed_seconds=time.perf_counter() - started,
    )
    return manifest
