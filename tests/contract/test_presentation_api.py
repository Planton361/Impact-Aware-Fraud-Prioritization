import hashlib
import json
import math
import os
import subprocess
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from fraud_detection.presentation import (
    PresentationConfig,
    PresentationError,
    PresentationResult,
    build_presentation,
)

pytestmark = pytest.mark.contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_group(relative: str) -> str:
    root = relative.split("/", maxsplit=1)[0]
    return {
        "comparison": "integrity",
        "diagnostics": "aggregation",
        "figure_data": "aggregation",
        "final_outer_run": "final_outer",
        "preflight": "preflight",
    }[root]


def _write_completed_run(root: Path, profile: str) -> dict[str, Any]:
    from fraud_detection.experiment.config import (
        EXPECTED_DEDUPLICATED_SHA256,
        EXPECTED_RAW_SHA256,
        resolve_experiment_profile,
    )

    effective = resolve_experiment_profile(profile)
    relative_paths = {
        "comparison/checksums.sha256",
        "diagnostics/global_metrics_seedwise.csv",
        "figure_data/all_budget_matched_results.csv",
        "final_outer_run/final_outer_manifest.json",
        "preflight/preflight_validation.json",
        *(
            f"final_outer_run/seed_{seed}/ranking_dump.csv"
            for seed in effective.seeds
        ),
    }
    for relative in sorted(relative_paths - {"comparison/checksums.sha256"}):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n", encoding="utf-8")
    checksum_path = root / "comparison" / "checksums.sha256"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    if effective.data_source_kind == "synthetic":
        source_rows, source_fraud, source_legitimate = 5_000, 100, 4_900
        rows, fraud, legitimate = source_rows, source_fraud, source_legitimate
        removed_duplicates = 0
        data_identity = "a" * 64
        preflight_data = {
            "data_source_kind": "synthetic",
            "synthetic_generator_schema": (
                "fraud_detection.synthetic_engineering.v1"
            ),
            "synthetic_generation_seed": effective.synthetic_generation_seed,
            "synthetic_requested_row_count": effective.synthetic_row_target,
            "synthetic_generated_row_count": rows,
            "synthetic_generated_fraud_count": fraud,
            "synthetic_generated_legitimate_count": legitimate,
            "deterministic_data_identity": data_identity,
            "evidence_classification": effective.evidence_classification,
            "evidence_boundary": (
                "synthetic engineering data; not thesis evidence; not "
                "comparable with canonical empirical results"
            ),
            "deduplication_keep": "first",
            "deduplication_before_split": True,
        }
    else:
        source_rows, source_fraud, source_legitimate = 284_807, 492, 284_315
        rows, fraud, legitimate = 283_726, 473, 283_253
        removed_duplicates = 1_081
        data_identity = EXPECTED_DEDUPLICATED_SHA256
        preflight_data = {
            "raw_path": "data/creditcard.csv",
            "expected_raw_sha256": EXPECTED_RAW_SHA256,
            "actual_raw_sha256": EXPECTED_RAW_SHA256,
            "deduplication_keep": "first",
            "deduplication_before_split": True,
            "rows": rows,
            "class_0": legitimate,
            "class_1": fraud,
            "expected_deduplicated_dataframe_sha256": (
                EXPECTED_DEDUPLICATED_SHA256
            ),
            "actual_deduplicated_dataframe_sha256": (
                EXPECTED_DEDUPLICATED_SHA256
            ),
        }
    preflight = {
        "schema": "ranker_gain_validation.preflight.v2",
        "status": "PASS",
        "data": preflight_data,
        "locked_definitions": {
            "outer_seeds": list(effective.seeds),
            "target_budgets": list(effective.target_budgets),
            "ranker_scope": "candidate_rerank",
            "candidate_pool_size": effective.candidate_pool_size,
        },
    }
    (root / "preflight" / "preflight_validation.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )
    final_manifest = {
        "schema": "ranker_gain_validation.final_outer.v1",
        "status": "PASS",
        "outer_seed_count": len(effective.seeds),
        "target_budget_count": len(effective.target_budgets),
        "ranking_dump_count": len(effective.seeds),
        "ranker_scope": "candidate_rerank",
    }
    (root / "final_outer_run" / "final_outer_manifest.json").write_text(
        json.dumps(final_manifest), encoding="utf-8"
    )
    checksum_path.write_text(
        "".join(
            f"{_sha256(root / relative)}  {relative}\n"
            for relative in sorted(
                relative_paths - {"comparison/checksums.sha256"}
            )
        ),
        encoding="utf-8",
    )
    data_summary: dict[str, object] = {
        "source_kind": effective.data_source_kind,
        "data_identity": data_identity,
        "source_counts": {
            "kind": (
                "generated"
                if effective.data_source_kind == "synthetic"
                else "raw"
            ),
            "rows": source_rows,
            "fraud": source_fraud,
            "legitimate": source_legitimate,
        },
        "deduplicated_counts": {
            "rows": rows,
            "fraud": fraud,
            "legitimate": legitimate,
        },
        "removed_duplicate_count": removed_duplicates,
    }
    if effective.data_source_kind == "synthetic":
        data_summary["synthetic"] = {
            "generator_schema": "fraud_detection.synthetic_engineering.v1",
            "generation_seed": effective.synthetic_generation_seed,
            "requested_row_count": effective.synthetic_row_target,
        }
    manifest: dict[str, Any] = {
        "schema": "fraud_detection.run_manifest.v1",
        "status": "COMPLETE",
        "profile": effective.profile_name,
        "evidence_classification": effective.evidence_classification,
        "completed_phases": [
            "preflight",
            "inner_selection",
            "selection_freeze",
            "final_outer",
            "aggregation",
            "qa",
        ],
        "effective_config": effective.as_dict(),
        "data_summary": data_summary,
        "produced_artifacts": [
            {
                "path": relative,
                "group": _artifact_group(relative),
                "format": Path(relative).suffix.removeprefix("."),
            }
            for relative in sorted(relative_paths)
        ],
    }
    (root / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _seed_metrics(profile: str) -> pd.DataFrame:
    from fraud_detection.experiment.config import resolve_experiment_profile
    from fraud_detection.presentation import METHOD_ORDER

    effective = resolve_experiment_profile(profile)
    rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(effective.seeds):
        for budget in effective.target_budgets:
            for method_index, method in enumerate(METHOD_ORDER):
                rows.append(
                    {
                        "seed": seed,
                        "target_budget": budget,
                        "method_family": method,
                        "score_path": (
                            method if method_index == 0 else f"{method}_k{budget}"
                        ),
                        "prevented_loss_ratio_at_k": (
                            0.10
                            + seed_index * 0.01
                            + method_index * 0.02
                            + budget / 10_000
                        ),
                        "frauds_at_k": budget / 20 + method_index + seed_index,
                        "precision_at_k": (
                            0.20 + method_index * 0.01 + seed_index * 0.02
                        ),
                        "recall_at_k": (
                            0.30 + method_index * 0.01 + seed_index * 0.02
                        ),
                        "fraud_amount_sum_at_k": 10.0 + budget + method_index,
                        "amount_ndcg_at_k": 0.40 + method_index * 0.01,
                        "q90_captured_fraud_count": 1 + method_index,
                        "q90_captured_ratio_at_k": 0.10 + method_index * 0.01,
                        "selected_gain": "linear",
                        "selection_status": "SELECTED",
                    }
                )
    return pd.DataFrame(rows)


def _ranking_dump_fixture(
    *,
    seeds: tuple[int, ...] = (42,),
    budgets: tuple[int, ...] = (20, 50, 100),
    ranking_rows: int = 6,
    candidate_pool_size: int = 2,
) -> pd.DataFrame:
    from fraud_detection.presentation import METHOD_BCE, METHOD_ORDER

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for budget in budgets:
            for method in METHOD_ORDER:
                for position in range(ranking_rows):
                    candidate = position < candidate_pool_size
                    probability = 0.9 - position / 10
                    rows.append(
                        {
                            "seed": seed,
                            "target_budget": budget,
                            "method_family": method,
                            "score_path": (
                                method
                                if method == METHOD_BCE
                                else f"{method}_k{budget}"
                            ),
                            "row_index": 10_000 + position,
                            "original_position": position,
                            "candidate_flag": candidate,
                            "candidate_pool_size": candidate_pool_size,
                            "candidate_pool_sha256": "a" * 64,
                            "p_fraud": probability,
                            "raw_ranker_score": (
                                probability
                                if method == METHOD_BCE or candidate
                                else float("nan")
                            ),
                            "final_rank_position": position + 1,
                            "bce_rank_position": position + 1,
                            "priority_order_score": ranking_rows - position,
                            "Class": int(position == 0),
                            "Amount": float(position + 1),
                            "selected_gain": "linear",
                            "selection_status": "SELECTED",
                            "truncation_level": budget + 3,
                            "final_n_estimators": 1,
                            "score_type": (
                                "fraud_probability"
                                if method == METHOD_BCE
                                else "ordinal_candidate_postprocessing"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _global_curve_fixture(
    *,
    brier_values: tuple[object, ...] = (0.01,),
) -> tuple[dict[tuple[int, int, str], pd.DataFrame], pd.DataFrame]:
    from fraud_detection.presentation import METHOD_BCE, METHOD_ORDER

    ranking = pd.DataFrame(
        {
            "Class": [1, 0],
            "final_rank_position": [1, 2],
            "p_fraud": [0.9, 0.1],
            "candidate_flag": [True, True],
        }
    )
    groups = {
        (42, 50, method): ranking.copy()
        for method in METHOD_ORDER
    }
    diagnostics: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        diagnostics.extend(
            [
                {
                    "seed": 42,
                    "target_budget": 50,
                    "method_family": method,
                    "metric": "roc_auc_of_final_order",
                    "value": 1.0,
                },
                {
                    "seed": 42,
                    "target_budget": 50,
                    "method_family": method,
                    "metric": "average_precision_of_final_order",
                    "value": 1.0,
                },
            ]
        )
        if method == METHOD_BCE:
            diagnostics.extend(
                {
                    "seed": 42,
                    "target_budget": 50,
                    "method_family": method,
                    "metric": "brier_score_probability",
                    "value": value,
                }
                for value in brier_values
            )
    return groups, pd.DataFrame(diagnostics)


def _accept_profile_data_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fraud_detection.presentation.catalog import (
        CANONICAL_ARTIFACT_IDS,
        ENGINEERING_ARTIFACT_IDS,
    )
    from fraud_detection.presentation.preparation import data
    from fraud_detection.presentation.preparation.derivations import (
        CANONICAL_OUTPUT_PATHS,
        ENGINEERING_OUTPUT_PATHS,
    )
    from fraud_detection.presentation.rendering import figures, tables

    contexts: dict[str, data.PresentationInputContext] = {}
    for profile in ("canonical", "mini-real", "smoke-synthetic"):
        experiment = tmp_path / "fixture-inputs" / profile
        _write_completed_run(experiment, profile)
        contexts[profile] = data.load_presentation_input_context(experiment)
    fixture_metrics = {
        context.experiment_root: _seed_metrics(profile)
        for profile, context in contexts.items()
    }
    original_store_csv = data.ExperimentStore.csv

    def fixture_csv(
        store: data.ExperimentStore,
        relative: str,
    ) -> pd.DataFrame:
        store.path(relative)
        if relative == "figure_data/all_budget_matched_results.csv":
            return fixture_metrics[store.root].copy()
        if relative == "diagnostics/global_metrics_seedwise.csv":
            return pd.DataFrame()
        return original_store_csv(store, relative)

    def fixture_groups(
        store: data.ExperimentStore,
        seeds: tuple[int, ...],
        _budgets: tuple[int, ...],
        _expected_pool_size: int,
    ) -> dict[tuple[int, int, str], pd.DataFrame]:
        for seed in seeds:
            store.path(f"final_outer_run/seed_{seed}/ranking_dump.csv")
        return {}

    observed_pool_sizes: list[int] = []

    def fixture_pool_validation(
        _groups: dict[tuple[int, int, str], pd.DataFrame],
        _seeds: tuple[int, ...],
        _budgets: tuple[int, ...],
        expected_pool_size: int,
    ) -> None:
        observed_pool_sizes.append(expected_pool_size)

    def canonical_fixture_outputs(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        return {
            path: pd.DataFrame({"canonical_contract": [index]})
            for index, path in enumerate(CANONICAL_OUTPUT_PATHS)
        }

    fixture_manifests: dict[str, dict[str, object]] = {}
    with monkeypatch.context() as fixture_patch:
        fixture_patch.setattr(data.ExperimentStore, "csv", fixture_csv)
        fixture_patch.setattr(data, "_load_ranking_groups", fixture_groups)
        fixture_patch.setattr(
            data, "_validate_candidate_pools", fixture_pool_validation
        )
        fixture_patch.setattr(data, "derive_all", canonical_fixture_outputs)
        for profile, context in contexts.items():
            output_root = tmp_path / "generated" / f"fixture-{profile}"
            manifest = data.build(
                tmp_path,
                context.experiment_root,
                output_root,
                input_context=context,
                presentation_role=context.presentation_role,
            )
            fixture_manifests[profile] = manifest
            expected_paths = (
                CANONICAL_OUTPUT_PATHS
                if profile == "canonical"
                else ENGINEERING_OUTPUT_PATHS
            )
            assert manifest["profile"] == profile
            assert manifest["presentation_role"] == context.presentation_role
            assert manifest["evidence_classification"] == (
                context.evidence_classification
            )
            assert manifest["data_source_kind"] == context.data_source_kind
            assert manifest["seeds"] == list(context.seeds)
            assert manifest["budgets"] == list(context.target_budgets)
            assert manifest["candidate_pool_size"] == (
                context.candidate_pool_size
            )
            assert manifest["data_summary"] == _rewriteable_data_summary(
                context.data_summary
            )
            assert manifest["selected_catalog_artifact_ids"] == list(
                CANONICAL_ARTIFACT_IDS
                if profile == "canonical"
                else ENGINEERING_ARTIFACT_IDS
            )
            if profile == "canonical":
                assert "evidence_statement" not in manifest
                assert "comparability_boundary" not in manifest
            else:
                assert "not thesis evidence" in str(
                    manifest["evidence_statement"]
                )
                assert manifest["comparability_boundary"] == (
                    "not comparable with canonical empirical results"
                )
            produced = manifest["outputs"]
            assert isinstance(produced, list)
            assert [row["path"] for row in produced] == sorted(expected_paths)
            assert all(len(str(row["sha256"])) == 64 for row in produced)
            assert not (output_root / "figures").exists()
            assert not (output_root / "tables").exists()
            assert not (output_root / "preview").exists()
    assert observed_pool_sizes == [1000, 1000, 200]
    for profile, expected in (
        ("canonical", 29),
        ("mini-real", 2),
        ("smoke-synthetic", 2),
    ):
        outputs = fixture_manifests[profile]["outputs"]
        assert isinstance(outputs, list)
        assert len(outputs) == expected

    visible_figure_text: dict[str, str] = {}

    def valid_colourbar_tick_sign(value: float, text: str) -> bool:
        unsigned = text.removeprefix("-").removeprefix("−")
        _integer, separator, fractional = unsigned.partition(",")
        decimals = len(fractional) if separator else 0
        normalized = round(float(value), decimals)
        has_minus_sign = text.startswith(("-", "−"))
        if normalized == 0.0:
            return not has_minus_sign
        return has_minus_sign == (normalized < 0.0)

    assert valid_colourbar_tick_sign(-0.060, "-0,060")
    assert valid_colourbar_tick_sign(-0.191, "-0,191")
    assert valid_colourbar_tick_sign(0.0, "0,000")
    assert not valid_colourbar_tick_sign(0.0, "-0,000")

    def assert_engineering_tick_contract(figure: Any) -> None:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        heatmap_boxes = [
            axis.get_window_extent(renderer=renderer)
            for axis in figure.axes[:2]
        ]
        expected_main_title = (
            "Engineering-Heatmap der Seed-Budget-Deltas gegenüber BCE"
        )
        assert figure._suptitle is not None
        observed_main_title = unicodedata.normalize(
            "NFC", figure._suptitle.get_text()
        )
        assert observed_main_title == unicodedata.normalize(
            "NFC", expected_main_title
        )
        assert "gegenüber BCE" in observed_main_title
        expected_titles = ["(a) PLR-Delta", "(b) Fraud@k-Delta"]
        observed_titles = [
            unicodedata.normalize("NFC", axis.get_title())
            for axis in figure.axes[:2]
        ]
        assert observed_titles == expected_titles
        assert all("BCE" not in title for title in observed_titles)
        panel_title_boxes = [
            axis.title.get_window_extent(renderer=renderer)
            for axis in figure.axes[:2]
        ]
        assert len(figure.axes) == 5
        colourbar_axes = figure.axes[2:4]
        footer_axis = figure.axes[4]
        assert not footer_axis.axison
        assert not footer_axis.get_frame_on()
        assert not footer_axis.get_xticks().size
        assert not footer_axis.get_yticks().size
        footer_box = footer_axis.get_window_extent(renderer=renderer)
        x_label_boxes = [
            axis.xaxis.label.get_window_extent(renderer=renderer)
            for axis in figure.axes[:2]
        ]
        assert not any(
            footer_box.overlaps(box) for box in heatmap_boxes + x_label_boxes
        )
        footer_texts = [
            artist
            for artist in footer_axis.texts
            if artist.get_visible() and artist.get_text().strip()
        ]
        assert len(footer_texts) == 2
        for artist in footer_texts:
            bbox = artist.get_window_extent(renderer=renderer)
            assert not any(
                bbox.overlaps(box) for box in heatmap_boxes + x_label_boxes
            )
        separate_markers = [
            artist
            for artist in figures._visible_text_artists(figure)
            if artist.get_text().strip() in {"a", "b", "(a)", "(b)"}
        ]
        assert not separate_markers
        for title_box in panel_title_boxes:
            assert not any(
                title_box.overlaps(box)
                for box in heatmap_boxes
                + [
                    axis.get_window_extent(renderer=renderer)
                    for axis in colourbar_axes
                ]
            )
        for colourbar_axis in colourbar_axes:
            assert not colourbar_axis.get_title()
            assert not colourbar_axis.yaxis.label.get_text()
            assert isinstance(
                colourbar_axis.yaxis.get_major_locator(),
                figures.FixedLocator,
            )
            ticks = colourbar_axis.get_yticks()
            assert all(math.isfinite(float(tick)) for tick in ticks)
            assert any(math.isclose(float(tick), 0.0) for tick in ticks)
            labels = [
                label
                for label in colourbar_axis.get_yticklabels()
                if label.get_visible()
            ]
            assert len(labels) == len(ticks)
            for tick, label in zip(ticks, labels, strict=True):
                text = label.get_text()
                display = label.get_transform().transform(
                    label.get_position()
                )
                bbox = label.get_window_extent(renderer=renderer)
                assert text
                assert valid_colourbar_tick_sign(float(tick), text)
                assert all(
                    math.isfinite(float(value))
                    for value in (
                        *display,
                        bbox.x0,
                        bbox.y0,
                        bbox.x1,
                        bbox.y1,
                    )
                )
                assert bbox.x0 >= canvas.x0 - 2.0
                assert bbox.y0 >= canvas.y0 - 2.0
                assert bbox.x1 <= canvas.x1 + 2.0
                assert bbox.y1 <= canvas.y1 + 2.0
                assert not any(
                    bbox.overlaps(heatmap_box)
                    for heatmap_box in heatmap_boxes
                )
                assert not any(
                    bbox.overlaps(title_box)
                    for title_box in panel_title_boxes
                )

    def fixture_figure_save(
        figure: object,
        output: Path,
        stem: str,
        **_metadata: object,
    ) -> list[Path]:
        assert stem == "engineering_f1_seed_budget_delta_heatmap"
        assert_engineering_tick_contract(figure)
        figures._validate_figure_canvas(figure, stem)
        visible_figure_text[output.parents[1].name] = "\n".join(
            artist.get_text()
            for artist in figure.findobj(match=figures.Text)  # type: ignore[attr-defined]
            if artist.get_text()
        )
        figures.plt.close(figure)
        output.mkdir(parents=True, exist_ok=True)
        paths = [output / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")]
        for path in paths:
            path.write_text(f"fixture {path.suffix}\n", encoding="utf-8")
        return paths

    def skipped_preview(
        _preview_dir: Path,
        harness: Path,
        **_role: object,
    ) -> dict[str, object]:
        return {
            "schema": "fixture.preview.v1",
            "status": "SKIPPED_NO_ENGINE",
            "engine": None,
            "harness": harness.name,
            "pdf": None,
            "required_packages": [],
        }

    with monkeypatch.context() as render_patch:
        render_patch.setattr(figures, "_save_figure", fixture_figure_save)
        render_patch.setattr(tables, "_compile_preview", skipped_preview)
        for profile in ("smoke-synthetic", "mini-real"):
            prepared_root = tmp_path / "generated" / f"fixture-{profile}"
            figure_manifest = figures.render(
                tmp_path,
                prepared_root / "data",
                tmp_path / "generated" / f"rendered-{profile}" / "figures",
            )
            table_manifest = tables.render(
                tmp_path,
                prepared_root / "data",
                tmp_path / "generated" / f"rendered-{profile}" / "tables",
                preview_dir=(
                    tmp_path
                    / "generated"
                    / f"rendered-{profile}"
                    / "preview"
                ),
            )
            assert figure_manifest["presentation_role"] == "engineering"
            assert figure_manifest["profile"] == profile
            assert figure_manifest["logical_figure_count"] == 1
            assert figure_manifest["rendered_file_count"] == 3
            assert [row["path"] for row in figure_manifest["outputs"]] == [
                "engineering/engineering_f1_seed_budget_delta_heatmap.pdf",
                "engineering/engineering_f1_seed_budget_delta_heatmap.png",
                "engineering/engineering_f1_seed_budget_delta_heatmap.svg",
            ]
            assert table_manifest["presentation_role"] == "engineering"
            assert table_manifest["profile"] == profile
            assert table_manifest["logical_table_count"] == 1
            assert table_manifest["rendered_file_count"] == 2
            assert table_manifest["latex_preview"] == {
                "status": "SKIPPED_NO_ENGINE",
                "engine": None,
            }
            assert [row["path"] for row in table_manifest["outputs"]] == [
                "engineering/engineering_t1_central_topk_summary.csv",
                "engineering/engineering_t1_central_topk_summary.tex",
            ]
            warning = visible_figure_text[f"rendered-{profile}"]
            assert profile in warning
            assert "not thesis evidence" in warning
            assert "not comparable with canonical empirical results" in warning
            if profile == "smoke-synthetic":
                assert "synthetic engineering data" in warning
            rendered_root = tmp_path / "generated" / f"rendered-{profile}"
            assert not list(rendered_root.rglob("ch5_*"))
            assert not list(rendered_root.rglob("app_*"))
            table_csv = pd.read_csv(
                rendered_root
                / "tables"
                / "engineering"
                / "engineering_t1_central_topk_summary.csv",
                sep=";",
                dtype=str,
                keep_default_na=False,
            )
            assert len(table_csv) == 12
            assert table_csv["Profile"].eq(profile).all()
            assert table_csv["Evidence"].str.contains(
                "not thesis evidence", regex=False
            ).all()
            sd_columns = [
                column for column in table_csv if column.endswith("sample SD")
            ]
            if profile == "smoke-synthetic":
                assert table_csv[sd_columns].eq(tables.NA_DISPLAY).all().all()
                assert "NaN" not in table_csv.to_csv(index=False)
                assert "± 0" not in table_csv.to_csv(index=False)
            else:
                assert not table_csv[sd_columns].eq(
                    tables.NA_DISPLAY
                ).any().any()
            latex = (
                rendered_root
                / "tables"
                / "engineering"
                / "engineering_t1_central_topk_summary.tex"
            ).read_text(encoding="utf-8")
            assert profile in latex
            assert "not thesis evidence" in latex
            assert "not comparable with canonical empirical results" in latex
            if profile == "smoke-synthetic":
                assert tables.NA_TEX in latex

    smoke_data_root = (
        tmp_path / "generated" / "fixture-smoke-synthetic" / "data"
    )
    heatmap_path = (
        smoke_data_root
        / "engineering"
        / "figures"
        / "engineering_seed_budget_delta_heatmap.csv"
    )
    table_path = (
        smoke_data_root
        / "engineering"
        / "tables"
        / "engineering_central_topk_summary.csv"
    )
    manifest_path = smoke_data_root / "PRESENTATION_DATA_MANIFEST.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_heatmap = heatmap_path.read_bytes()
    original_table = table_path.read_bytes()

    def refresh_hash(relative: str, path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["outputs"]:
            if row["path"] == relative:
                row["sha256"] = _sha256(path)
                row["size_bytes"] = path.stat().st_size
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    narrow = pd.read_csv(heatmap_path)
    narrow["delta_plr_vs_bce"] = (
        -0.200,
        -0.100,
        0.000,
        0.100,
        0.200,
        -0.050,
        0.150,
        -0.150,
        0.050,
    )
    narrow["plr_at_k"] = (
        narrow["bce_plr_at_k"] + narrow["delta_plr_vs_bce"]
    )
    narrow.to_csv(heatmap_path, index=False, lineterminator="\n")
    refresh_hash(
        "engineering/figures/engineering_seed_budget_delta_heatmap.csv",
        heatmap_path,
    )
    narrow_figure = figures._engineering_heatmap_figure(
        figures.FigureStore(smoke_data_root), 160.0
    )
    assert_engineering_tick_contract(narrow_figure)
    plr_ticks = narrow_figure.axes[2].get_yticks()
    assert math.isclose(float(plr_ticks[0]), -0.200)
    assert math.isclose(float(plr_ticks[-1]), 0.200)
    plr_labels = [
        label.get_text()
        for label in narrow_figure.axes[2].get_yticklabels()
        if label.get_visible()
    ]
    assert plr_labels[0] == "-0,200"
    assert "0,000" in plr_labels
    assert plr_labels[-1] == "0,200"
    figures._validate_figure_canvas(
        narrow_figure,
        "engineering_f1_seed_budget_delta_heatmap",
    )
    figures.plt.close(narrow_figure)
    heatmap_path.write_bytes(original_heatmap)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    heatmap = pd.read_csv(heatmap_path)
    heatmap.iloc[1:].to_csv(heatmap_path, index=False, lineterminator="\n")
    refresh_hash(
        "engineering/figures/engineering_seed_budget_delta_heatmap.csv",
        heatmap_path,
    )
    with pytest.raises(RuntimeError, match="heatmap grid is incomplete"):
        figures._engineering_heatmap_figure(
            figures.FigureStore(smoke_data_root), 160.0
        )
    heatmap_path.write_bytes(original_heatmap)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    summary = pd.read_csv(table_path)
    summary.iloc[1:].to_csv(table_path, index=False, lineterminator="\n")
    refresh_hash(
        "engineering/tables/engineering_central_topk_summary.csv",
        table_path,
    )
    with pytest.raises(RuntimeError, match="table grid mismatch"):
        tables._render_engineering_t1(tables.TableStore(smoke_data_root))
    table_path.write_bytes(original_table)
    registered_store = tables.TableStore(smoke_data_root)
    with pytest.raises(RuntimeError, match="Unregistered table data"):
        registered_store.csv("engineering/tables/unregistered.csv")
    mixed_manifest = json.loads(original_manifest)
    mixed_manifest["selected_catalog_artifact_ids"] = [
        "engineering_f1_seed_budget_delta_heatmap",
        "ch5_t1_central_topk_results",
    ]
    manifest_path.write_text(json.dumps(mixed_manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="role or catalog is invalid"):
        tables.TableStore(smoke_data_root)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    summary = pd.read_csv(table_path)
    summary.loc[0, "plr_sample_sd"] = 0.0
    summary.to_csv(table_path, index=False, lineterminator="\n")
    refresh_hash(
        "engineering/tables/engineering_central_topk_summary.csv",
        table_path,
    )
    with pytest.raises(RuntimeError, match="must remain missing, not zero"):
        tables._render_engineering_t1(tables.TableStore(smoke_data_root))
    table_path.write_bytes(original_table)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    tampered = json.loads(original_manifest)
    tampered["schema"] = "wrong"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema is not the approved"):
        tables.TableStore(smoke_data_root)
    manifest_path.write_text(original_manifest, encoding="utf-8")
    table_path.write_text("tampered\n", encoding="utf-8")
    store = tables.TableStore(smoke_data_root)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.csv(
            "engineering/tables/engineering_central_topk_summary.csv"
        )
    table_path.write_bytes(original_table)


def _rewriteable_data_summary(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _rewriteable_data_summary(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_rewriteable_data_summary(item) for item in value]
    return value


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def test_presentation_package_import_is_lazy(tmp_path: Path) -> None:
    program = r"""
import sys
import fraud_detection.presentation as presentation

forbidden = (
    "matplotlib",
    "lightgbm",
    "scripts",
    "fraud_detection.experiment",
    "fraud_detection.presentation.preparation.data",
    "fraud_detection.presentation.rendering.figures",
    "fraud_detection.presentation.rendering.tables",
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == item or name.startswith(item + ".") for item in forbidden)
)
if loaded:
    raise SystemExit("eager presentation imports: " + ", ".join(loaded))

build_module = sys.modules.get("fraud_detection.presentation.build")
if build_module is None:
    raise SystemExit("public presentation build facade module was not imported")
checks = {
    "__all__": presentation.__all__ == [
        "PresentationConfig",
        "PresentationStepResult",
        "PresentationResult",
        "PresentationError",
        "build_presentation",
    ],
    "PresentationConfig": presentation.PresentationConfig
    is build_module.PresentationConfig,
    "PresentationStepResult": presentation.PresentationStepResult
    is build_module.PresentationStepResult,
    "PresentationError": presentation.PresentationError
    is build_module.PresentationError,
    "PresentationResult": presentation.PresentationResult
    is build_module.PresentationResult,
    "build_presentation": presentation.build_presentation
    is build_module.build_presentation,
    "repository_root_from absence": not hasattr(
        presentation, "repository_root_from"
    ),
    "FIGURE_STYLE absence": not hasattr(presentation, "FIGURE_STYLE"),
    "parse_int_csv absence": not hasattr(presentation, "parse_int_csv"),
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("public presentation API identity failure: " + ", ".join(failed))
"""
    source_root = Path(__file__).resolve().parents[2] / "src"
    mpl_config = tmp_path / "mplconfig"
    mpl_config.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source_root), environment.get("PYTHONPATH", ""))
        if value
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl_config)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    diagnostic = "\n".join(
        value.strip()
        for value in (completed.stdout, completed.stderr)
        if value.strip()
    )

    assert completed.returncode == 0, diagnostic
    assert mpl_config.resolve().is_relative_to(tmp_path.resolve())
    assert not (tmp_path / "generated").exists()
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "thesis_build").exists()
    assert not list(tmp_path.rglob("__pycache__"))


def test_build_presentation_runs_ordered_stages_with_one_event_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    modules_before = set(sys.modules)
    from fraud_detection.presentation.catalog import (
        CANONICAL_ARTIFACT_IDS,
        ENGINEERING_ARTIFACT_IDS,
        build_profile_selection_registry,
        build_selection_registry,
    )
    from fraud_detection.presentation.preparation import data
    from fraud_detection.presentation.preparation.derivations import (
        CANONICAL_OUTPUT_PATHS,
        ENGINEERING_OUTPUT_PATHS,
        derive_central_results,
        derive_engineering,
        derive_global_and_pool_curves,
        derive_tradeoff,
    )
    from fraud_detection.presentation.rendering import figures, tables

    newly_loaded = set(sys.modules) - modules_before
    forbidden_runtime_imports = (
        "lightgbm",
        "fraud_detection.experiment.execution",
        "fraud_detection.experiment.preparation",
        "fraud_detection.experiment.prioritization",
        "fraud_detection.experiment.comparison_paths",
        "fraud_detection.experiment.evaluation",
    )
    assert not [
        name
        for name in newly_loaded
        if any(
            name == item or name.startswith(item + ".")
            for item in forbidden_runtime_imports
        )
    ]

    expected_dimensions = {
        "canonical": ((42, 7, 13, 123, 202), (5, 10, 20, 50, 100, 200, 500), 1000),
        "mini-real": ((42, 7, 13), (20, 50, 100), 1000),
        "smoke-synthetic": ((42,), (20, 50, 100), 200),
    }
    contexts: dict[str, data.PresentationInputContext] = {}
    preflight_data_by_profile: dict[str, dict[str, object]] = {}
    for profile, expected in expected_dimensions.items():
        experiment = tmp_path / profile
        _write_completed_run(experiment, profile)
        context = data.load_presentation_input_context(experiment)
        contexts[profile] = context
        preflight = json.loads(
            (
                experiment / "preflight" / "preflight_validation.json"
            ).read_text(encoding="utf-8")
        )
        preflight_data = preflight["data"]
        assert isinstance(preflight_data, dict)
        preflight_data_by_profile[profile] = preflight_data
        normalized = data._normalize_preflight_data(preflight_data, context)
        assert context.profile == profile
        assert context.presentation_role == (
            "canonical" if profile == "canonical" else "engineering"
        )
        assert (
            context.seeds,
            context.target_budgets,
            context.candidate_pool_size,
        ) == expected
        assert context.primary_budgets == (20, 50, 100)
        assert context.evidence_classification == (
            context.effective_config.evidence_classification
        )
        assert context.data_source_kind == context.effective_config.data_source_kind
        assert context.inner_folds == context.effective_config.inner_folds
        assert context.bce_oof_folds == context.effective_config.bce_oof_folds
        assert context.experiment_root == experiment.resolve()
        assert data.CHECKSUM_MANIFEST_PATH in context.registered_artifact_paths
        assert not (experiment / ".git").exists()
        assert normalized.source_kind == context.data_source_kind
        assert normalized.data_identity == context.data_summary["data_identity"]
        with pytest.raises(FrozenInstanceError):
            context.profile = "canonical"  # type: ignore[misc]

    smoke_summary = data._normalize_preflight_data(
        preflight_data_by_profile["smoke-synthetic"],
        contexts["smoke-synthetic"],
    )
    assert (
        smoke_summary.source_rows,
        smoke_summary.source_fraud,
        smoke_summary.source_legitimate,
        smoke_summary.deduplicated_rows,
        smoke_summary.removed_duplicate_count,
    ) == (5_000, 100, 4_900, 5_000, 0)
    assert "raw_path" not in preflight_data_by_profile["smoke-synthetic"]

    ranking_dump = _ranking_dump_fixture()

    class RankingFixtureStore:
        def ranking(self, seed: int) -> pd.DataFrame:
            return ranking_dump.loc[ranking_dump["seed"].eq(seed)].copy()

    ranking_groups = data._load_ranking_groups(
        RankingFixtureStore(),  # type: ignore[arg-type]
        (42,),
        (20, 50, 100),
        2,
    )
    structural_missing = 0
    for (seed, budget, method), group in ranking_groups.items():
        candidate = group["candidate_flag"].astype(bool)
        raw = group["raw_ranker_score"]
        assert raw.loc[candidate].map(math.isfinite).all()
        if method == data.METHOD_BCE:
            assert raw.map(math.isfinite).all()
        else:
            assert raw.loc[~candidate].isna().all()
        structural_missing += data._validate_ranking_score_semantics(
            group,
            seed=seed,
            budget=budget,
            method=method,
            expected_pool_size=2,
        )
    assert structural_missing == 3 * 3 * (6 - 2)
    smoke_context = contexts["smoke-synthetic"]
    assert (
        (len(data.METHOD_ORDER) - 1)
        * len(smoke_context.target_budgets)
        * (1_000 - smoke_context.candidate_pool_size)
        == 7_200
    )

    p_only_key = (42, 20, data.METHOD_ORDER[1])
    valid_p_only = ranking_groups[p_only_key]
    candidate_row = valid_p_only.index[
        valid_p_only["candidate_flag"].astype(bool)
    ][0]
    non_candidate_row = valid_p_only.index[
        ~valid_p_only["candidate_flag"].astype(bool)
    ][0]
    invalid_rankings: list[tuple[pd.DataFrame, str]] = []
    candidate_missing = valid_p_only.copy()
    candidate_missing.loc[candidate_row, "raw_ranker_score"] = float("nan")
    invalid_rankings.append((candidate_missing, "candidate raw score is missing"))
    non_candidate_finite = valid_p_only.copy()
    non_candidate_finite.loc[non_candidate_row, "raw_ranker_score"] = 0.0
    invalid_rankings.append(
        (non_candidate_finite, "non-candidate raw score is unexpectedly finite")
    )
    for infinity in (float("inf"), float("-inf")):
        infinite = valid_p_only.copy()
        infinite.loc[candidate_row, "raw_ranker_score"] = infinity
        invalid_rankings.append((infinite, "raw score contains infinity"))
    wrong_pool = valid_p_only.copy()
    wrong_pool.loc[non_candidate_row, "candidate_flag"] = True
    invalid_rankings.append((wrong_pool, "candidate count differs"))
    global_missing = valid_p_only.copy()
    global_missing.loc[candidate_row, "priority_order_score"] = float("nan")
    invalid_rankings.append(
        (global_missing, "globally required numeric value is non-finite")
    )
    placeholder = valid_p_only.copy()
    placeholder["raw_ranker_score"] = placeholder["raw_ranker_score"].astype(object)
    placeholder.loc[candidate_row, "raw_ranker_score"] = "missing"
    invalid_rankings.append((placeholder, "raw score column is not numeric"))
    incomplete = valid_p_only.drop(columns="raw_ranker_score")
    invalid_rankings.append((incomplete, "schema is incomplete"))
    for invalid, message in invalid_rankings:
        with pytest.raises(RuntimeError, match=message):
            data._validate_ranking_score_semantics(
                invalid,
                seed=42,
                budget=20,
                method=data.METHOD_ORDER[1],
                expected_pool_size=2,
            )

    bce_missing = ranking_groups[(42, 20, data.METHOD_BCE)].copy()
    bce_missing.loc[bce_missing.index[0], "raw_ranker_score"] = float("nan")
    with pytest.raises(RuntimeError, match="unsupported raw-score missingness"):
        data._validate_ranking_score_semantics(
            bce_missing,
            seed=42,
            budget=20,
            method=data.METHOD_BCE,
            expected_pool_size=2,
        )
    invalid_positions = ranking_dump.copy()
    position_mask = (
        invalid_positions["target_budget"].eq(20)
        & invalid_positions["method_family"].eq(data.METHOD_BCE)
    )
    last_position = invalid_positions.index[position_mask][-1]
    invalid_positions.loc[last_position, "final_rank_position"] = 1

    class InvalidPositionStore:
        def ranking(self, seed: int) -> pd.DataFrame:
            return invalid_positions.loc[
                invalid_positions["seed"].eq(seed)
            ].copy()

    with pytest.raises(RuntimeError, match="Invalid full ranking positions"):
        data._load_ranking_groups(
            InvalidPositionStore(),  # type: ignore[arg-type]
            (42,),
            (20, 50, 100),
            2,
        )

    smoke_mismatch = dict(preflight_data_by_profile["smoke-synthetic"])
    smoke_mismatch["synthetic_generated_fraud_count"] = 99
    smoke_mismatch["synthetic_generated_legitimate_count"] = 4_901
    with pytest.raises(RuntimeError, match="counts or deduplication"):
        data._normalize_preflight_data(
            smoke_mismatch, contexts["smoke-synthetic"]
        )
    smoke_missing = dict(preflight_data_by_profile["smoke-synthetic"])
    del smoke_missing["synthetic_generated_fraud_count"]
    with pytest.raises(
        RuntimeError,
        match="missing required field synthetic_generated_fraud_count",
    ):
        data._normalize_preflight_data(
            smoke_missing, contexts["smoke-synthetic"]
        )
    smoke_bad_classes = dict(preflight_data_by_profile["smoke-synthetic"])
    smoke_bad_classes["synthetic_generated_fraud_count"] = 101
    with pytest.raises(RuntimeError, match="class-count arithmetic"):
        data._normalize_preflight_data(
            smoke_bad_classes, contexts["smoke-synthetic"]
        )
    smoke_real_fields = dict(preflight_data_by_profile["smoke-synthetic"])
    smoke_real_fields.update({"rows": 5_000, "class_0": 4_900, "class_1": 100})
    with pytest.raises(RuntimeError, match="contains real-data fields"):
        data._normalize_preflight_data(
            smoke_real_fields, contexts["smoke-synthetic"]
        )

    canonical_synthetic_fields = dict(preflight_data_by_profile["canonical"])
    canonical_synthetic_fields["data_source_kind"] = "synthetic"
    with pytest.raises(RuntimeError, match="contains synthetic fields"):
        data._normalize_preflight_data(
            canonical_synthetic_fields, contexts["canonical"]
        )
    canonical_wrong_counts = dict(preflight_data_by_profile["canonical"])
    canonical_wrong_counts.update({"rows": 12, "class_0": 10, "class_1": 2})
    with pytest.raises(RuntimeError, match="canonical dataset contract"):
        data._normalize_preflight_data(
            canonical_wrong_counts, contexts["canonical"]
        )

    invalid_dedup_root = tmp_path / "invalid-synthetic-deduplication"
    invalid_dedup_manifest = _write_completed_run(
        invalid_dedup_root, "smoke-synthetic"
    )
    invalid_summary = invalid_dedup_manifest["data_summary"]
    assert isinstance(invalid_summary, dict)
    invalid_summary["deduplicated_counts"] = {
        "rows": 4_999,
        "fraud": 100,
        "legitimate": 4_899,
    }
    invalid_summary["removed_duplicate_count"] = 1
    _rewrite_manifest(invalid_dedup_root, invalid_dedup_manifest)
    invalid_dedup_context = data.load_presentation_input_context(
        invalid_dedup_root
    )
    with pytest.raises(RuntimeError, match="counts or deduplication"):
        data._normalize_preflight_data(
            preflight_data_by_profile["smoke-synthetic"],
            invalid_dedup_context,
        )

    expected_canonical_paths = (
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
    assert CANONICAL_OUTPUT_PATHS == expected_canonical_paths
    assert len(CANONICAL_OUTPUT_PATHS) == data.EXPECTED_PRESENTATION_FRAMES == 29
    canonical_catalog = build_selection_registry()
    canonical_artifacts = canonical_catalog["artifacts"]
    assert isinstance(canonical_artifacts, list)
    assert tuple(row["artifact_id"] for row in canonical_artifacts) == (
        CANONICAL_ARTIFACT_IDS
    )
    assert len(CANONICAL_ARTIFACT_IDS) == 18
    selected_canonical = build_profile_selection_registry(
        presentation_role="canonical",
        profile="canonical",
        evidence_classification="thesis-evidentiary",
        data_source_kind="real",
    )
    assert selected_canonical == canonical_catalog
    with pytest.raises(RuntimeError, match="Canonical.*noncanonical"):
        build_profile_selection_registry(
            presentation_role="canonical",
            profile="mini-real",
            evidence_classification="engineering mini profile — not thesis evidence",
            data_source_kind="real",
        )
    with pytest.raises(RuntimeError, match="Engineering.*canonical"):
        build_profile_selection_registry(
            presentation_role="engineering",
            profile="canonical",
            evidence_classification="thesis-evidentiary",
            data_source_kind="real",
        )
    with pytest.raises(RuntimeError, match="Unsupported presentation role"):
        build_profile_selection_registry(
            presentation_role="unknown",
            profile="canonical",
            evidence_classification="thesis-evidentiary",
            data_source_kind="real",
        )

    canonical_metrics = _seed_metrics("canonical")
    canonical_tradeoff, _ = derive_tradeoff(
        canonical_metrics,
        contexts["canonical"].seeds,
        contexts["canonical"].target_budgets,
    )
    _, canonical_central_summary = derive_central_results(
        canonical_metrics,
        contexts["canonical"].seeds,
        contexts["canonical"].target_budgets,
    )
    assert list(canonical_tradeoff.columns) == [
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
    assert list(canonical_central_summary.columns) == [
        "target_budget",
        "method_family",
        "path_id",
        "path_label",
        "row_count",
        "prevented_loss_ratio_at_k_n",
        "prevented_loss_ratio_at_k_mean",
        "prevented_loss_ratio_at_k_sd",
        "frauds_at_k_n",
        "frauds_at_k_mean",
        "frauds_at_k_sd",
        "precision_at_k_n",
        "precision_at_k_mean",
        "precision_at_k_sd",
        "recall_at_k_n",
        "recall_at_k_mean",
        "recall_at_k_sd",
    ]

    curve_groups, valid_diagnostics = _global_curve_fixture()
    _curve_rows, _curve_summary, metric_rows, _metric_summary = (
        derive_global_and_pool_curves(
            curve_groups,
            valid_diagnostics,
            (42,),
            (50,),
        )
    )
    brier_rows = metric_rows.loc[metric_rows["metric"] == "brier"]
    assert len(brier_rows) == 1
    assert brier_rows.iloc[0]["method_family"] == "baseline_bce_probability"
    assert math.isclose(float(brier_rows.iloc[0]["value"]), 0.01)
    assert metric_rows.loc[
        (metric_rows["metric"] == "brier")
        & (metric_rows["method_family"] != "baseline_bce_probability")
    ].empty

    _missing_groups, missing_diagnostics = _global_curve_fixture(
        brier_values=()
    )
    with pytest.raises(RuntimeError, match="Missing registered BCE Brier"):
        derive_global_and_pool_curves(
            _missing_groups,
            missing_diagnostics,
            (42,),
            (50,),
        )
    _mismatching_groups, mismatching_diagnostics = _global_curve_fixture(
        brier_values=(0.02,)
    )
    with pytest.raises(RuntimeError, match="BCE Brier mismatch"):
        derive_global_and_pool_curves(
            _mismatching_groups,
            mismatching_diagnostics,
            (42,),
            (50,),
        )
    _duplicate_groups, duplicate_diagnostics = _global_curve_fixture(
        brier_values=(0.01, 0.01)
    )
    with pytest.raises(RuntimeError, match="Duplicate registered BCE Brier"):
        derive_global_and_pool_curves(
            _duplicate_groups,
            duplicate_diagnostics,
            (42,),
            (50,),
        )
    _nonfinite_groups, nonfinite_diagnostics = _global_curve_fixture(
        brier_values=(float("nan"),)
    )
    with pytest.raises(
        RuntimeError,
        match="Registered BCE Brier diagnostic is non-finite",
    ):
        derive_global_and_pool_curves(
            _nonfinite_groups,
            nonfinite_diagnostics,
            (42,),
            (50,),
        )

    engineering_outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for profile in ("mini-real", "smoke-synthetic"):
        context = contexts[profile]
        catalog = build_profile_selection_registry(
            presentation_role=context.presentation_role,
            profile=context.profile,
            evidence_classification=context.evidence_classification,
            data_source_kind=context.data_source_kind,
        )
        artifacts = catalog["artifacts"]
        assert isinstance(artifacts, list)
        assert tuple(row["artifact_id"] for row in artifacts) == (
            ENGINEERING_ARTIFACT_IDS
        )
        assert all(
            not str(row["artifact_id"]).startswith(("ch5_", "app_"))
            and not Path(str(row["input_data_file"])).name.startswith(
                ("ch5_", "app_")
            )
            for row in artifacts
        )
        assert catalog["presentation_role"] == "engineering"
        assert catalog["source_profile"] == profile
        outputs = derive_engineering(
            _seed_metrics(profile),
            context.seeds,
            context.target_budgets,
            profile=context.profile,
            evidence_classification=context.evidence_classification,
            data_source_kind=context.data_source_kind,
        )
        engineering_outputs[profile] = outputs
        assert tuple(outputs) == ENGINEERING_OUTPUT_PATHS
        heatmap, summary = outputs.values()
        assert len(heatmap) == len(context.seeds) * 3 * 3
        assert set(
            heatmap[["seed", "target_budget", "method_family"]].itertuples(
                index=False, name=None
            )
        ) == {
            (seed, budget, method)
            for seed in context.seeds
            for budget in (20, 50, 100)
            for method in (
                "selected_candidate_p_only",
                "selected_candidate_amount_gain",
                "candidate_postprocessing_p_times_log_amount",
            )
        }
        assert len(summary) == 3 * 4
        assert summary["seed_count"].eq(len(context.seeds)).all()
        assert set(summary["target_budget"].astype(int)) == {20, 50, 100}
        assert set(summary["method_family"]) == set(
            _seed_metrics(profile)["method_family"]
        )
        for frame in outputs.values():
            assert frame["profile"].eq(profile).all()
            assert frame["evidence_classification"].eq(
                context.evidence_classification
            ).all()
            assert frame["data_source_kind"].eq(
                context.data_source_kind
            ).all()
            assert frame["evidence_statement"].str.contains(
                "not thesis evidence", regex=False
            ).all()
            assert frame["evidence_statement"].str.contains(
                "not comparable with canonical empirical results", regex=False
            ).all()

    smoke_metrics = _seed_metrics("smoke-synthetic")
    smoke_summary = engineering_outputs["smoke-synthetic"][
        ENGINEERING_OUTPUT_PATHS[1]
    ]
    smoke_row = smoke_summary.iloc[0]
    smoke_observed = smoke_metrics.loc[
        (smoke_metrics["target_budget"] == smoke_row["target_budget"])
        & (smoke_metrics["method_family"] == smoke_row["method_family"]),
        "prevented_loss_ratio_at_k",
    ].iloc[0]
    assert smoke_row["plr_mean"] == smoke_observed
    assert smoke_summary.filter(like="sample_sd").isna().all().all()
    assert smoke_summary["evidence_statement"].str.contains(
        "synthetic engineering data", regex=False
    ).all()

    mini_metrics = _seed_metrics("mini-real")
    mini_summary = engineering_outputs["mini-real"][
        ENGINEERING_OUTPUT_PATHS[1]
    ]
    mini_row = mini_summary.iloc[0]
    mini_values = mini_metrics.loc[
        (mini_metrics["target_budget"] == mini_row["target_budget"])
        & (mini_metrics["method_family"] == mini_row["method_family"]),
        "prevented_loss_ratio_at_k",
    ]
    assert math.isclose(
        float(mini_row["plr_sample_sd"]),
        float(mini_values.std(ddof=1)),
    )

    complete_experiment = tmp_path / "outputs" / "complete"
    _write_completed_run(complete_experiment, "canonical")
    actual_context_loader = data.load_presentation_input_context
    loaded_contexts: list[data.PresentationInputContext] = []
    received_contexts: list[data.PresentationInputContext] = []

    def record_context(experiment_root: Path) -> data.PresentationInputContext:
        context = actual_context_loader(experiment_root)
        loaded_contexts.append(context)
        return context

    calls: list[tuple[str, object]] = []
    events: list[tuple[str, object]] = []

    def sink(kind: str, payload: object) -> None:
        events.append((kind, payload))

    def fake_data(
        repository_root: Path,
        experiment_root: Path,
        output_root: Path,
        *,
        input_context: data.PresentationInputContext,
        presentation_role: data.PresentationRole,
        force: bool,
        event_sink: object,
    ) -> dict[str, object]:
        assert repository_root == tmp_path.resolve()
        assert experiment_root == input_context.experiment_root
        expected_name = (
            "presentation"
            if input_context.presentation_role == "canonical"
            else "engineering-presentation"
        )
        assert output_root == tmp_path / "generated" / expected_name
        assert force
        assert presentation_role == input_context.presentation_role
        received_contexts.append(input_context)
        calls.append(("data", event_sink))
        if event_sink is not None:
            event_sink(
                "status",
                {"level": "INFO", "event": "data", "fields": {}},
            )
        return {"status": "PASS", "stage": "data"}

    def fake_figures(
        repository_root: Path,
        data_dir: Path,
        figures_dir: Path,
        *,
        width_mm: float,
        force: bool,
        event_sink: object,
    ) -> dict[str, object]:
        assert repository_root == tmp_path.resolve()
        assert figures_dir == data_dir.parent / "figures"
        assert width_mm == 160.0
        assert force
        calls.append(("figures", event_sink))
        if event_sink is not None:
            event_sink(
                "status",
                {"level": "INFO", "event": "figures", "fields": {}},
            )
        return {"status": "PASS", "stage": "figures"}

    def fake_tables(
        repository_root: Path,
        data_dir: Path,
        tables_dir: Path,
        *,
        preview_dir: Path,
        width_mm: float,
        force: bool,
        event_sink: object,
    ) -> dict[str, object]:
        assert repository_root == tmp_path.resolve()
        assert tables_dir == data_dir.parent / "tables"
        assert preview_dir == data_dir.parent / "preview"
        assert width_mm == 160.0
        assert force
        calls.append(("tables", event_sink))
        if event_sink is not None:
            event_sink(
                "status",
                {"level": "INFO", "event": "tables", "fields": {}},
            )
        return {"status": "PASS", "stage": "tables"}

    monkeypatch.setattr(data, "load_presentation_input_context", record_context)
    monkeypatch.setattr(data, "build", fake_data)
    monkeypatch.setattr(figures, "render", fake_figures)
    monkeypatch.setattr(tables, "render", fake_tables)
    config = PresentationConfig(
        repository_root=tmp_path,
        experiment_root=Path("outputs/complete"),
        output_root=Path("generated/presentation"),
        force=True,
        event_sink=sink,
    )

    result = build_presentation(config)

    assert isinstance(result, PresentationResult)
    assert result.status == "COMPLETE"
    assert result.output_root == tmp_path / "generated" / "presentation"
    assert result.data_dir == result.output_root / "data"
    assert result.figures_dir == result.output_root / "figures"
    assert result.tables_dir == result.output_root / "tables"
    assert result.preview_dir == result.output_root / "preview"
    assert [step.step for step in result.steps] == ["data", "figures", "tables"]
    assert [step.manifest for step in result.steps] == [
        {"status": "PASS", "stage": "data"},
        {"status": "PASS", "stage": "figures"},
        {"status": "PASS", "stage": "tables"},
    ]
    assert [step.manifest_path for step in result.steps] == [
        result.data_dir / "PRESENTATION_DATA_MANIFEST.json",
        result.figures_dir / "FIGURE_RENDER_MANIFEST.json",
        result.tables_dir / "TABLE_RENDER_MANIFEST.json",
    ]
    assert all(step.elapsed_seconds >= 0.0 for step in result.steps)
    assert [name for name, _ in calls] == ["data", "figures", "tables"]
    assert all(boundary is sink for _, boundary in calls)
    assert received_contexts == loaded_contexts
    assert received_contexts[0] is loaded_contexts[0]
    assert [payload[0] for payload in events] == ["status"] * 3

    calls.clear()
    default_result = build_presentation(
        PresentationConfig(
            repository_root=tmp_path,
            experiment_root=Path("outputs/complete"),
            output_root=Path("generated/presentation"),
            force=True,
        )
    )
    captured = capsys.readouterr()
    assert default_result.status == "COMPLETE"
    assert [name for name, _ in calls] == ["data", "figures", "tables"]
    assert all(boundary is None for _, boundary in calls)
    assert received_contexts[-1] is loaded_contexts[-1]
    assert len(loaded_contexts) == 2
    assert captured.out == ""
    assert captured.err == ""

    engineering_experiment = tmp_path / "outputs" / "engineering"
    _write_completed_run(engineering_experiment, "smoke-synthetic")
    calls.clear()
    engineering_result = build_presentation(
        PresentationConfig(
            repository_root=tmp_path,
            experiment_root=Path("outputs/engineering"),
            output_root=Path("generated/engineering-presentation"),
            force=True,
        )
    )
    assert engineering_result.status == "COMPLETE"
    assert [name for name, _ in calls] == ["data", "figures", "tables"]
    assert loaded_contexts[-1].presentation_role == "engineering"
    assert received_contexts[-1] is loaded_contexts[-1]


def test_build_presentation_stops_and_preserves_failure_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fraud_detection.presentation.preparation import data
    from fraud_detection.presentation.rendering import figures, tables

    config = PresentationConfig(
        repository_root=tmp_path,
        experiment_root=Path("outputs/complete"),
        output_root=Path("generated/presentation"),
    )
    calls: list[str] = []

    def unexpected_stage(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("a later stage must not execute")

    monkeypatch.setattr(figures, "render", unexpected_stage)
    monkeypatch.setattr(tables, "render", unexpected_stage)
    with pytest.raises(PresentationError) as data_failure:
        build_presentation(config)

    assert calls == []
    assert data_failure.value.failed_step == "data"
    assert data_failure.value.completed_steps == ()
    assert data_failure.value.original_exception_type == "FileNotFoundError"
    assert "Missing RUN_MANIFEST.json" in data_failure.value.original_message
    assert isinstance(data_failure.value.__cause__, FileNotFoundError)
    assert not (tmp_path / "generated" / "presentation").exists()
    protected_output = tmp_path / "generated" / "protected"
    protected_output.mkdir(parents=True)
    sentinel = protected_output / "keep.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    with pytest.raises(PresentationError):
        build_presentation(
            PresentationConfig(
                repository_root=tmp_path,
                experiment_root=Path("outputs/complete"),
                output_root=Path("generated/protected"),
                force=True,
            )
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"

    def set_nested(
        manifest: dict[str, Any],
        key: str,
        nested_key: str,
        value: object,
    ) -> None:
        nested = manifest[key]
        assert isinstance(nested, dict)
        nested[nested_key] = value

    def add_unsafe(manifest: dict[str, Any], path: str) -> None:
        entries = manifest["produced_artifacts"]
        assert isinstance(entries, list)
        entries.append(
            {"path": path, "group": "aggregation", "format": "csv"}
        )

    invalid_cases: tuple[
        tuple[str, Callable[[dict[str, Any]], None], str], ...
    ] = (
        (
            "schema",
            lambda manifest: manifest.__setitem__("schema", "invalid"),
            "schema is invalid",
        ),
        (
            "failed",
            lambda manifest: manifest.__setitem__("status", "FAILED"),
            "status is not COMPLETE",
        ),
        (
            "incomplete",
            lambda manifest: manifest.__setitem__(
                "completed_phases", ["preflight"]
            ),
            "completed phases are incomplete",
        ),
        (
            "unknown-profile",
            lambda manifest: manifest.__setitem__("profile", "unknown"),
            "profile is unknown",
        ),
        (
            "evidence",
            lambda manifest: manifest.__setitem__(
                "evidence_classification", "wrong"
            ),
            "evidence classification does not match profile",
        ),
        (
            "effective",
            lambda manifest: set_nested(
                manifest, "effective_config", "candidate_pool_size", 17
            ),
            "effective configuration does not match profile",
        ),
        (
            "unsafe-parent",
            lambda manifest: add_unsafe(manifest, "../outside.csv"),
            "unsafe artifact path",
        ),
        (
            "unsafe-absolute",
            lambda manifest: add_unsafe(manifest, "C:/outside.csv"),
            "unsafe artifact path",
        ),
        (
            "non-finite",
            lambda manifest: set_nested(
                manifest, "data_summary", "removed_duplicate_count", float("nan")
            ),
            "contains non-finite numbers",
        ),
        (
            "arithmetic",
            lambda manifest: set_nested(
                manifest,
                "data_summary",
                "deduplicated_counts",
                {"rows": 11, "fraud": 2, "legitimate": 10},
            ),
            "data-count arithmetic is invalid",
        ),
    )
    for name, mutate, message in invalid_cases:
        experiment = tmp_path / "invalid" / name
        manifest = _write_completed_run(experiment, "canonical")
        mutate(manifest)
        _rewrite_manifest(experiment, manifest)
        with pytest.raises(RuntimeError, match=message):
            data.load_presentation_input_context(experiment)

    duplicate_root = tmp_path / "invalid" / "duplicate"
    duplicate_manifest = _write_completed_run(duplicate_root, "canonical")
    duplicate_entries = duplicate_manifest["produced_artifacts"]
    assert isinstance(duplicate_entries, list)
    duplicate_entries.append(dict(duplicate_entries[-1]))
    _rewrite_manifest(duplicate_root, duplicate_manifest)
    with pytest.raises(RuntimeError, match="duplicate artifact paths"):
        data.load_presentation_input_context(duplicate_root)

    missing_file_root = tmp_path / "invalid" / "missing-file"
    _write_completed_run(missing_file_root, "canonical")
    (missing_file_root / "preflight" / "preflight_validation.json").unlink()
    with pytest.raises(FileNotFoundError, match="Registered experiment file"):
        data.load_presentation_input_context(missing_file_root)

    missing_registration_root = tmp_path / "invalid" / "missing-registration"
    missing_registration = _write_completed_run(
        missing_registration_root, "canonical"
    )
    missing_entries = missing_registration["produced_artifacts"]
    assert isinstance(missing_entries, list)
    missing_registration["produced_artifacts"] = [
        entry
        for entry in missing_entries
        if entry["path"] != "preflight/preflight_validation.json"
    ]
    (
        missing_registration_root
        / "preflight"
        / "preflight_validation.json"
    ).unlink()
    _rewrite_manifest(missing_registration_root, missing_registration)
    with pytest.raises(RuntimeError, match="lacks required presentation input"):
        data.load_presentation_input_context(missing_registration_root)

    invalid_json_root = tmp_path / "invalid" / "json"
    _write_completed_run(invalid_json_root, "canonical")
    (invalid_json_root / "RUN_MANIFEST.json").write_text(
        "{invalid\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="not valid UTF-8 JSON") as invalid_json:
        data.load_presentation_input_context(invalid_json_root)
    assert not isinstance(invalid_json.value, json.JSONDecodeError)

    _write_completed_run(tmp_path / "outputs" / "complete", "canonical")
    calls.clear()

    def pass_data(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("data")
        return {"status": "PASS"}

    def fail_figures(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("figures")
        raise RuntimeError("controlled figure failure")

    monkeypatch.setattr(data, "build", pass_data)
    monkeypatch.setattr(figures, "render", fail_figures)
    with pytest.raises(PresentationError) as figure_failure:
        build_presentation(config)

    assert calls == ["data", "figures"]
    assert figure_failure.value.failed_step == "figures"
    assert [step.step for step in figure_failure.value.completed_steps] == ["data"]
    assert figure_failure.value.original_exception_type == "RuntimeError"
    assert figure_failure.value.original_message == "controlled figure failure"
    assert isinstance(figure_failure.value.__cause__, RuntimeError)

    engineering_root = tmp_path / "outputs" / "engineering"
    _write_completed_run(engineering_root, "smoke-synthetic")
    calls.clear()
    with pytest.raises(PresentationError) as engineering_failure:
        build_presentation(
            PresentationConfig(
                repository_root=tmp_path,
                experiment_root=Path("outputs/engineering"),
                output_root=Path("generated/engineering-presentation"),
            )
        )
    assert calls == ["data", "figures"]
    assert engineering_failure.value.failed_step == "figures"
    assert engineering_failure.value.original_message == (
        "controlled figure failure"
    )
    assert [step.step for step in engineering_failure.value.completed_steps] == [
        "data"
    ]


def test_build_presentation_validates_before_work_and_preserves_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fraud_detection.presentation.preparation import data
    from fraud_detection.presentation.rendering import figures

    _accept_profile_data_fixtures(tmp_path, monkeypatch)
    experiment = tmp_path / "registered-read-contract"
    _write_completed_run(experiment, "canonical")
    context = data.load_presentation_input_context(experiment)
    store = data.ExperimentStore(context)
    unregistered = experiment / "unregistered.csv"
    unregistered.write_text("not registered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unregistered experiment input"):
        store.path("unregistered.csv")
    registered = experiment / "figure_data" / "all_budget_matched_results.csv"
    registered.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        store.path("figure_data/all_budget_matched_results.csv")

    calls: list[str] = []

    def unexpected_data(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("data")
        return {}

    monkeypatch.setattr(data, "build", unexpected_data)
    with pytest.raises(ValueError, match="between 80 and 300"):
        build_presentation(
            PresentationConfig(
                repository_root=tmp_path,
                experiment_root=Path("outputs/complete"),
                output_root=Path("generated/presentation"),
                width_mm=float("nan"),
            )
        )
    assert calls == []
    with pytest.raises(TypeError, match="PresentationConfig"):
        build_presentation(object())  # type: ignore[arg-type]

    def interrupt_data(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append("data")
        raise KeyboardInterrupt

    monkeypatch.setattr(data, "build", interrupt_data)
    monkeypatch.setattr(
        figures,
        "render",
        lambda *_args, **_kwargs: pytest.fail("figures must not execute"),
    )
    _write_completed_run(tmp_path / "outputs" / "complete", "canonical")
    with pytest.raises(KeyboardInterrupt):
        build_presentation(
            PresentationConfig(
                repository_root=tmp_path,
                experiment_root=Path("outputs/complete"),
                output_root=Path("generated/presentation"),
            )
        )
    assert calls == ["data"]
