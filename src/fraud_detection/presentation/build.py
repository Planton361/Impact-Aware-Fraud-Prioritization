"""Direct, rendering-independent orchestration for presentation stages."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Literal

PresentationStep = Literal["data", "figures", "tables"]
PresentationEventSink = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class PresentationConfig:
    """Paths, rendering width, and event boundary for one presentation build."""

    repository_root: Path
    experiment_root: Path
    output_root: Path
    width_mm: float = 160.0
    preview_dir: Path | None = None
    force: bool = False
    event_sink: PresentationEventSink | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class PresentationStepResult:
    """One successfully completed presentation stage."""

    step: PresentationStep
    manifest_path: Path
    manifest: dict[str, object]
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PresentationResult:
    """Structured result returned after all presentation stages succeed."""

    output_root: Path
    data_dir: Path
    figures_dir: Path
    tables_dir: Path
    preview_dir: Path
    steps: tuple[PresentationStepResult, ...]
    status: Literal["COMPLETE"]


class PresentationError(RuntimeError):
    """A presentation-stage failure with completed-stage context."""

    def __init__(
        self,
        failed_step: PresentationStep,
        completed_steps: tuple[PresentationStepResult, ...],
        original_exception_type: str,
        original_message: str,
    ) -> None:
        self.failed_step = failed_step
        self.completed_steps = completed_steps
        self.original_exception_type = original_exception_type
        self.original_message = original_message
        super().__init__(
            f"Presentation step {failed_step} failed: "
            f"{original_exception_type}: {original_message}"
        )


def _resolved_from(root: Path, path: Path) -> Path:
    selected = path if path.is_absolute() else root / path
    return selected.resolve()


def build_presentation(config: PresentationConfig) -> PresentationResult:
    """Build data, figures, and tables serially from completed artifacts."""

    if not isinstance(config, PresentationConfig):
        raise TypeError("config must be a PresentationConfig instance.")

    repository_root = config.repository_root.resolve()
    experiment_root = _resolved_from(repository_root, config.experiment_root)
    output_root = _resolved_from(repository_root, config.output_root)
    data_dir = output_root / "data"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    preview_dir = (
        _resolved_from(repository_root, config.preview_dir)
        if config.preview_dir is not None
        else output_root / "preview"
    )
    width_mm = float(config.width_mm)
    if not math.isfinite(width_mm) or not 80.0 <= width_mm <= 300.0:
        raise ValueError("width_mm must be finite and between 80 and 300.")

    data_stage = import_module(
        "fraud_detection.presentation.preparation.data"
    )
    try:
        input_context = data_stage.load_presentation_input_context(
            experiment_root
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise PresentationError(
            failed_step="data",
            completed_steps=(),
            original_exception_type=type(exc).__name__,
            original_message=str(exc),
        ) from exc
    figures_stage = import_module(
        "fraud_detection.presentation.rendering.figures"
    )
    tables_stage = import_module(
        "fraud_detection.presentation.rendering.tables"
    )
    stages: tuple[
        tuple[PresentationStep, Path, Callable[[], dict[str, object]]],
        ...,
    ] = (
        (
            "data",
            data_dir / "PRESENTATION_DATA_MANIFEST.json",
            lambda: data_stage.build(
                repository_root,
                experiment_root,
                output_root,
                input_context=input_context,
                presentation_role=input_context.presentation_role,
                force=config.force,
                event_sink=config.event_sink,
            ),
        ),
        (
            "figures",
            figures_dir / "FIGURE_RENDER_MANIFEST.json",
            lambda: figures_stage.render(
                repository_root,
                data_dir,
                figures_dir,
                width_mm=width_mm,
                force=config.force,
                event_sink=config.event_sink,
            ),
        ),
        (
            "tables",
            tables_dir / "TABLE_RENDER_MANIFEST.json",
            lambda: tables_stage.render(
                repository_root,
                data_dir,
                tables_dir,
                preview_dir=preview_dir,
                width_mm=width_mm,
                force=config.force,
                event_sink=config.event_sink,
            ),
        ),
    )

    completed: list[PresentationStepResult] = []
    for step, manifest_path, invoke in stages:
        started = time.perf_counter()
        try:
            manifest = invoke()
            if not isinstance(manifest, dict):
                raise TypeError(
                    f"Presentation step {step} returned a non-dict manifest."
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise PresentationError(
                failed_step=step,
                completed_steps=tuple(completed),
                original_exception_type=type(exc).__name__,
                original_message=str(exc),
            ) from exc
        completed.append(
            PresentationStepResult(
                step=step,
                manifest_path=manifest_path,
                manifest=dict(manifest),
                elapsed_seconds=time.perf_counter() - started,
            )
        )

    return PresentationResult(
        output_root=output_root,
        data_dir=data_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        preview_dir=preview_dir,
        steps=tuple(completed),
        status="COMPLETE",
    )
