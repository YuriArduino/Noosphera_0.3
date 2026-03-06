#!/usr/bin/env python3
"""Ephemeral Glyphar runner for Docker/Compose usage by LLM agents."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from glyphar.core.pipeline import OCRPipeline
from glyphar.engines.managed.tesseract_managed import TesseractManagedEngine
from glyphar.engines.validation import _resolve_default_tessdata
from glyphar.layout.column_detector import ColumnLayoutDetector
from glyphar.models.config import OCRConfig


ROOT = Path("/workspace")
DEFAULT_RUNTIME = ROOT / "tools" / "Glyphar" / "config" / "runtime.yaml"
DEFAULT_ENV = ROOT / "tools" / "Glyphar" / "config" / "environment.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build_config(runtime: dict[str, Any]) -> OCRConfig:
    """Construct OCRConfig from merged runtime and environment configurations."""
    pipeline_cfg = runtime.get("pipeline", {})
    analysis_cfg = runtime.get("analysis", {})
    quality_cfg = analysis_cfg.get("quality_assessment", {})
    limits_cfg = runtime.get("limits", {})

    return OCRConfig(
        languages=runtime.get("engine", {}).get("language", "pt"),
        min_confidence=float(analysis_cfg.get("confidence_threshold", 85.0)),
        parallel=bool(pipeline_cfg.get("enable_parallelism", True)),
        max_workers=int(pipeline_cfg.get("max_workers", 4)),
        timeout_per_page_s=int(limits_cfg.get("timeout_seconds", 300)),
        enable_quality_assessment=bool(quality_cfg.get("enabled", False)),
    )


def main() -> None:
    """Main entry point for the Glyphar runner."""
    input_path = Path(os.environ.get("GLYPHAR_INPUT", "/data/input/document.pdf"))
    output_dir = Path(os.environ.get("GLYPHAR_OUTPUT_DIR", "/data/output"))

    runtime_path = Path(os.environ.get("GLYPHAR_RUNTIME_CONFIG", str(DEFAULT_RUNTIME)))
    env_path = Path(os.environ.get("GLYPHAR_ENV_CONFIG", str(DEFAULT_ENV)))

    if not input_path.exists():
        raise FileNotFoundError(f"Input document not found: {input_path}")

    runtime = _load_yaml(runtime_path)
    env_cfg = _load_yaml(env_path)
    merged_runtime = _merge(runtime, env_cfg.get("overrides", {}))

    config = _build_config(merged_runtime)
    model_type = os.environ.get(
        "GLYPHAR_MODEL_TYPE",
        merged_runtime.get("engine", {}).get("model_type", "standard"),
    )

    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages="por+eng",
        model_type=model_type,
        config=config,
    )

    pipeline = OCRPipeline(
        engine=engine,
        layout_detector=ColumnLayoutDetector(),
        _preprocessing_strategies=[],
        config=config,
        include_llm_input=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.process(
        file_path=str(input_path),
        parallel=config.parallel,
        max_workers=config.max_workers,
        batch_size=int(merged_runtime.get("pipeline", {}).get("batch_size", 8)),
        show_progress=False,
    )

    stem = input_path.stem
    (output_dir / f"{stem}.txt").write_text(result.full_text, encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{stem}.summary.json").write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "success",
                "input": str(input_path),
                "output_dir": str(output_dir),
                "pages": result.total_pages,
                "avg_confidence": result.average_confidence,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
