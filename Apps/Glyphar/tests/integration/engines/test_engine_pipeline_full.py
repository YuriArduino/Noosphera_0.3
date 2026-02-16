# pylint: disable=wrong-import-position
"""Integration coverage for the managed Tesseract engine full pipeline.

The tests validate smoke and batch scenarios, persist OCR artifacts for manual
inspection, and print runtime/configuration metrics to simplify debugging.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pytest

np = pytest.importorskip("numpy")
pdf2image = pytest.importorskip("pdf2image")
convert_from_path = pdf2image.convert_from_path

from glyphar.engines.batch import TesseractBatchProcessor
from glyphar.engines.managed.tesseract_managed import TesseractManagedEngine
from glyphar.engines.validation import _resolve_default_tessdata


pytestmark = pytest.mark.integration


def _resolve_data_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "Test" / "Data"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate Test/Data directory from test path.")


def _resolve_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = (
        project_root / "tests" / "output_data" / "engines" / "tesseract_core_full"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _to_bgr_uint8(pil_img: Any) -> Any:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    return arr.astype(np.uint8)


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not available"
)
def test_managed_engine_full_pipeline_smoke() -> None:
    """Run a full managed-engine smoke flow and persist OCR outputs + metrics."""
    data_dir = _resolve_data_dir()
    output_dir = _resolve_output_dir()
    pdfs = sorted(data_dir.glob("*.pdf"))
    assert pdfs, "No PDF files found in Test/Data"

    pages = convert_from_path(str(pdfs[0]), dpi=200, first_page=1, last_page=1)
    assert pages, "Failed to render first PDF page"

    image = _to_bgr_uint8(pages[0])

    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages="por+eng",
        model_type="fast",
    )

    config = {"psm": 3, "quality_hint": "good", "min_confidence": 20.0}

    result_1 = engine.recognize(image, config)
    result_2 = engine.recognize(image, config)

    for result in (result_1, result_2):
        assert isinstance(result.get("text", ""), str)
        assert isinstance(result.get("words", []), list)
        assert isinstance(result.get("confidence", 0.0), float)
        assert isinstance(result.get("word_count", 0), int)
        assert isinstance(result.get("char_count", 0), int)
        assert result["char_count"] == len(result["text"])
        assert "processing_time_ms" in result
        assert "config_used" in result

    assert engine.stats.cache_hits >= 1
    assert len(engine.cache) >= 1

    stats_summary = engine.stats.get_summary()
    metrics = {
        "configured": {
            "tessdata_dir": str(_resolve_default_tessdata()),
            "languages": "por+eng",
            "model_type": "fast",
            "config": config,
        },
        "runtime": {
            "cache_hits": engine.stats.cache_hits,
            "cache_misses": engine.stats.cache_misses,
            "cache_entries": len(engine.cache),
            "avg_confidence": stats_summary["avg_confidence"],
            "avg_time_ms": stats_summary["avg_time_per_page_ms"],
            "result_1_word_count": result_1.get("word_count", 0),
            "result_2_word_count": result_2.get("word_count", 0),
        },
    }

    _write_json(output_dir / "managed_smoke_result_1.json", result_1)
    _write_json(output_dir / "managed_smoke_result_2.json", result_2)
    _write_json(output_dir / "managed_smoke_metrics.json", metrics)

    print(f"[test_engine_pipeline_full] output_dir={output_dir}")
    print(
        f"[test_engine_pipeline_full] metrics={json.dumps(metrics, ensure_ascii=False)}"
    )


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract binary not available"
)
def test_managed_engine_with_batch_processor() -> None:
    """Run managed engine through batch processor and persist output payload."""
    data_dir = _resolve_data_dir()
    output_dir = _resolve_output_dir()
    pdfs = sorted(data_dir.glob("*.pdf"))
    assert pdfs, "No PDF files found in Test/Data"

    pages = convert_from_path(str(pdfs[0]), dpi=150, first_page=1, last_page=2)
    assert pages, "Failed to render PDF pages"

    images = [_to_bgr_uint8(page) for page in pages]

    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages="por+eng",
        model_type="fast",
    )
    batch = TesseractBatchProcessor(engine)

    batch_config = {"psm": 3, "min_confidence": 10.0}
    results = batch.recognize_batch(images, batch_config)
    assert len(results) == len(images)

    for out in results:
        assert "text" in out
        assert "confidence" in out
        assert "words" in out

    payload = {
        "configured": {
            "batch_config": batch_config,
            "total_images": len(images),
        },
        "results": results,
    }
    _write_json(output_dir / "managed_batch_results.json", payload)

    print(
        f"[test_engine_pipeline_full] batch_output={output_dir / 'managed_batch_results.json'}"
    )
