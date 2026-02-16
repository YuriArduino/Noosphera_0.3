# pylint: disable=wrong-import-position
"""Integration coverage for the managed Tesseract engine full pipeline.

The tests validate smoke and batch scenarios over all pages of control PDFs,
persist OCR artifacts for manual inspection, and print runtime/configuration
metrics to simplify debugging.
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


def _list_test_pdfs(data_dir: Path) -> list[Path]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise AssertionError("No PDF files found in Test/Data")
    return pdfs


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
    pdfs = _list_test_pdfs(data_dir)

    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages="por+eng",
        model_type="fast",
    )

    config = {"psm": 3, "quality_hint": "good", "min_confidence": 20.0}

    per_file_results = []

    total_pages = 0
    for pdf_path in pdfs:
        pages = convert_from_path(str(pdf_path), dpi=200)
        assert pages, f"Failed to render pages for {pdf_path.name}"

        for page_index, page in enumerate(pages, start=1):
            total_pages += 1
            image = _to_bgr_uint8(page)

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

            per_file_results.append(
                {
                    "pdf_file": pdf_path.name,
                    "page": page_index,
                    "first_pass": result_1,
                    "second_pass": result_2,
                }
            )

            stem = pdf_path.stem
            _write_json(
                output_dir
                / f"managed_smoke_{stem}_page_{page_index:03d}_result_1.json",
                result_1,
            )
            _write_json(
                output_dir
                / f"managed_smoke_{stem}_page_{page_index:03d}_result_2.json",
                result_2,
            )

    assert engine.stats.cache_hits >= total_pages
    assert len(engine.cache) >= total_pages

    stats_summary = engine.stats.get_summary()
    metrics = {
        "configured": {
            "tessdata_dir": str(_resolve_default_tessdata()),
            "languages": "por+eng",
            "model_type": "fast",
            "config": config,
            "total_pdf_files": len(pdfs),
            "total_pages": total_pages,
            "pdf_files": [pdf.name for pdf in pdfs],
        },
        "runtime": {
            "cache_hits": engine.stats.cache_hits,
            "cache_misses": engine.stats.cache_misses,
            "cache_entries": len(engine.cache),
            "avg_confidence": stats_summary["avg_confidence"],
            "avg_time_ms": stats_summary["avg_time_per_page_ms"],
            "per_file_word_counts": [
                {
                    "pdf_file": entry["pdf_file"],
                    "page": entry["page"],
                    "first_pass": entry["first_pass"].get("word_count", 0),
                    "second_pass": entry["second_pass"].get("word_count", 0),
                }
                for entry in per_file_results
            ],
        },
    }

    _write_json(output_dir / "managed_smoke_summary.json", {"files": per_file_results})
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
    pdfs = _list_test_pdfs(data_dir)

    images = []
    pages_per_file: dict[str, int] = {}
    for pdf_path in pdfs:
        pages = convert_from_path(str(pdf_path), dpi=150)
        assert pages, f"Failed to render PDF pages for {pdf_path.name}"
        pages_per_file[pdf_path.name] = len(pages)
        images.extend(_to_bgr_uint8(page) for page in pages)

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
            "total_pdf_files": len(pdfs),
            "pages_per_file": pages_per_file,
        },
        "results": results,
    }
    _write_json(output_dir / "managed_batch_results.json", payload)

    print(
        f"[test_engine_pipeline_full] batch_output={output_dir / 'managed_batch_results.json'}"
    )
