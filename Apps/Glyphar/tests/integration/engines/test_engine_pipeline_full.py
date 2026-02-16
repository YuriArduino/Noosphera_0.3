from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
from pdf2image import convert_from_path

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


def _to_bgr_uint8(pil_img) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.uint8)
    return arr.astype(np.uint8)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not available")
def test_managed_engine_full_pipeline_smoke() -> None:
    data_dir = _resolve_data_dir()
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


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not available")
def test_managed_engine_with_batch_processor() -> None:
    data_dir = _resolve_data_dir()
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

    results = batch.recognize_batch(images, {"psm": 3, "min_confidence": 10.0})
    assert len(results) == len(images)

    for out in results:
        assert "text" in out
        assert "confidence" in out
        assert "words" in out
