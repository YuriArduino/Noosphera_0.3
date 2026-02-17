"""Tessdata validation
        ↓
Config builder
        ↓
Core recognize()
        ↓
Processor normalization
        ↓
Resultado estruturado consistente
stats
"""

# tests/integration/engines/test_engine_pipeline_basic.py
from pathlib import Path
import time
import pytest
from collections import defaultdict
import numpy as np
from pdf2image import convert_from_path

from glyphar.engines.validation import validate_tessdata
from glyphar.engines.validation import _resolve_default_tessdata
from glyphar.engines.config_builder import TesseractConfigBuilder
from glyphar.engines.core.tesseract_core import TesseractCoreEngine
from glyphar.engines.batch import TesseractBatchProcessor
from glyphar.engines.stats import OCRStats

# Adjust as needed for your environment
DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/engines/pipeline_basic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _enrich_core_result(core_result: dict) -> dict:
    """
    Lightweight adapter to produce simple post-processed metrics from
    the core result. We do this here (in the test) because the project's
    processor implementation expects raw pytesseract output (not the
    already-parsed core output). This adapter provides minimal metrics
    for assertions and human inspection.
    """
    text = core_result.get("text", "") or ""
    words = core_result.get("words", []) or []

    word_count = len(words)
    char_count = len(text)

    # confidence already aggregated by core (0.0 - 100.0)
    confidence = float(core_result.get("confidence", 0.0))

    if confidence >= 80.0:
        confidence_bucket = "high"
    elif confidence >= 40.0:
        confidence_bucket = "medium"
    else:
        confidence_bucket = "low"

    enriched = {
        "text": text,
        "words": words,
        "word_count": word_count,
        "char_count": char_count,
        "confidence": confidence,
        "confidence_bucket": confidence_bucket,
    }
    return enriched


def group_words_by_line(words, tolerance=10):
    lines = defaultdict(list)

    for w in words:
        bbox = w.get("bbox", {})
        top = bbox.get("top")
        if top is None:
            continue

        line_key = round(top / tolerance)
        lines[line_key].append(w)

    ordered_lines = []
    for key in sorted(lines.keys()):
        sorted_words = sorted(lines[key], key=lambda x: x["bbox"].get("left", 0) or 0)
        ordered_lines.append(" ".join(w["text"] for w in sorted_words))

    return "\n".join(ordered_lines)


@pytest.mark.integration
def test_engine_pipeline_basic():
    """
    Integration micro-pipeline test:

    validation -> config_builder -> core (via batch) -> local enrichment

    Notes:
    - TesseractConfigBuilder.build() returns a CLI string. Core currently
      expects a dict (psm/oem). We assert the builder output but pass a dict
      to the core to keep compatibility with current core API.
    - The project's `process_ocr_data` expects raw pytesseract output; since
      core already returns parsed results we adapt in-test for lightweight
      post-processing.
    """

    # 1) Validation

    languages = validate_tessdata(None, "por+eng")
    tessdata_dir = _resolve_default_tessdata()
    assert isinstance(languages, str) and languages != ""

    # 2) Config builder (we assert the string, but core uses a dict)
    builder = TesseractConfigBuilder(tessdata_dir, "standard")
    cfg_str = builder.build(psm=3, oem=1)
    # ensure builder produced expected flags (basic sanity)
    assert "--psm 3" in cfg_str
    assert "--oem 1" in cfg_str

    # 3) Core engine (pure) + 4) Batch wrapper
    engine = TesseractCoreEngine(languages)
    batcher = TesseractBatchProcessor(engine)
    stats = OCRStats()

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "No PDF files found in DATA_DIR."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)
        assert pages, f"No pages extracted from {pdf_path.name}"

        # We'll process pages in small batches using the batcher (here batch size = 1)
        for page_idx, pil_img in enumerate(pages):
            img = np.array(pil_img).astype(np.uint8)

            start_ms = time.perf_counter() * 1000.0

            # We pass a dict config to the core (current core API)
            config_dict = {"psm": 3, "oem": 1, "timeout": 60}

            # batcher expects a list of images and returns list of results
            results = batcher.recognize_batch([img], config_dict)
            assert isinstance(results, list) and len(results) == 1

            core_result = results[0]

            # Basic structural assertions (core contract)
            assert "text" in core_result
            assert "confidence" in core_result
            assert "words" in core_result

            assert isinstance(core_result["text"], str)
            assert isinstance(core_result["confidence"], float)
            assert isinstance(core_result["words"], list)

            # Local enrichment (lightweight "processor" in test)
            processed = _enrich_core_result(core_result)

            # Additional assertions about enriched metrics
            assert processed["word_count"] >= 0
            assert processed["char_count"] >= 0
            assert 0.0 <= processed["confidence"] <= 100.0
            assert processed["confidence_bucket"] in ("low", "medium", "high")

            duration_ms = (time.perf_counter() * 1000.0) - start_ms
            stats.update(processed["confidence"], duration_ms)

            # Save textual output for manual inspection
            output_file = OUTPUT_DIR / f"{pdf_path.stem}_p{page_idx+1}.txt"
            formatted_text = group_words_by_line(processed["words"])
            output_file.write_text(formatted_text, encoding="utf-8")

            # Print a short log line (visible in pytest -s)
            print(
                f"[pipeline] {pdf_path.name} p{page_idx+1}: "
                f"chars={processed['char_count']} words={processed['word_count']} "
                f"conf={processed['confidence']:.1f}% time_ms={duration_ms:.0f}"
            )

            # Sanity acceptance rule (example):
            # If confidence < 30% mark as suspicious (but don't fail test).
            if processed["confidence"] < 30.0:
                print(
                    f"⚠ suspicious low-confidence page: {pdf_path.name} p{page_idx+1}"
                )
            summary = stats.get_summary()

            print("\n[stats summary]")
            for k, v in summary.items():
                print(f"{k}: {v}")

            # Basic structural assertions
            assert summary["total_pages"] > 0
            assert 0.0 <= summary["avg_confidence"] <= 100.0
            assert summary["min_confidence"] <= summary["max_confidence"]
            assert summary["avg_time_per_page_ms"] > 0
