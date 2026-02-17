"""Diagnostic test that exercises the full OCR pipeline on real PDFs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from glyphar.core.pipeline import OCRPipeline
from glyphar.engines.managed.tesseract_managed import TesseractManagedEngine
from glyphar.engines.validation import _resolve_default_tessdata
from glyphar.layout.column_detector import ColumnLayoutDetector
from glyphar.models.config import OCRConfig

pytest.importorskip("cv2")
pytest.importorskip("pdf2image")
pytest.importorskip("pytesseract")


pytestmark = pytest.mark.integration


def _resolve_data_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "Test" / "Data"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Test/Data directory from diagnostics test."
    )


def _resolve_output_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "tests" / "output_data" / "full_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


@pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not available",
)
def test_full_pipeline_diagnostic_real() -> None:
    """Run complete OCR pipeline and persist structured output artifacts."""
    data_dir = _resolve_data_dir()
    output_dir = _resolve_output_dir()

    pdfs = sorted(data_dir.glob("*.pdf"))
    assert pdfs, "No PDF files found in Test/Data"

    config = OCRConfig(dpi=200, min_confidence=20.0, parallel=True, max_workers=4)
    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages="por+eng",
        model_type="fast",
        config=config,
    )

    pipeline = OCRPipeline(
        engine=engine,
        layout_detector=ColumnLayoutDetector(),
        _preprocessing_strategies=[],
        config=config,
        include_llm_input=True,
    )

    summary = []

    for pdf_path in pdfs:
        result = pipeline.process(
            file_path=str(pdf_path),
            parallel=True,
            max_workers=4,
            batch_size=5,
            show_progress=False,
        )

        # Use the summary() method which already includes hash information
        result_summary = result.summary()

        payload = result.model_dump(mode="json")
        _dump_json(output_dir / f"{pdf_path.stem}.json", payload)
        (output_dir / f"{pdf_path.stem}.txt").write_text(
            result.full_text,
            encoding="utf-8",
        )

        # Build summary with hash information from result.summary()
        summary.append(
            {
                "file": result_summary["file"],
                "file_hash": result_summary["file_hash"],
                "pages": result_summary["pages"],
                "page_hashes": result_summary["page_hashes"],
                "words": result_summary["words"],
                "avg_confidence": result_summary["average_confidence"],
                "processing_time_s": result_summary["processing_time_s"],
                "needs_llm_correction": result_summary["needs_llm_correction"],
            }
        )

        assert result.total_pages > 0
        assert isinstance(result.full_text, str)
        assert 0.0 <= result.average_confidence <= 100.0

    _dump_json(
        output_dir / "summary.json",
        {
            "pdf_count": len(summary),
            "results": summary,
        },
    )
