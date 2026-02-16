from pathlib import Path
import pytest
import numpy as np
from pdf2image import convert_from_path
from difflib import SequenceMatcher

from glyphar.engines.validation import validate_tessdata
from glyphar.engines.core.tesseract_core import TesseractCoreEngine
from glyphar.engines.batch import TesseractBatchProcessor


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@pytest.mark.integration
def test_pipeline_vs_core_equivalence():
    """
    This test guarantees that the pipeline does not alter
    the raw OCR output from the core engine.
    """

    languages = validate_tessdata(None, "por")
    core_engine = TesseractCoreEngine(languages)
    batch_engine = TesseractBatchProcessor(core_engine)

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "No PDF files found."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = np.array(pil_img)

            config = {"psm": 3}

            # --- Core direct ---
            core_result = core_engine.recognize(img, config)

            # --- Pipeline (batch wrapper) ---
            batch_result = batch_engine.recognize_batch([img], config)[0]

            # 1️⃣ Structural checks
            assert isinstance(core_result["text"], str)
            assert isinstance(batch_result["text"], str)

            assert isinstance(core_result["words"], list)
            assert isinstance(batch_result["words"], list)

            # 2️⃣ Word count equivalence
            assert len(core_result["words"]) == len(
                batch_result["words"]
            ), f"Word count mismatch on {pdf_path.name} p{i+1}"

            # 3️⃣ Confidence equivalence
            assert (
                abs(core_result["confidence"] - batch_result["confidence"]) < 0.001
            ), f"Confidence mismatch on {pdf_path.name} p{i+1}"

            # 4️⃣ Text similarity (robust comparison)
            similarity = text_similarity(core_result["text"], batch_result["text"])

            assert (
                similarity > 0.99
            ), f"Text similarity too low ({similarity:.4f}) on {pdf_path.name} p{i+1}"

            print(
                f"[equivalence] {pdf_path.name} p{i+1}: "
                f"words={len(core_result['words'])} "
                f"conf={core_result['confidence']:.2f} "
                f"similarity={similarity:.4f}"
            )
