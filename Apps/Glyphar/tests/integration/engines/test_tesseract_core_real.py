from pathlib import Path
import pytest
import numpy as np
from pdf2image import convert_from_path
from collections import defaultdict

from glyphar.engines.core.tesseract_core import TesseractCoreEngine


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/engines/tesseract_core_simple")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
def test_ocr_on_real_pdfs():
    engine = TesseractCoreEngine("por")

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "No PDF files found."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = np.array(pil_img)

            result = engine.recognize(img, {"psm": 3})

            formatted_text = group_words_by_line(result["words"])

            output_text = OUTPUT_DIR / f"{pdf_path.stem}_p{i+1}.txt"
            output_text.write_text(formatted_text, encoding="utf-8")

            # --- Qualitative assertions ---
            assert isinstance(result["text"], str)
            assert len(result["text"].strip()) > 0
            assert result["confidence"] >= 0.0
            assert isinstance(result["words"], list)
