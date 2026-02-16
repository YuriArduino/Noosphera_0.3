from pathlib import Path
import pytest
import cv2
import numpy as np
from pdf2image import convert_from_path

from glyphar.preprocessing.polarity import PolarityCorrectionStrategy


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/pre_processing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
def test_polarity_on_real_pdfs():
    """Integration test for PolarityCorrectionStrategy on real PDF pages."""

    strategy = PolarityCorrectionStrategy()

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "Nenhum PDF encontrado na pasta de teste."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            processed = strategy.apply(img)

            # Salva para inspeção visual
            OUTPUT_DIR = Path("tests/output_data/pre_processing/polarity")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            output_path = OUTPUT_DIR / f"{pdf_path.stem}_p{i+1}.png"

            cv2.imwrite(str(output_path), processed)

            # Assert mínimo estrutural
            assert processed.dtype == np.uint8
            assert processed.shape[:2] == img.shape[:2]

            print("CWD:", Path.cwd())
            print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
            print("PDFs encontrados:", pdf_files)


if __name__ == "__main__":
    test_polarity_on_real_pdfs()
