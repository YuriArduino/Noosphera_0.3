from pathlib import Path
import pytest
import cv2
import numpy as np
from pdf2image import convert_from_path

from glyphar.preprocessing.shadow import ShadowRemovalStrategy


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/pre_processing/shadow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
def test_shadow_on_real_pdfs():
    strategy = ShadowRemovalStrategy()

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "Nenhum PDF encontrado."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            processed = strategy.apply(img)

            output_path = OUTPUT_DIR / f"{pdf_path.stem}_p{i+1}.png"
            cv2.imwrite(str(output_path), processed)

            assert processed.dtype == np.uint8
            assert len(processed.shape) == 2
