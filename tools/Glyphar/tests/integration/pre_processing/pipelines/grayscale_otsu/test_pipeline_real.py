from pathlib import Path
import cv2
import numpy as np
import pytest
from pdf2image import convert_from_path

from glyphar.preprocessing.grayscale import GrayscaleStrategy
from glyphar.preprocessing.threshold.otsu import OtsuThresholdStrategy


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/pre_processing/grayscale_otsu")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
def test_pipeline_grayscale_otsu_real():
    gray_strategy = GrayscaleStrategy()
    otsu_strategy = OtsuThresholdStrategy(pre_blur=True)

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "Nenhum PDF encontrado."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            gray = gray_strategy.apply(img)
            binary = otsu_strategy.apply(gray)

            output_path = OUTPUT_DIR / f"{pdf_path.stem}_p{i+1}.png"
            cv2.imwrite(str(output_path), binary)

            assert binary.ndim == 2
            assert binary.dtype == np.uint8
            assert set(np.unique(binary)).issubset({0, 255})
