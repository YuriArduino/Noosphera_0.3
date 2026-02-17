from pathlib import Path
import cv2
import numpy as np
import pytest
from pdf2image import convert_from_path

from glyphar.preprocessing.grayscale import GrayscaleStrategy
from glyphar.preprocessing.polarity import PolarityCorrectionStrategy
from glyphar.preprocessing.shadow import ShadowRemovalStrategy
from glyphar.preprocessing.denoise import DenoiseStrategy
from glyphar.preprocessing.deskew import DeskewStrategy
from glyphar.preprocessing.crop import SmartCropStrategy
from glyphar.preprocessing.threshold.adaptive import AdaptiveThresholdStrategy
from glyphar.preprocessing.threshold.otsu import OtsuThresholdStrategy


DATA_DIR = Path("/media/yuri/Noosphera/Noosphera_0.3/Test/Data")
OUTPUT_DIR = Path("tests/output_data/pre_processing/full_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.integration
def test_full_pipeline_real():
    gray_strategy = GrayscaleStrategy()
    polarity_strategy = PolarityCorrectionStrategy()
    shadow_strategy = ShadowRemovalStrategy()
    denoise_strategy = DenoiseStrategy(method="nlm", strength=10.0)
    deskew_strategy = DeskewStrategy()
    crop_strategy = SmartCropStrategy()
    adaptive_strategy = AdaptiveThresholdStrategy()
    otsu_strategy = OtsuThresholdStrategy()

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    assert pdf_files, "Nenhum PDF encontrado."

    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=200)

        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            gray = gray_strategy.apply(img)
            polarity = polarity_strategy.apply(gray)
            shadow = shadow_strategy.apply(polarity)
            denoised = denoise_strategy.apply(shadow)
            deskewed = deskew_strategy.apply(denoised)
            cropped = crop_strategy.apply(deskewed)
            adaptive = adaptive_strategy.apply(cropped)
            binary = otsu_strategy.apply(adaptive)

            output_path = OUTPUT_DIR / f"{pdf_path.stem}_p{i+1}.png"
            cv2.imwrite(str(output_path), binary)

            assert binary.ndim == 2
            assert binary.dtype == np.uint8
            assert set(np.unique(binary)).issubset({0, 255})
