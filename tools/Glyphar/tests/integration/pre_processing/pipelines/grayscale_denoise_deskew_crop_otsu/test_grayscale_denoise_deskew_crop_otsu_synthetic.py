import numpy as np
import pytest

from glyphar.preprocessing.grayscale import GrayscaleStrategy
from glyphar.preprocessing.denoise import DenoiseStrategy
from glyphar.preprocessing.deskew import DeskewStrategy
from glyphar.preprocessing.crop import SmartCropStrategy
from glyphar.preprocessing.threshold.otsu import OtsuThresholdStrategy


@pytest.mark.integration
def test_pipeline_grayscale_denoise_deskew_crop_otsu_synthetic():
    gray_strategy = GrayscaleStrategy()
    denoise_strategy = DenoiseStrategy(method="nlm", strength=10.0)
    deskew_strategy = DeskewStrategy()
    crop_strategy = SmartCropStrategy()
    otsu_strategy = OtsuThresholdStrategy()

    # Synthetic random image
    img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    gray = gray_strategy.apply(img)
    denoised = denoise_strategy.apply(gray)
    deskewed = deskew_strategy.apply(denoised)
    cropped = crop_strategy.apply(deskewed)
    binary = otsu_strategy.apply(cropped)

    assert binary.ndim == 2
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
