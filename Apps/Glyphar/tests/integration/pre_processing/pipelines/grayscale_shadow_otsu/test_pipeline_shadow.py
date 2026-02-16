import numpy as np
import pytest

from glyphar.preprocessing.grayscale import GrayscaleStrategy
from glyphar.preprocessing.shadow import ShadowRemovalStrategy
from glyphar.preprocessing.threshold.otsu import OtsuThresholdStrategy


@pytest.mark.integration
def test_pipeline_grayscale_shadow_otsu_synthetic():
    gray_strategy = GrayscaleStrategy()
    shadow_strategy = ShadowRemovalStrategy()
    otsu_strategy = OtsuThresholdStrategy()

    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    gray = gray_strategy.apply(img)
    shadow_corrected = shadow_strategy.apply(gray)
    binary = otsu_strategy.apply(shadow_corrected)

    assert binary.ndim == 2
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
