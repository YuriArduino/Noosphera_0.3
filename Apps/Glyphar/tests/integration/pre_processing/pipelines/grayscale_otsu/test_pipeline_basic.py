import numpy as np
import pytest

from glyphar.preprocessing.grayscale import GrayscaleStrategy
from glyphar.preprocessing.threshold.otsu import OtsuThresholdStrategy


@pytest.mark.integration
def test_pipeline_grayscale_otsu_synthetic():
    gray_strategy = GrayscaleStrategy()
    otsu_strategy = OtsuThresholdStrategy()

    # Synthetic noisy RGB image
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    gray = gray_strategy.apply(img)
    binary = otsu_strategy.apply(gray)

    # Structural validation
    assert binary.ndim == 2
    assert binary.dtype == np.uint8

    # Ensure strictly binary
    unique_vals = np.unique(binary)
    assert set(unique_vals).issubset({0, 255})
