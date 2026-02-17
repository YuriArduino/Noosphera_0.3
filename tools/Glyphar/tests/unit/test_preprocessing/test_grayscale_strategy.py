import numpy as np
import pytest
from glyphar.preprocessing.grayscale import GrayscaleStrategy


def test_converts_bgr_to_grayscale():
    strategy = GrayscaleStrategy()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = strategy.apply(img)

    assert gray.ndim == 2
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_idempotent_on_grayscale():
    strategy = GrayscaleStrategy()

    img = np.zeros((50, 50), dtype=np.uint8)
    result = strategy.apply(img)

    assert result is img  # No copy


def test_rejects_wrong_dtype():
    strategy = GrayscaleStrategy()

    img = np.zeros((10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        strategy.apply(img)


def test_rejects_wrong_shape():
    strategy = GrayscaleStrategy()

    img = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(ValueError):
        strategy.apply(img)


def test_black_image_stays_black():
    strategy = GrayscaleStrategy()
    img = np.zeros((20, 20, 3), dtype=np.uint8)

    result = strategy.apply(img)

    assert np.all(result == 0)
