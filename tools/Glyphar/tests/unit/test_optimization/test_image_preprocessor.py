"""Unit tests for optimization image preprocessor."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from glyphar.optimization.image_preprocessor import ImagePreprocessor


def test_apply_unknown_pre_type_falls_back_to_grayscale() -> None:
    """Unknown pre_type should still return grayscale output when non-strict."""
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    result = ImagePreprocessor.apply(image, pre_type="unknown_strategy", strict=False)

    assert result.ndim == 2
    assert result.shape == (40, 60)
    assert result.dtype == np.uint8


def test_apply_normalizes_pre_type_name() -> None:
    """Pre-type normalization should accept whitespace and casing variants."""
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    result = ImagePreprocessor.apply(image, pre_type="  OTSU  ")

    assert result.ndim == 2
    assert result.shape == (30, 30)


def test_validate_rejects_invalid_channel_count() -> None:
    """3D images with invalid channel count must be rejected."""
    image = np.zeros((20, 20, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="1, 3 or 4 channels"):
        ImagePreprocessor.apply(image, pre_type="gray")
