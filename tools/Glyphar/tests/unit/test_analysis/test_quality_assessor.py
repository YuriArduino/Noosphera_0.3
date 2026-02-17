"""Unit tests for analysis quality assessor."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from glyphar.analysis import QualityAssessor


def test_quality_assessor_returns_expected_keys() -> None:
    """Assessor should return stable metrics contract."""
    image = np.zeros((128, 256, 3), dtype=np.uint8)

    result = QualityAssessor.assess(image)

    assert set(result.keys()) == {
        "sharpness",
        "contrast",
        "is_clean_digital",
        "quality_score",
    }
    assert isinstance(result["sharpness"], float)
    assert isinstance(result["contrast"], float)
    assert isinstance(result["is_clean_digital"], bool)
    assert isinstance(result["quality_score"], float)


def test_quality_assessor_detects_low_quality_uniform_image() -> None:
    """Uniform image should be classified as not clean digital."""
    image = np.full((256, 256), 127, dtype=np.uint8)

    result = QualityAssessor.assess(image)

    assert result["sharpness"] == 0.0
    assert result["contrast"] == 0.0
    assert result["is_clean_digital"] is False
    assert result["quality_score"] == 0.0
