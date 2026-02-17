"""Unit tests for layout detector modules."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from glyphar.layout import AdvancedLayoutDetector, ColumnLayoutDetector, LayoutType


def test_column_detector_returns_single_for_small_image() -> None:
    """Column detector should short-circuit trivial image sizes."""
    detector = ColumnLayoutDetector()
    image = np.zeros((120, 300, 3), dtype=np.uint8)

    result = detector.detect(image)

    assert result["layout_type"] == LayoutType.SINGLE
    assert result["method"] == "trivial"
    assert len(result["regions"]) == 1


def test_advanced_detector_returns_structured_response() -> None:
    """Advanced detector should return standard output keys."""
    detector = AdvancedLayoutDetector()
    image = np.zeros((400, 600), dtype=np.uint8)

    result = detector.detect(image)

    assert "layout_type" in result
    assert "regions" in result
    assert "confidence" in result
    assert result["layout_type"] in {
        LayoutType.SINGLE,
        LayoutType.DOUBLE,
        LayoutType.MULTI,
        LayoutType.COMPLEX,
    }
