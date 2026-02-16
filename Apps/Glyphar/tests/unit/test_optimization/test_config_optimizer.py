"""Unit tests for optimization config optimizer."""

# pylint: disable=wrong-import-position

from __future__ import annotations

from typing import Any, Mapping

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from glyphar.optimization.config_optimizer import ConfigOptimizer, UInt8Image


class _EngineWithInvalidPrimaryResult:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: UInt8Image, config: Mapping[str, Any]) -> dict[str, Any]:
        """Simulate invalid first response and valid fallback response."""
        _ = (image, config)
        self.calls += 1
        if self.calls == 1:
            return {"text": "only text"}
        return {"text": "fallback", "confidence": 88.0}


class _EngineAlwaysOk:
    def recognize(self, image: UInt8Image, config: Mapping[str, Any]) -> dict[str, Any]:
        """Simulate invalid first response and valid fallback response."""
        _ = (image, config)
        return {"text": "ok", "confidence": 96.5, "words": []}


def test_optimizer_fallbacks_when_primary_result_contract_is_invalid() -> None:
    """Invalid primary payload should trigger fallback execution path."""
    optimizer = ConfigOptimizer(_EngineWithInvalidPrimaryResult())
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = optimizer.find_optimal_config(
        image=image,
        layout_type="single",
        quality_metrics={"is_clean_digital": True, "sharpness": 220.0, "contrast": 0.8},
    )

    assert result["text"] == "fallback"
    assert result["config_used"] == "fallback_psm6"
    assert result["confidence"] == 15.0
    assert "error" in result


def test_optimizer_primary_path_includes_serialized_config() -> None:
    """Valid primary OCR execution should return serialized config metadata."""
    optimizer = ConfigOptimizer(_EngineAlwaysOk())
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = optimizer.find_optimal_config(
        image=image,
        layout_type="single",
        quality_metrics={"is_clean_digital": True, "sharpness": 220.0, "contrast": 0.8},
    )

    assert result["text"] == "ok"
    assert result["config_used"].startswith("gray_psm3_scale1.0_oem1")
    assert "time_s" in result
