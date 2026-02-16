"""Unit tests for optimization config strategy."""

from __future__ import annotations

from enum import Enum

import pytest

pytest.importorskip("numpy")
pytest.importorskip("cv2")

from glyphar.optimization.config_strategy import ConfigStrategy


class _LayoutType(str, Enum):
    SINGLE = "single"


def test_decide_normalizes_layout_and_handles_invalid_quality_values() -> None:
    """Strategy should normalize layout and safely coerce quality metrics."""
    config = ConfigStrategy.decide(
        layout_type=" DOUBLE ",
        quality={
            "is_clean_digital": False,
            "sharpness": "invalid",
            "contrast": None,
        },
    )

    assert config.pre_type == "adaptive"
    assert config.psm == 6
    assert config.scale == 1.5


def test_decide_supports_layout_enum_input() -> None:
    """Strategy should accept LayoutType enum values directly."""
    config = ConfigStrategy.decide(
        layout_type=_LayoutType.SINGLE,
        quality={
            "is_clean_digital": True,
            "sharpness": 220.0,
            "contrast": 0.8,
        },
    )

    assert config.pre_type == "gray"
    assert config.psm == 3
    assert config.scale == 1.0
    assert config.oem == 1
