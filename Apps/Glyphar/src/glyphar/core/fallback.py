"""
Fallback mechanisms for graceful degradation on processing failures.

Provides safe defaults when page processing encounters unrecoverable errors.
Ensures pipeline always returns valid (though degraded) output for LLM correction.
"""

from models.page import PageResult
from models.enums import LayoutType


def create_fallback_page(page_number: int) -> PageResult:
    """
    Generate safe fallback page result on processing failure.

    Returns minimally valid PageResult with:
        - Empty columns list (no text extracted)
        - 0.0 confidence (signals failure to downstream LLM)
        - Warning flag for audit trails
        - UNKNOWN layout type

    Design rationale:
        - Never return None — pipeline expects valid PageResult
        - Confidence = 0.0 ensures LLM prioritizes correction
        - Warning enables post-hoc failure analysis

    Example:
        >>> page = create_fallback_page(42)
        >>> assert page.page_number == 42
        >>> assert page.page_confidence_mean == 0.0
        >>> assert "processing_failed" in page.warnings
    """
    return PageResult(
        page_number=page_number,
        layout_type=LayoutType.UNKNOWN,
        columns=[],
        page_confidence_mean=0.0,
        processing_time_s=0.0,
        warnings=["processing_failed"],
        config_used=None,
    )
