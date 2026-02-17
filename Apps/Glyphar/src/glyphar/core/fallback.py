"""
Fallback utilities for graceful error handling.

Provides safe default results when page processing fails.
Ensures pipeline continues even on individual page errors.
"""

from glyphar.models.page import PageResult
from glyphar.models.column import ColumnResult
from glyphar.models.enums import LayoutType, PageQuality
from glyphar.core.identity import Identity


def create_fallback_page(
    page_number: int,
    doc_prefix: str = "doc",
    doc_date: str = "20260101",
) -> PageResult:
    """
    Create a minimal valid PageResult for failed pages.

    Args:
        page_number: 1-based page number.
        doc_prefix: Document prefix for ID generation.
        doc_date: Date string for ID generation (YYYYMMDD).

    Returns:
        PageResult with empty text, 0.0 confidence, and error indicators.

    Use cases:
        - Page processing exceptions
        - Corrupted image data
        - Layout detection failures

    Design:
        - Always returns valid PageResult (never None)
        - Confidence = 0.0 signals downstream to skip/flag
        - ID is still generated for tracking/audit purposes
    """
    page_id = Identity.canonical_id(doc_prefix, doc_date, page_number)

    return PageResult(
        id=page_id,
        page_number=page_number,
        layout_type=LayoutType.UNKNOWN,
        columns=[
            ColumnResult(
                col_index=1,
                text="[ERROR: Page processing failed]",
                confidence=0.0,
                word_count=0,
                char_count=0,
                processing_time_s=0.0,
                bbox=None,
                region_id="fallback",
                config_used=None,
            )
        ],
        page_quality=PageQuality.UNKNOWN,
        page_confidence_mean=0.0,
        processing_time_s=0.0,
        config_used=None,
        warnings=["Page processing failed - fallback result generated"],
        page_text_hash=None,
    )
