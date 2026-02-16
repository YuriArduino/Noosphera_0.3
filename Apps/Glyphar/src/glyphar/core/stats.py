"""
OCR processing statistics computation.

Pure functions that aggregate page-level metrics into document-level summaries.
No side effects — inputs in, statistics out.
"""

import numpy as np
from models.page import PageResult
from models.stats import ProcessingStatistics
from models.enums import PageQuality


def page_word_count(page: PageResult) -> int:
    """
    Compute total word count for a page.

    Args:
        page: PageResult instance.

    Returns:
        Sum of word counts across all columns.

    Note:
        Handles empty pages/fallback pages gracefully (returns 0).
    """
    return sum(getattr(col, "word_count", 0) for col in getattr(page, "columns", []))


def page_char_count(page: PageResult) -> int:
    """
    Compute total character count for a page.

    Args:
        page: PageResult instance.

    Returns:
        Sum of character counts across all columns.
    """
    return sum(getattr(col, "char_count", 0) for col in getattr(page, "columns", []))


def calculate_statistics(
    *,
    pages_results: list[PageResult],
    confidences: list[float],
    quality_distribution: dict,
    _start_time: float,
    elapsed: float,
    min_confidence: float,
) -> ProcessingStatistics:
    """
    Aggregate page-level metrics into document-level statistics.

    Args:
        pages_results: List of PageResult instances.
        confidences: List of page confidence scores (parallel to pages_results).
        quality_distribution: Histogram of page quality categories (optional).
        start_time: Processing start timestamp (for duration calculation).
        min_confidence: Threshold for successful page classification.

    Returns:
        ProcessingStatistics instance with aggregated metrics.

    Computed metrics:
        - total_words: Sum of all page word counts
        - total_characters: Sum of all page character counts
        - average_confidence: Mean of page confidence scores
        - successful_pages: Pages with confidence > min_confidence
        - failed_pages: Pages with confidence ≤ min_confidence
        - low_confidence_pages: Page numbers with confidence < 60.0%
        - pages_per_second: Throughput metric (computed externally)

    Note:
        Caller must set total_processing_time_s and pages_per_second after
        receiving this object (requires end timestamp not available here).
    """
    total_words = sum(page_word_count(p) for p in pages_results)
    total_chars = sum(page_char_count(p) for p in pages_results)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    successful = sum(
        1 for p in pages_results if p.page_confidence_mean > min_confidence
    )
    failed = len(pages_results) - successful

    low_conf_pages = [
        p.page_number for p in pages_results if p.page_confidence_mean < 60.0
    ]

    pages_per_second = len(pages_results) / elapsed if elapsed > 0 else 0.0

    return ProcessingStatistics(
        total_pages=len(pages_results),
        successful_pages=successful,
        failed_pages=failed,
        total_words=total_words,
        total_characters=total_chars,
        average_confidence=avg_conf,
        total_processing_time_s=elapsed,
        pages_per_second=pages_per_second,
        quality_distribution=quality_distribution,
        low_confidence_pages=low_conf_pages,
    )
