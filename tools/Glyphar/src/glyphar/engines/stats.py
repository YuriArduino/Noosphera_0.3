"""
OCR engine performance tracking and statistics collection.

This module provides lightweight, O(1) runtime statistics
for monitoring OCR execution quality and performance.

It is intentionally:
    - Non-intrusive
    - Incremental (no history storage)
    - Suitable for real-time decision making (fallback, optimization)
"""

from typing import Dict


class OCRStats:
    """
    Tracks OCR engine performance metrics for monitoring,
    optimization, and fallback decision strategies.

    Metrics collected:
        - Total pages processed
        - Cumulative processing time
        - Running average confidence
        - Minimum and maximum confidence observed
        - Low-confidence page count (threshold-based)
        - Cache hit/miss ratio

    Design rationale:
        - O(1) updates (no historical storage)
        - Safe for large batch processing
        - Prepared for adaptive pipeline logic

    Example:
        >>> stats = OCRStats()
        >>> stats.update(85.3, 1250.5)
        >>> stats.get_summary()["avg_confidence"]
        85.3
    """

    def __init__(self):
        self.total_pages = 0
        self.total_time_ms = 0.0
        self.avg_confidence = 0.0
        self.min_confidence = float("inf")
        self.max_confidence = 0.0
        self.low_confidence_pages = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def update(self, confidence: float, time_ms: float) -> None:
        """
        Update running statistics with new page-level OCR results.

        Args:
            confidence: Page confidence score (0.0–100.0).
            time_ms: Processing time for the page in milliseconds.
        """
        new_total = self.total_pages + 1

        self.avg_confidence = (
            (self.avg_confidence * self.total_pages) + confidence
        ) / new_total

        self.total_pages = new_total
        self.total_time_ms += time_ms

        self.min_confidence = min(self.min_confidence, confidence)
        self.max_confidence = max(self.max_confidence, confidence)

        if confidence < 50.0:
            self.low_confidence_pages += 1

    def record_cache_hit(self) -> None:
        """Register a cache hit event."""
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Register a cache miss event."""
        self.cache_misses += 1

    def get_summary(self) -> Dict[str, float]:
        """
        Return a snapshot of accumulated statistics.

        Returns:
            Dictionary containing:
                - total_pages
                - avg_confidence
                - min_confidence
                - max_confidence
                - low_confidence_pages
                - total_time_ms
                - avg_time_per_page_ms
                - cache_hit_ratio
        """
        total_cache_ops = self.cache_hits + self.cache_misses

        return {
            "total_pages": self.total_pages,
            "avg_confidence": self.avg_confidence,
            "min_confidence": (self.min_confidence if self.total_pages > 0 else 0.0),
            "max_confidence": self.max_confidence,
            "low_confidence_pages": self.low_confidence_pages,
            "total_time_ms": self.total_time_ms,
            "avg_time_per_page_ms": (
                self.total_time_ms / self.total_pages if self.total_pages > 0 else 0.0
            ),
            "cache_hit_ratio": (
                self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0
            ),
        }
