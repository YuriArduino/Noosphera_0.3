"""
Standardized enums for document quality and layout classification.

Used across pipeline stages to maintain consistent classification semantics.
All enums inherit from `str` for seamless JSON serialization and API compatibility.

Design rationale:
    - str inheritance enables direct use in JSON/dicts without `.value` boilerplate
    - Explicit UNKNOWN variants handle edge cases gracefully
    - Values match lowercase strings for internationalization (no accents/special chars)
"""

from enum import Enum


class PageQuality(str, Enum):
    """
    Document image quality classification for preprocessing strategy selection.

    Used by QualityAssessor to determine optimal OCR pipeline configuration:
        - EXCELLENT: Digital-born PDFs → minimal preprocessing (grayscale only)
        - GOOD: High-quality scans → light preprocessing (grayscale + denoise)
        - FAIR: Medium-quality scans → moderate preprocessing (shadow removal)
        - POOR: Low-quality scans → aggressive preprocessing (full stack)
        - UNKNOWN: Unassessed pages → conservative defaults

    Thresholds (empirically validated on Portuguese documents):
        - EXCELLENT: sharpness > 250 AND contrast > 0.6
        - GOOD: sharpness > 150 AND contrast > 0.4
        - FAIR: sharpness > 80 AND contrast > 0.25
        - POOR: below FAIR thresholds

    Example:
        >>> metrics = QualityAssessor.assess(image)
        >>> if metrics["is_clean_digital"]:
        ...     quality = PageQuality.EXCELLENT
        ... else:
        ...     quality = PageQuality.GOOD  # Conservative fallback
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class LayoutType(str, Enum):
    """
    Document layout structure classification for region-based OCR processing.

    Determines how pages are segmented before OCR engine invocation:
        - SINGLE: Process entire page as one region (standard text documents)
        - DOUBLE: Split vertically into two columns (academic papers, books)
        - MULTI: Three or more columns (journals, magazines)
        - COMPLEX: Mixed content (tables, forms, irregular layouts)
        - UNKNOWN: Unassessed layout → default to SINGLE

    Detection strategies:
        - SINGLE/DOUBLE: ColumnLayoutDetector (projection-based, <2ms/page)
        - MULTI/COMPLEX: AdvancedLayoutDetector (feature-based, ~15ms/page)

    Example:
        >>> detector = ColumnLayoutDetector()
        >>> result = detector.detect(image)
        >>> layout = result["layout_type"]  # LayoutType.DOUBLE
        >>> if layout == LayoutType.DOUBLE:
        ...     for region in result["regions"]:
        ...         ocr_region(region)
    """

    SINGLE = "single"
    DOUBLE = "double"
    MULTI = "multi"
    COMPLEX = "complex"
    UNKNOWN = "unknown"
