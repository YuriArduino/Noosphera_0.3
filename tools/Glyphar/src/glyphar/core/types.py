"""
Standardized core types and enums for the Glyphar OCR pipeline.

Centralizes all domain classifications (quality, layout, batch lifecycle)
to act as the Single Source of Truth (SST) for both the execution pipeline
and the database schemas (SQLModel/SQLAlchemy).

Design rationale:
    - str inheritance ensures seamless mapping to database VARCHAR/ENUM columns,
      allowing direct queries without `.value` boilerplate in SQLAlchemy.
    - Explicit UNKNOWN variants handle corrupt legacy data or unassessed
      records gracefully without raising mapping exceptions.
    - Values match lowercase strings to ensure strict database collation
      consistency and predictable SQL querying.
"""

from enum import Enum


class PageQuality(str, Enum):
    """
    Document image quality classification for preprocessing strategy selection.

    Serves dual purpose:
      1. Pipeline: Guides QualityAssessor to determine optimal OCR configuration.
      2. Database: Allows querying and filtering pages by quality tier in the DB.

    Pipeline Thresholds (empirically validated on Portuguese documents):
        - EXCELLENT: sharpness > 250 AND contrast > 0.6 (grayscale only)
        - GOOD: sharpness > 150 AND contrast > 0.4 (grayscale + denoise)
        - FAIR: sharpness > 80 AND contrast > 0.25 (shadow removal)
        - POOR: below FAIR thresholds (full aggressive stack)
        - UNKNOWN: Unassessed pages → conservative defaults

    Example:
        >>> # Pipeline usage
        >>> metrics = QualityAssessor.assess(image)
        >>> if metrics["is_clean_digital"]:
        ...     quality = PageQuality.EXCELLENT

        >>> # SQLAlchemy querying usage
        >>> session.query(PageTable).filter(PageTable.page_quality == PageQuality.EXCELLENT)
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class LayoutType(str, Enum):
    """
    Document layout structure classification for region-based OCR processing.

    Determines how pages are segmented before OCR engine invocation and
    records the physical structure of the page in the database.

    Classifications:
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
    """

    SINGLE = "single"
    DOUBLE = "double"
    MULTI = "multi"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class BatchStatus(str, Enum):
    """
    Lifecycle state classification for asynchronous batch processing tasks.

    Acts as the primary state machine tracker in the `ocr_batches` table,
    governing how the pipeline queue interacts with the database.

    States:
        - PENDING: Task is queued in the DB and waiting for worker availability.
        - RUNNING: Task has been picked up by the OCR pipeline.
        - COMPLETED: Task finished successfully and results are persisted.
        - FAILED: Task encountered an unrecoverable error during execution.

    State Transitions:
        PENDING → RUNNING → COMPLETED
                          ↘ FAILED

    Example:
        >>> # Polling database for pending batches
        >>> pending_batches = session.query(BatchTable).filter(
        ...     BatchTable.status == BatchStatus.PENDING
        ... ).all()
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
