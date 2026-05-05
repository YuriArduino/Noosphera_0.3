# models/__init__.py
"""
Public API contracts for Glyphar's domain models.

ALL external consumers should import data models ONLY from this module.
This provides a stable, public-facing API for all data-carrying objects.

Note: Core domain enums (like `PageQuality`, `BatchStatus`) should now be imported
from `glyphar.core.types` to avoid circular dependencies.

Example:
    >>> from glyphar.models import OCROutput, OCRConfig
    >>> from glyphar.core.types import PageQuality

Import order is CRITICAL to avoid circular dependencies:
    1. Base schemas (file, quality)
    2. Component schemas (column, page)
    3. Composite schemas (config, stats)
    4. Aggregators (output, batch)
"""

# 1. Base schemas
from .file import FileMetadata
from .quality import QualityMetrics

# 2. Component schemas
from .column import ColumnResult
from .page import PageResult

# 3. Composite schemas
from .config import OCRConfig
from .stats import ProcessingStatistics

# 4. Aggregators
from .output import OCROutput
from .batch import BatchTask, BatchResult

__all__ = [
    # Base
    "FileMetadata",
    "QualityMetrics",
    # Components
    "ColumnResult",
    "PageResult",
    # Composites
    "OCRConfig",
    "ProcessingStatistics",
    # Aggregators
    "OCROutput",
    "BatchTask",
    "BatchResult",
]
