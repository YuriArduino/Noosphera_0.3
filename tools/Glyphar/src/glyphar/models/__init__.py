# models/__init__.py
"""
Public API contracts for Glyphar OCR engine.

ALL external consumers should import ONLY from this module:
    >>> from models import OCROutput, OCRConfig, PageQuality

Import order is CRITICAL to avoid circular dependencies:
    1. Enums (no dependencies)
    2. Base schemas (file, quality)
    3. Component schemas (column, page)
    4. Composite schemas (config, stats)
    5. Aggregators (output, batch)
"""

# 1. Enums (MUST be first)
from .enums import PageQuality, LayoutType

# 2. Base schemas
from .file import FileMetadata
from .quality import QualityMetrics

# 3. Component schemas
from .column import ColumnResult
from .page import PageResult

# 4. Composite schemas
from .config import OCRConfig
from .stats import ProcessingStatistics

# 5. Aggregators
from .output import OCROutput
from .batch import BatchTask, BatchResult, BatchStatus

__all__ = [
    # Enums
    "PageQuality",
    "LayoutType",
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
    "BatchStatus",
]
