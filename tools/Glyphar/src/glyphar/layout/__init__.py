"""
Document layout detection for OCR region segmentation.

Public API:
    - LayoutDetector: Abstract base class
    - ColumnLayoutDetector: Fast detector for standard documents (recommended)
    - AdvancedLayoutDetector: Feature-based detector for complex layouts
    - LayoutType: Enum (SINGLE, DOUBLE, MULTI, COMPLEX)
"""

from glyphar.layout.base import LayoutDetector
from glyphar.models.enums import LayoutType
from .column_detector import ColumnLayoutDetector
from .advanced_detector import AdvancedLayoutDetector


__all__ = [
    "LayoutDetector",
    "ColumnLayoutDetector",
    "AdvancedLayoutDetector",
    "LayoutType",
]
