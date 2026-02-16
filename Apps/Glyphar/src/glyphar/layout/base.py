"""
Abstract base class for document layout detection.

Defines minimal contract for all layout detectors. Stateless design enables
reuse across pipeline stages without side effects.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LayoutDetector(ABC):
    """
    Abstract interface for document layout analysis.

    All concrete detectors must implement `detect()` returning standardized output:
        - layout_type: LayoutType enum (SINGLE/DOUBLE/MULTI/COMPLEX)
        - regions: List of bounding box dicts with col_index
        - confidence: Float 0.0-1.0 (optional but recommended)

    Example:
        >>> detector = ColumnLayoutDetector()
        >>> result = detector.detect(image)
        >>> if result["layout_type"] == LayoutType.DOUBLE:
        ...     process_columns(result["regions"])
    """

    @abstractmethod
    def detect(self, image: Any) -> Dict[str, Any]:
        """
        Analyze document image and return layout structure.

        Args:
            image: Input image (numpy array, BGR or grayscale).

        Returns:
            Dict with keys:
                - layout_type: LayoutType enum value
                - regions: List[{"x": int, "y": int, "w": int, "h": int, "col_index": int}]
                - confidence: Optional float (0.0-1.0)
                - method: Optional string identifying detection strategy
        """
