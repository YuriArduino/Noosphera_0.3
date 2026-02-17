"""
Thresholding strategies for document binarization.

Exports only strategies that live within this subpackage:
    - OtsuThresholdStrategy: Global thresholding via Otsu's method
    - AdaptiveThresholdStrategy: Local thresholding for non-uniform illumination
"""

from .otsu import OtsuThresholdStrategy
from .adaptive import AdaptiveThresholdStrategy

__all__ = [
    "OtsuThresholdStrategy",
    "AdaptiveThresholdStrategy",
]
