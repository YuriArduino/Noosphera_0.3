"""
Image preprocessing strategies for OCR pipeline optimization.

Strategies transform raw document images into optimal input for OCR engines.
Each strategy implements the PreprocessingStrategy protocol and can be
composed into processing pipelines.

Public API:
    Base protocol:
        - PreprocessingStrategy: Structural protocol for all strategies

    Core strategies:
        - GrayscaleStrategy: Color → luminance conversion
        - OtsuThresholdStrategy: Global binarization
        - AdaptiveThresholdStrategy: Local binarization for shadows
        - ShadowRemovalStrategy: Illumination normalization
        - SmartCropStrategy: Content-aware margin removal
        - DenoiseStrategy: Noise reduction with text preservation
        - DeskewStrategy: Skew angle correction
"""

from .base import PreprocessingStrategy
from .grayscale import GrayscaleStrategy
from .threshold.otsu import OtsuThresholdStrategy
from .threshold.adaptive import AdaptiveThresholdStrategy
from .shadow import ShadowRemovalStrategy
from .crop import SmartCropStrategy
from .denoise import DenoiseStrategy
from .deskew import DeskewStrategy

__all__ = [
    "PreprocessingStrategy",
    "GrayscaleStrategy",
    "OtsuThresholdStrategy",
    "AdaptiveThresholdStrategy",
    "ShadowRemovalStrategy",
    "SmartCropStrategy",
    "DenoiseStrategy",
    "DeskewStrategy",
]
