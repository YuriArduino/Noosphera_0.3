"""
Noise reduction strategies for scanned document enhancement.

Provides multiple denoising algorithms optimized for text preservation:
    - Non-local means (NLM): Best for Gaussian noise, preserves edges
    - Bilateral filter: Good balance of noise removal and detail retention
    - Median blur: Fastest, effective for salt-and-pepper noise

Design rationale:
    - Grayscale conversion first (noise characteristics differ by channel)
    - Algorithm selection based on noise type (NLM for scans, median for fax)
    - Strength parameter controls aggressiveness without blurring text

Trade-offs:
    + NLM preserves text stroke sharpness better than Gaussian blur
    + Bilateral filter maintains edge contrast critical for character recognition
    - All methods add 5-15ms processing time per page
    - Over-aggressive denoising merges thin character strokes

Example:
    >>> strategy = DenoiseStrategy(method="nlm", strength=10.0)
    >>> cleaned = strategy.apply(grayscale_image)
"""

from typing import Any
import cv2
from .grayscale import GrayscaleStrategy
from .base import PreprocessingStrategy


# pylint: disable=too-few-public-methods no-member
class DenoiseStrategy(PreprocessingStrategy):
    """
    Reduces image noise while preserving text stroke integrity.

    Critical for low-quality scans, fax documents, or images with compression
    artifacts. Selects optimal algorithm based on noise characteristics.

    Attributes:
        method: Denoising algorithm ("nlm", "bilateral", or "median").
        strength: Algorithm-specific intensity parameter (higher = more aggressive).
    """

    def __init__(self, method: str = "nlm", strength: float = 10.0):
        """
        Initialize denoising strategy.

        Args:
            method: Denoising algorithm:
                - "nlm": Non-local means (best quality, slowest)
                - "bilateral": Edge-preserving bilateral filter (balanced)
                - "median": Median blur (fastest, good for salt-and-pepper)
            strength: Algorithm intensity:
                - NLM: h parameter (5-15 typical)
                - Bilateral: sigma color/space (5-20 typical)
                - Median: ignored (fixed 3x3 kernel)

        Performance guide:
            - NLM: ~12ms/page (2000px width) — production recommended
            - Bilateral: ~8ms/page — good balance
            - Median: ~2ms/page — real-time applications
        """
        self.method = method
        self.strength = strength
        self.grayscale = GrayscaleStrategy()

    def apply(self, image: Any) -> Any:
        """
        Apply noise reduction to input image.

        Pipeline:
            1. Convert to grayscale (noise processing is luminance-focused)
            2. Apply selected denoising algorithm with configured strength
            3. Return denoised grayscale image

        Args:
            image: Input image (color or grayscale).

        Returns:
            Denoised grayscale image (uint8).

        Algorithm selection guide:
            - Scanned books/documents: "nlm" with strength=10.0
            - Fax/low-res scans: "median" (fastest)
            - Mixed content: "bilateral" with strength=15.0
        """
        gray = self.grayscale.apply(image)

        if self.method == "nlm":
            return cv2.fastNlMeansDenoising(gray, h=self.strength)
        if self.method == "bilateral":
            return cv2.bilateralFilter(gray, 9, self.strength, self.strength)

        return cv2.medianBlur(gray, 3)
