"""
Adaptive thresholding strategy for non-uniform illumination documents.

Computes pixel-wise thresholds based on local neighborhood statistics.
Essential for documents with shadows, gradients, or uneven lighting where
global thresholding (Otsu) fails.

Design rationale:
    - Gaussian-weighted neighborhoods preserve text stroke continuity
    - Block size adapts to text scale (larger blocks for body text)
    - Automatic downscaling prevents excessive computation on high-res images

Trade-offs:
    + Handles shadows and gradients robustly
    + Preserves fine details in mixed-content documents
    - Slower than global thresholding (~3-5x runtime)
    - May introduce artifacts in very low-contrast regions

Example:
    >>> strategy = AdaptiveThresholdStrategy(block_size=31, C=10)
    >>> binary = strategy.apply(grayscale_image)
"""

import numpy as np
from numpy.typing import NDArray
import cv2
from ..grayscale import GrayscaleStrategy
from ..base import PreprocessingStrategy


# pylint: disable=too-few-public-methods no-member
class AdaptiveThresholdStrategy(PreprocessingStrategy):
    """
    Binarizes images using locally adaptive thresholding.

    Computes threshold per pixel based on weighted average of neighborhood.
    Critical for documents with non-uniform lighting conditions.

    Attributes:
        block_size: Neighborhood size for threshold calculation (must be odd).
            Larger values = smoother thresholds, smaller values = finer detail.
        C: Constant subtracted from mean (fine-tunes sensitivity).
            Positive values reduce false positives (text detection).
        method: Adaptive method (GAUSSIAN_C recommended for text).
        threshold_type: Binary inversion mode (THRESH_BINARY standard).
    """

    def __init__(
        self,
        block_size: int = 29,
        method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type: int = cv2.THRESH_BINARY,
        c_offset: int = 11,
    ):
        """
        Initialize adaptive thresholding strategy.

        Args:
            block_size: Neighborhood size (pixels). Must be odd integer >= 3.
                Automatically adjusted to nearest valid odd value if needed.
                Recommended: 21-31 for body text, 11-15 for fine print.
            c_offset: Constant subtracted from weighted mean. Controls sensitivity:
                - Higher C = stricter threshold (fewer false positives)
                - Lower C = more inclusive (risk of noise inclusion)
            method: cv2.ADAPTIVE_THRESH_MEAN_C or GAUSSIAN_C (preferred).
            threshold_type: cv2.THRESH_BINARY or THRESH_BINARY_INV.

        Performance note:
            Block sizes > 41 significantly increase processing time with
            diminishing returns for document OCR.
        """

        # Enforce minimum size and odd constraint
        block_size = max(block_size, 3)

        if block_size % 2 == 0:
            block_size += 1

        self.block_size = block_size
        self.method = method
        self.threshold_type = threshold_type
        self.c_offset = c_offset

        self.grayscale = GrayscaleStrategy()

    def apply(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply adaptive binarization to input image.

        Pipeline:
            1. Convert to grayscale (if needed)
            2. Optional downscaling for very large images (>3000px)
            3. Gaussian-weighted adaptive thresholding
            4. Upscale result if downscaling was applied

        Args:
            image: Input image (color or grayscale).

        Returns:
            Binary image (uint8, values 0 or 255).

        Optimization:
            Automatically downscales images >3000px to ~2000px max dimension
            to maintain reasonable processing time without significant quality loss.
        """

        gray = self.grayscale.apply(image)
        original_h, original_w = gray.shape

        scaled = False

        # Downscale very large images to maintain performance
        if max(original_h, original_w) > 3000:
            scale = 2000 / max(original_h, original_w)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled = True

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            self.method,
            self.threshold_type,
            self.block_size,
            self.c_offset,  # ← único C válido
        )

        if scaled:
            binary = cv2.resize(
                binary,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = binary.astype(np.uint8, copy=False)
        return binary
