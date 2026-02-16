"""
Otsu's binarization strategy for document image enhancement.

Automatically determines optimal threshold value by maximizing inter-class
variance between foreground (text) and background (paper). Includes optional
Gaussian pre-blurring to reduce noise sensitivity.

Design rationale:
    - Otsu excels on high-contrast documents with bimodal intensity histograms
    - Pre-blur reduces salt-and-pepper noise that fragments text strokes
    - Fallback to fixed threshold (127) ensures robustness on degenerate images

Trade-offs:
    + Works well on clean scans with uniform lighting
    - Fails on documents with gradual shadows or mixed content types
    - Global threshold cannot handle localized contrast variations

Example:
    >>> strategy = OtsuThresholdStrategy(pre_blur=True)
    >>> binary = strategy.apply(grayscale_image)
"""

import numpy as np
from numpy.typing import NDArray
import cv2
from ..base import PreprocessingStrategy
from ..grayscale import GrayscaleStrategy


# pylint: disable=too-few-public-methods no-member
class OtsuThresholdStrategy(PreprocessingStrategy):
    """
    Binarizes images using Otsu's automatic thresholding method.

    Optimized for document images with clear text/background separation.
    Includes noise resilience via optional Gaussian pre-filtering.

    Attributes:
        pre_blur: Apply Gaussian blur before thresholding (reduces noise sensitivity).
        blur_kernel: Kernel size for Gaussian blur (must be odd integer).
    """

    def __init__(self, pre_blur: bool = True, blur_kernel: int = 3):
        """
        Initialize Otsu thresholding strategy.

        Args:
            pre_blur: Enable Gaussian pre-blur to reduce noise artifacts.
            blur_kernel: Size of Gaussian kernel (3=light, 5=medium, 7=strong).
                Automatically adjusted to nearest odd integer.

        Note:
            Larger kernels increase noise resilience but may blur fine text details.
            Recommended: kernel=3 for high-quality scans, kernel=5 for noisy scans.
        """
        self.pre_blur = pre_blur

        blur_kernel = max(blur_kernel, 1)
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        self.blur_kernel = blur_kernel
        self.grayscale = GrayscaleStrategy()

    def apply(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply Otsu binarization to input image.

        Pipeline:
            1. Convert to grayscale (if needed)
            2. Optional Gaussian blur (noise reduction)
            3. Otsu thresholding with automatic threshold selection
            4. Fallback to fixed threshold on failure

        Args:
            image: Input image (color or grayscale).

        Returns:
            Binary image (uint8, values 0 or 255).

        Robustness:
            - Handles empty images via try/except fallback
            - Preserves text stroke connectivity better than adaptive methods
            - Output always binary (no grayscale remnants)
        """
        gray = self.grayscale.apply(image)

        if self.pre_blur and self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        try:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return thresh.astype(np.uint8, copy=False)
