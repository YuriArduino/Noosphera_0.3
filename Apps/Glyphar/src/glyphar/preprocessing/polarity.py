"""Detects and corrects inverted polarity (white text on dark background)."""

import cv2
import numpy as np
from .base import PreprocessingStrategy


# pylint: disable=too-few-public-methods no-member
class PolarityCorrectionStrategy(PreprocessingStrategy):
    """
    Ensures standard OCR polarity: dark text on light background.

    Some scanned documents contain inverted polarity
    (white text over dark background), which severely degrades OCR accuracy.

    This strategy:
        - Estimates overall pixel distribution
        - Detects dominant dark background
        - Inverts image only when strongly indicated

    Designed to be applied FIRST in the preprocessing chain.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct inverted polarity.

        Expected:
            uint8 numpy array
            Shape: (H, W) or (H, W, 3)

        Heuristic:
            If >60% of pixels are dark (<128), assume inverted.
        """

        # ---- Contract enforcement (fail fast) ----
        if image.dtype != np.uint8:
            raise ValueError("PolarityCorrectionStrategy expects uint8 image.")

        # ---- Convert to grayscale for analysis only ----
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # ---- Estimate dark pixel ratio ----
        dark_ratio = np.mean(gray < 128)

        # ---- Invert only if strongly dark-dominant ----
        if dark_ratio > 0.60:
            return 255 - image

        return image
