"""
Shadow removal strategy for scanned documents with uneven illumination.

Combines CLAHE (Contrast-Limited Adaptive Histogram Equalization) with
background estimation via median filtering. Critical for book scans with
spine shadows or documents with directional lighting artifacts.

Design rationale:
    - LAB color space separates luminance (L) from chrominance (A/B)
    - CLAHE on L-channel enhances local contrast without amplifying noise
    - Background division normalizes illumination gradients
    - Final histogram equalization maximizes text/background separation

Trade-offs:
    + Recovers text in shadowed regions (spine shadows, page curvature)
    + Preserves color fidelity better than naive histogram stretching
    - Computationally intensive (~15-20ms per page)
    - May over-enhance noise in very dark regions

Example:
    >>> strategy = ShadowRemovalStrategy(clip_limit=2.0)
    >>> cleaned = strategy.apply(scanned_page)
"""

from typing import Any, Tuple
import cv2
import numpy as np
from .base import PreprocessingStrategy


# pylint: disable=too-few-public-methods no-member
class ShadowRemovalStrategy(PreprocessingStrategy):
    """
    Removes shadows and illumination gradients from scanned documents.

    Uses multi-stage pipeline: LAB conversion → CLAHE → background estimation
    → division normalization → histogram equalization. Optimized for book
    scans with spine shadows and documents with directional lighting.

    Attributes:
        clip_limit: CLAHE contrast limit (higher = more aggressive enhancement).
        tile_grid_size: Grid size for CLAHE (smaller = finer local adaptation).
        blur_kernel: Median blur kernel for background estimation (odd integer).
    """

    def __init__(
        self,
        clip_limit: float = 3.7,
        tile_grid_size: Tuple[int, int] = (8, 8),
        blur_kernel: int = 40,
    ):
        """
        Initialize shadow removal strategy.

        Args:
            clip_limit: CLAHE contrast clipping limit (1.0-4.0 typical).
                Lower values preserve natural appearance; higher values enhance
                shadowed regions more aggressively.
            tile_grid_size: Grid subdivisions for CLAHE (e.g., (8,8) = 64 regions).
                Smaller tiles adapt better to local variations but increase noise.
            blur_kernel: Median blur kernel size for background estimation.
                Automatically adjusted to nearest odd integer. Larger kernels
                handle broader illumination gradients but may blur text.(21)

        Recommended presets:
            - Light shadows: clip_limit=1.5, blur_kernel=15
            - Heavy spine shadows: clip_limit=3.0, blur_kernel=31
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def apply(self, image: Any) -> Any:
        """
        Remove shadows and normalize illumination.

        Pipeline:
            1. Convert to LAB color space (separate luminance)
            2. Apply CLAHE to L-channel for local contrast enhancement
            3. Reconstruct image and convert to grayscale
            4. Estimate background via median blur
            5. Divide image by background (normalization)
            6. Final histogram equalization

        Args:
            image: Input image (BGR color or grayscale).

        Returns:
            Shadow-normalized grayscale image (uint8).

        Robustness features:
            - Handles grayscale input by converting to BGR temporarily
            - Adapts blur kernel size to image dimensions (prevents artifacts)
            - Division safety via max(background, 1) to avoid zero division
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        l = clahe.apply(l)

        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # Adaptive kernel size based on image dimensions
        kernel_size = (
            self.blur_kernel
            if min(gray.shape) > self.blur_kernel * 2
            else max(3, min(gray.shape) // 4 | 1)
        )

        bg = cv2.medianBlur(gray, kernel_size)
        bg = np.maximum(bg, 1)  # Avoid division by zero

        result = cv2.divide(gray, bg, scale=255)
        return cv2.equalizeHist(result.astype("uint8"))
