"""
Document deskewing strategy using minimum-area rectangle detection.

Detects page skew angle via contour analysis and applies affine rotation
correction. Critical for book scans with page curvature or misaligned feeds.

Design rationale:
    - Uses Otsu binarization for adaptive thresholding
    - Finds largest contour assuming main text block dominance
    - Limits correction to avoid over-rotation on noisy backgrounds

Trade-offs:
    + Improves OCR accuracy on scanned documents
    + Handles moderate skew (±15° typical)
    - May fail on sparse pages (e.g., single-line documents)
    - Adds minor computational overhead (~5–10ms per page)

Example:
    >>> strategy = DeskewStrategy(max_angle=15.0)
    >>> corrected = strategy.apply(page_image)
"""

import cv2
import numpy as np
import numpy.typing as npt

from .grayscale import GrayscaleStrategy
from .base import PreprocessingStrategy


# pylint: disable=too-few-public-methods no-member
class DeskewStrategy(PreprocessingStrategy):
    """
    Corrects document skew using contour-based angle detection.

    Estimates rotation angle from the minimum-area bounding rectangle
    of the largest text contour.

    Attributes:
        max_angle: Maximum correction angle in degrees (±value).
            Prevents over-rotation on noisy inputs.
    """

    def __init__(self, max_angle: float = 15.0) -> None:
        """
        Initialize deskew strategy.

        Args:
            max_angle: Maximum allowed rotation correction (degrees).
                Typical range: 10–20 degrees.
        """
        self.max_angle = max_angle
        self.grayscale = GrayscaleStrategy()

    def apply(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """
        Detect and correct document skew angle.

        Algorithm:
            1. Convert to grayscale
            2. Apply Otsu threshold (binary inversion)
            3. Detect contours
            4. Select largest contour (assumed main text region)
            5. Compute minimum-area rectangle
            6. Extract rotation angle
            7. Apply affine rotation if within max_angle limit

        Args:
            image: Input image (color or grayscale).

        Returns:
            Rotated image with corrected skew.
            Returns original image if:
                - No contours detected
                - Angle exceeds max_angle
                - Angle is negligible (<0.5°)

        Edge cases handled:
            - Uniform images (no content)
            - Extremely noisy backgrounds
            - Over-rotated scans beyond correction threshold
        """
        gray = self.grayscale.apply(image)

        h, w = gray.shape

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return image

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        # OpenCV angle normalization
        if angle < -45:
            angle += 90

        if abs(angle) > self.max_angle or abs(angle) < 0.5:
            return image

        rotation_matrix = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            angle,
            1.0,
        )

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
        )

        return rotated.astype(np.uint8)
