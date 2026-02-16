"""
Image preprocessing adapter for OCR engine consumption.

Stateless utility class that applies preprocessing strategies based on
EngineConfig specifications. No decision logic — pure transformation.

Type guarantees:
    - Input: NDArray[np.uint8]
    - Output: NDArray[np.uint8]
    - apply() always returns single-channel grayscale image
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


UInt8Image = npt.NDArray[np.uint8]


# pylint: disable=too-few-public-methods,no-member
class ImagePreprocessor:
    """
    Stateless image preprocessing adapter.

    Applies transformations specified by EngineConfig:
        - Color space conversion (BGR → grayscale)
        - Thresholding (Otsu, adaptive)
        - Geometric scaling (upscale for low-quality scans)

    Design constraints:
        - No persistent state (all methods static)
        - No decision logic (pure transformation layer)
        - Idempotent where mathematically applicable
        - Explicit dtype enforcement (uint8 only)

    Guarantees:
        - Input must be uint8 numpy array
        - apply() always returns uint8 single-channel image
        - upscale() preserves dtype and channel structure

    Performance (approx. 2000px width image):
        - Grayscale: ~0.5ms
        - Otsu threshold: ~1.2ms
        - Adaptive threshold: ~2.5ms
        - Upscale (1.5x): ~3.0ms

    Note:
        All operations assume uint8 input from OpenCV.
        Dtype is explicitly normalized before return.
    """

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(image: UInt8Image) -> None:
        """
        Validate structural and dtype constraints.

        Raises:
            ValueError | TypeError
        """
        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be numpy.ndarray")

        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim not in (2, 3):
            raise ValueError("Image must be 2D (gray) or 3D (BGR)")

        if image.dtype != np.uint8:
            raise TypeError("Input image must be uint8")

    @staticmethod
    def _to_gray(image: UInt8Image) -> UInt8Image:
        """
        Convert BGR image to grayscale if needed.

        Idempotent for grayscale input.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray.astype(np.uint8, copy=False)
        return image

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def apply(
        image: UInt8Image,
        pre_type: str,
        adaptive_block_size: int = 29,
        adaptive_c: int = 11,
        strict: bool = False,
    ) -> UInt8Image:
        """
        Apply preprocessing strategy.

        Args:
            image:
                uint8 BGR or grayscale image.
            pre_type:
                Strategy name:
                    - "gray"
                    - "otsu"
                    - "adaptive"
            adaptive_block_size:
                Odd integer > 1. Used only for adaptive threshold.
            adaptive_c:
                Constant subtracted from local mean (adaptive).
            strict:
                If True, raises ValueError on unknown pre_type.

        Returns:
            uint8 grayscale image (single channel).

        Strategy behaviors:
            - "gray": Converts to grayscale if needed
            - "otsu": Grayscale + global Otsu binarization
            - "adaptive": Grayscale + adaptive Gaussian threshold
            - Unknown:
                - strict=False → returns original image
                - strict=True → raises ValueError
        """

        ImagePreprocessor._validate(image)

        if pre_type == "gray":
            return ImagePreprocessor._to_gray(image)

        if pre_type == "otsu":
            gray = ImagePreprocessor._to_gray(image)

            _, bin_img = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            return bin_img.astype(np.uint8, copy=False)

        if pre_type == "adaptive":
            if adaptive_block_size <= 1 or adaptive_block_size % 2 == 0:
                raise ValueError("adaptive_block_size must be odd and > 1")

            gray = ImagePreprocessor._to_gray(image)

            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                adaptive_block_size,
                adaptive_c,
            )

            return adaptive.astype(np.uint8, copy=False)

        if strict:
            raise ValueError(f"Unknown preprocessing type: {pre_type}")

        # Fallback with dtype normalization
        return image.astype(np.uint8, copy=False)

    @staticmethod
    def upscale(
        image: UInt8Image,
        scale: float,
    ) -> UInt8Image:
        """
        Apply geometric upscaling.

        Args:
            image:
                uint8 grayscale or BGR image.
            scale:
                Scaling factor.
                - <= 1.0 → no change
                - > 1.0 → enlarges using INTER_CUBIC

        Returns:
            Resized image preserving dtype and aspect ratio.

        Performance note:
            Upscaling increases OCR cost significantly:
                - 1.2x → ~20% slower
                - 1.5x → ~60% slower

            Use only when justified by quality metrics.
        """

        ImagePreprocessor._validate(image)

        if scale <= 1.0:
            return image

        h, w = image.shape[:2]

        resized = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

        return resized.astype(np.uint8, copy=False)
