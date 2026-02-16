"""
Base protocol for OCR image preprocessing strategies.

Defines a structural contract for stateless image transformations
used in the OCR pipeline (grayscale, thresholding, denoising, etc.).

Design principles:
    - Protocol (duck typing) instead of ABC inheritance
    - Stateless strategies
    - Composable in preprocessing chains
"""

from typing import Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt


# pylint: disable=too-few-public-methods, unnecessary-ellipsis
@runtime_checkable
class PreprocessingStrategy(Protocol):
    """
    Structural contract for preprocessing strategies.

    Any class implementing `apply()` with the correct signature
    is considered a valid preprocessing strategy.
    """

    def apply(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """
        Transform image for OCR consumption.

        Expected:
            - uint8 numpy array
            - Shape: (H, W) or (H, W, C)

        Must:
            - Return uint8 numpy array
            - Preserve image integrity (no in-place mutation)
            - Remain stateless
        """
        ...
