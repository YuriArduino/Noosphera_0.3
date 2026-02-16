"""
OCR engine adapters — abstraction layer for Tesseract and future engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
import numpy.typing as npt


UInt8Image = npt.NDArray[np.uint8]


class OCREngine(ABC):
    """
    Abstract base class for OCR engine adapters.

    Defines the minimal contract required by the OCR pipeline.
    Enables engine swapping without modifying orchestration logic.

    Design principles:
        - Stable minimal output contract
        - Deterministic structure
        - Engine-agnostic interface
        - Fully mockable
    """

    @abstractmethod
    def recognize(
        self,
        image: UInt8Image,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform OCR on a single image.

        Args:
            image (NDArray[np.uint8]):
                Input image in uint8 format (grayscale or BGR).

            config (Dict[str, Any]):
                Engine-specific parameters (e.g., psm, oem).

        Returns:
            Dict[str, Any]:
                Standardized OCR result containing:

                    - text (str):
                        Full extracted text.

                    - confidence (float):
                        Mean word-level confidence (0.0–100.0).

                    - words (List[Dict[str, Any]]):
                        Word-level results with bounding boxes and confidence.

        Contract requirements:
            - Must always return a dictionary.
            - Must always include "text" and "confidence".
            - Must never return None.
            - On failure, implementation should return a valid structure
              (e.g., text="[OCR FAILED]", confidence=0.0).

        Raises:
            RuntimeError:
                Only for unrecoverable engine-level failures.
        """
