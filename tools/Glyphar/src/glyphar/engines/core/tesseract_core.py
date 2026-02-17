"""
Minimal Tesseract OCR engine (pure engine layer).

Responsibilities
----------------
- Invoke `pytesseract.image_to_data` with minimal configuration.
- Convert Tesseract `Output.DICT` into the minimal `OCREngine` contract:
    {
        "text": str,
        "confidence": float,
        "words": List[dict]
    }

Non-responsibilities
--------------------
- No caching
- No optimization logic
- No user dictionary management
- No fallback strategy
- No statistics tracking
- No advanced domain post-processing

All higher-level orchestration concerns must be handled externally.

Error Handling
--------------
- `TesseractError` is converted into `RuntimeError`.
- Unexpected exceptions are intentionally not swallowed and will propagate
  to the caller (orchestrator layer).

This module represents the pure OCR execution layer.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import numpy.typing as npt
import pytesseract
from pytesseract import Output, TesseractError

from glyphar.engines.base import OCREngine


# Explicit image contract: uint8 NumPy array
UInt8Image = npt.NDArray[np.uint8]


class TesseractCoreEngine(OCREngine):
    """
    Pure Tesseract OCR engine.

    This class provides a minimal, deterministic wrapper around
    `pytesseract.image_to_data`.

    Parameters
    ----------
    languages : str, optional
        Language codes passed to Tesseract (e.g. "por+eng").
        Defaults to "por+eng".

    Notes
    -----
    The engine enforces that input images are NumPy arrays with dtype uint8.
    This guarantees compatibility with OpenCV and Tesseract expectations.
    """

    def __init__(self, languages: str = "por+eng") -> None:
        self.languages = languages

    def recognize(self, image: UInt8Image, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute OCR using Tesseract.

        Parameters
        ----------
        image : UInt8Image
            Input image as a NumPy ndarray with dtype uint8.
            Can be grayscale or BGR.
        config : Dict[str, Any]
            Optional configuration dictionary:
                - psm (int): Page segmentation mode (default: 3)
                - oem (int): OCR engine mode (optional)
                - timeout (int): Execution timeout in seconds (optional)

        Returns
        -------
        Dict[str, Any]
            Dictionary with minimal OCR contract:
                {
                    "text": str,
                    "confidence": float,  # 0.0–100.0
                    "words": List[{
                        "text": str,
                        "conf": float,
                        "bbox": {
                            "left": int | None,
                            "top": int | None,
                            "width": int | None,
                            "height": int | None,
                        }
                    }]
                }

        Raises
        ------
        TypeError
            If image is not a NumPy ndarray or not uint8.
        ValueError
            If image is empty.
        RuntimeError
            If Tesseract raises a TesseractError.
        """

        # --- Input validation -------------------------------------------------

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be numpy.ndarray")

        if image.size == 0:
            raise ValueError("image is empty")

        if image.dtype != np.uint8:
            raise TypeError("image dtype must be uint8")

        # --- Config extraction -------------------------------------------------

        psm = config.get("psm", 3)
        oem = config.get("oem", None)

        timeout_raw = config.get("timeout", None)
        timeout: int | None = timeout_raw if isinstance(timeout_raw, int) else None

        # --- Build Tesseract CLI config string --------------------------------

        tess_config_parts: List[str] = [f"--psm {psm}"]
        if isinstance(oem, int):
            tess_config_parts.append(f"--oem {oem}")

        tess_config = " ".join(tess_config_parts)

        # --- Execute Tesseract -------------------------------------------------

        try:
            kwargs: Dict[str, Any] = {
                "lang": self.languages,
                "config": tess_config,
                "output_type": Output.DICT,
            }

            if timeout is not None:
                kwargs["timeout"] = timeout

            raw = pytesseract.image_to_data(image, **kwargs)

        except TesseractError as e:
            raise RuntimeError(f"tesseract error: {e}") from e

        # --- Parse Tesseract output -------------------------------------------

        words_raw = raw.get("text", [])
        confidences_raw = raw.get("conf", [])
        lefts = raw.get("left", [])
        tops = raw.get("top", [])
        widths = raw.get("width", [])
        heights = raw.get("height", [])

        words: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        confs: List[float] = []

        for idx, word in enumerate(words_raw):
            if not word or str(word).strip() == "":
                continue

            try:
                conf_val = float(confidences_raw[idx])
            except (ValueError, TypeError, IndexError):
                conf_val = -1.0

            bbox = {
                "left": int(lefts[idx]) if idx < len(lefts) else None,
                "top": int(tops[idx]) if idx < len(tops) else None,
                "width": int(widths[idx]) if idx < len(widths) else None,
                "height": int(heights[idx]) if idx < len(heights) else None,
            }

            words.append(
                {
                    "text": str(word),
                    "conf": conf_val,
                    "bbox": bbox,
                }
            )

            text_parts.append(str(word))

            if conf_val >= 0:
                confs.append(conf_val)

        # --- Aggregate results -------------------------------------------------

        text = " ".join(text_parts).strip()
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

        return {
            "text": text,
            "confidence": avg_conf,
            "words": words,
        }
