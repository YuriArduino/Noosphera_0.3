"""
Fallback OCR strategies for handling recognition failures gracefully.

Provides progressive degradation mechanisms when primary OCR
attempt fails due to timeout or engine errors.

Design principles:
    - Never return empty text
    - Preserve timing metrics
    - Explicitly tag fallback usage
    - Do not mask unexpected programming errors
"""

import time
import subprocess
from typing import Any, Dict

import pytesseract
from pytesseract import TesseractError


def apply_fallback_strategy(
    image: Any,
    languages: str,
    config_builder,
    start_time: float,
    error: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Execute fallback OCR attempts when primary recognition fails.

    Fallback hierarchy:
        1. PSM 6 + OEM 1 → Dense text recovery
        2. PSM 11 + OEM 1 → Sparse text handling
        3. PSM 3 + OEM 0 → Legacy engine fallback

    Only expected OCR-related exceptions are captured.
    Unexpected errors propagate upward intentionally.

    Returns:
        Structured OCR result with degraded confidence and error metadata.
    """

    fallback_attempts = [
        (6, 1, 30.0),
        (11, 1, 20.0),
        (3, 0, 10.0),
    ]

    last_exception: str | None = None

    for psm, oem, confidence in fallback_attempts:
        try:
            config = config_builder.build(psm, oem)

            text = pytesseract.image_to_string(
                image,
                lang=languages,
                config=config,
                timeout=timeout,
            ).strip()

            if text:
                return {
                    "text": text,
                    "confidence": confidence,
                    "words": [],
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                    "config_used": f"fallback_psm{psm}_oem{oem}",
                    "error_original": error[:100],
                }

        except (TesseractError, RuntimeError, subprocess.TimeoutExpired) as e:
            last_exception = f"{type(e).__name__}: {str(e)[:120]}"
            continue

    # Ultimate failure case with explicit error trace
    return {
        "text": f"[OCR FAILED: {error[:50]}]",
        "confidence": 0.0,
        "words": [],
        "word_count": 0,
        "char_count": 0,
        "processing_time_ms": (time.perf_counter() - start_time) * 1000,
        "config_used": "failed_all_fallbacks",
        "error_original": error,
        "fallback_error": last_exception,
    }
