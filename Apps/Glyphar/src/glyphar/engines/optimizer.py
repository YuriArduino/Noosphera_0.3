"""
Dynamic Tesseract configuration builder with adaptive tuning
for optimized OCR performance across varying document qualities.

This module centralizes OCR parameter optimization logic,
ensuring consistent behavior across engine profiles and
quality classifications.

Design principles:
    - Deterministic behavior (no hidden fallbacks)
    - Explicit quality handling (no silent enum drops)
    - Safe defaults for unknown model types
    - Separation of base profile vs. quality overrides
"""

from typing import Dict, Any, Optional

from glyphar.models.enums import PageQuality


def optimize_ocr_config(
    psm: Optional[int],
    quality: PageQuality,
    model_type: str,
) -> Dict[str, Any]:
    """
    Generate optimized Tesseract configuration based on
    page quality assessment and engine profile.

    Configuration strategy:

        Model Profiles:
            - fast: Lower timeout, faster inference
            - standard: Balanced configuration
            - best: Higher accuracy, longer base timeout

        Quality Overrides:
            - EXCELLENT: Minimal intervention (fast path)
            - GOOD: Conservative defaults
            - FAIR: Mild structural tuning
            - POOR: Aggressive recovery strategy
            - UNKNOWN: Treated conservatively as GOOD

    Args:
        psm:
            Requested Page Segmentation Mode.
            If None, a quality-dependent default is applied.

        quality:
            PageQuality classification from QualityAssessor.

        model_type:
            Engine profile identifier:
                "fast", "standard", or "best".

    Returns:
        Dict[str, Any]:
            {
                "psm": int,
                "oem": int,
                "timeout": int,
                "extra": str
            }

    Raises:
        ValueError:
            If model_type is invalid.

    Example:
        >>> optimize_ocr_config(None, PageQuality.POOR, "fast")
        {
            'psm': 6,
            'oem': 1,
            'timeout': 60,
            'extra': '-c textord_min_linesize=1.5'
        }
    """

    base_profiles = {
        "fast": {"oem": 1, "timeout": 15},
        "standard": {"oem": 2, "timeout": 30},
        "best": {"oem": 3, "timeout": 45},
    }

    if model_type not in base_profiles:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Expected one of: {list(base_profiles.keys())}"
        )

    base_params = base_profiles[model_type]

    # Determine default PSM safely
    default_psm = 3
    final_psm = psm if psm is not None else default_psm

    # EXCELLENT → minimal processing
    if quality == PageQuality.EXCELLENT:
        return {
            **base_params,
            "psm": final_psm,
            "extra": "",
        }

    # GOOD → conservative defaults
    if quality == PageQuality.GOOD:
        return {
            **base_params,
            "psm": final_psm,
            "extra": "",
        }

    # FAIR → mild structural assistance
    if quality == PageQuality.FAIR:
        return {
            **base_params,
            "psm": final_psm,
            "extra": "-c textord_min_linesize=1.2",
        }

    # POOR → aggressive recovery strategy
    if quality == PageQuality.POOR:
        return {
            **base_params,
            "psm": 6 if psm is None else psm,
            "extra": "-c textord_min_linesize=1.5",
            "timeout": max(base_params["timeout"], 60),
        }

    # UNKNOWN → treat conservatively (same as GOOD)
    return {
        **base_params,
        "psm": final_psm,
        "extra": "",
    }
