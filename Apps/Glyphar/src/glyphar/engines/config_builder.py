"""
Tesseract Configuration Builder
================================

This module provides a deterministic and cache-enabled configuration
builder for the Tesseract OCR engine.

Responsibilities
----------------
- Construct CLI configuration strings.
- Apply engine profile-specific flags.
- Provide LRU caching for repeated builds.

Non-Responsibilities
--------------------
- Does NOT resolve tessdata paths.
- Does NOT validate directory existence.
- Does NOT validate language availability.
- Does NOT execute OCR.

The tessdata directory MUST be resolved and validated externally
(e.g., by the validation layer) before being passed to this builder.
"""

from pathlib import Path
from functools import lru_cache


class TesseractConfigBuilder:
    """
    Build deterministic Tesseract CLI configuration strings.

    Attributes:
        tessdata_dir (Path): Validated tessdata directory.
        model_type (str): Engine profile ("fast", "standard", "best").
    """

    def __init__(self, tessdata_dir: Path, model_type: str):
        """
        Initialize the configuration builder.

        Args:
            tessdata_dir (Path):
                Absolute path to a validated tessdata directory.
            model_type (str):
                Engine profile selector ("fast", "standard", "best").

        Raises:
            ValueError:
                If model_type is unsupported.
        """
        if not isinstance(tessdata_dir, Path):
            raise TypeError("tessdata_dir must be a pathlib.Path instance.")

        if model_type not in {"fast", "standard", "best"}:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                "Expected one of: fast, standard, best."
            )

        self.tessdata_dir = tessdata_dir.resolve()
        self.model_type = model_type

    @lru_cache(maxsize=128)
    def build(self, psm: int, oem: int, extra: str = "") -> str:
        """
        Build a Tesseract CLI configuration string.

        Args:
            psm (int): Page Segmentation Mode.
            oem (int): OCR Engine Mode.
            extra (str, optional): Additional CLI flags.

        Returns:
            str: Fully assembled CLI configuration string.
        """
        cfg_parts = [
            f"--tessdata-dir {self.tessdata_dir}",
            f"--oem {oem}",
            f"--psm {psm}",
        ]

        # Profile-specific optimizations
        if self.model_type == "fast":
            cfg_parts.append("--user-words /tmp/tesseract_user_words.txt")

        elif self.model_type == "best":
            cfg_parts.append("--user-patterns /tmp/tesseract_user_patterns.txt")

        # Stability tuning parameters
        cfg_parts.extend(
            [
                "-c preserve_interword_spaces=1",
                "-c textord_min_linesize=2.5",
                "-c textord_initialx_ile=1.0",
            ]
        )

        if extra:
            cfg_parts.append(extra)

        return " ".join(cfg_parts)
