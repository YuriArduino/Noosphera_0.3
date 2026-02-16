"""
Validation utilities for OCR engine configuration and Tesseract setup.
"""

from pathlib import Path
from typing import Optional


def _resolve_default_tessdata() -> Path:
    """
    Resolve internal tessdata directory bundled with the project.

    Expected structure:
        src/glyphar/resources/tessdata
    """
    current_file = Path(__file__).resolve()
    glyphar_root = current_file.parents[1]

    return glyphar_root / "resources" / "tessdata"


def validate_tessdata(
    tessdata_dir: Optional[Path],
    languages: str,
) -> str:
    """
    Validate Tesseract tessdata directory and language availability.

    Behavior:
        - If tessdata_dir is None → uses internal bundled tessdata.
        - Validates directory existence.
        - Validates primary language.
        - Falls back to English if available.

    Args:
        tessdata_dir (Optional[Path]):
            Path to tessdata directory.
            If None → internal default is used.
        languages (str):
            Requested language string (e.g., "por", "por+eng").

    Returns:
        str:
            Validated language string.

    Raises:
        RuntimeError:
            If tessdata directory does not exist.
        RuntimeError:
            If neither primary language nor English exists.
        ValueError:
            If languages string is empty.
    """

    if not languages or not languages.strip():
        raise ValueError("Language string cannot be empty.")

    # Resolve tessdata directory
    if tessdata_dir is None:
        tessdata_dir = _resolve_default_tessdata()

    tessdata_dir = tessdata_dir.resolve()

    if not tessdata_dir.exists():
        raise RuntimeError(f"TESSDATA_DIR not found: {tessdata_dir}")

    languages = languages.strip()
    lang_parts = languages.split("+")
    primary_lang = lang_parts[0]

    primary_model = tessdata_dir / f"{primary_lang}.traineddata"

    if primary_model.exists():
        return languages

    # Attempt fallback to English
    english_model = tessdata_dir / "eng.traineddata"

    if english_model.exists():
        remaining = lang_parts[1:]
        if remaining:
            return "eng+" + "+".join(remaining)
        return "eng"

    raise RuntimeError(
        f"Neither '{primary_lang}.traineddata' nor "
        f"'eng.traineddata' found in {tessdata_dir}"
    )
