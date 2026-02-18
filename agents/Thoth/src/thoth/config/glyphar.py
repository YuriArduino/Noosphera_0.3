"""
Glyphar integration configuration for Thoth Agent.

Paths, Tesseract settings, and OCR pipeline configuration.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator

from .base import ThothBaseSettings, PathMixin


class GlypharSettings(ThothBaseSettings, PathMixin):
    """
    Configuration for Glyphar OCR integration.

    Example:
        >>> from thoth.config import glyphar_settings
        >>> print(glyphar_settings.GLYPHAR_ROOT)
        '/path/to/Glyphar'
    """

    # ---------------------------------------------------------------
    # GLYPHAR ROOT PATH
    # ---------------------------------------------------------------
    GLYPHAR_ROOT: Path = Field(
        default=Path(__file__).parent.parent.parent.parent.parent / "tools" / "Glyphar",
        description="Root path to Glyphar OCR tool",
    )

    GLYPHAR_TESSDATA_DIR: Optional[Path] = Field(
        default=None,
        description="Override path to Tesseract training data",
    )

    @field_validator("GLYPHAR_ROOT", mode="before")
    @classmethod
    def validate_glyphar_root(cls, v):
        """Ensure GLYPHAR_ROOT is a Path."""
        return Path(v) if isinstance(v, (str, Path)) else v

    @property
    def glyphar_tessdata(self) -> Path:
        """Returns resolved Tesseract data directory."""
        if self.GLYPHAR_TESSDATA_DIR:
            return self.resolve_path(self.GLYPHAR_TESSDATA_DIR)
        return self.GLYPHAR_ROOT / "src" / "glyphar" / "resources" / "tessdata"

    @property
    def glyphar_config_dir(self) -> Path:
        """Returns Glyphar docs/config directory."""
        return self.GLYPHAR_ROOT / "docs"

    @property
    def glyphar_strategies_dir(self) -> Path:
        """Returns Glyphar strategies directory."""
        return self.glyphar_config_dir / "strategies"

    # ---------------------------------------------------------------
    # TEST DATA PATHS
    # ---------------------------------------------------------------
    TEST_DATA_DIR: Path = Field(
        default=Path(__file__).parent.parent.parent.parent.parent / "Test" / "Data",
        description="Path to test PDFs and ground truth files",
    )

    @property
    def test_pdfs(self) -> list[Path]:
        """Returns list of test PDF files."""
        # type: ignore[no-member]
        # pylint: disable=no-member
        if not self.TEST_DATA_DIR.exists():  # type: ignore[attr-defined]
            return []
        return sorted(self.TEST_DATA_DIR.glob("PDF_*.pdf"))  # type: ignore[attr-defined]

    @property
    def test_ground_truth(self) -> dict[str, Path]:
        """Returns mapping of PDF name to ground truth file."""
        # type: ignore[no-member]
        # pylint: disable=no-member
        if not self.TEST_DATA_DIR.exists():  # type: ignore[attr-defined]
            return {}
        return {
            p.stem: p for p in self.TEST_DATA_DIR.glob("*_GT.txt")  # type: ignore[attr-defined]
        }


# ================================================================
# GLOBAL INSTANCE
# ================================================================
glyphar_settings = GlypharSettings()
