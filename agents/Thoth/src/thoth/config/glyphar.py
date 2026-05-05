"""
Glyphar Integration Settings — Thoth Agent.

Configures paths for Glyphar doctrine (YAMLs), Tesseract resources,
and Prefect orchestration targets.
"""

from pathlib import Path
from typing import Optional, List, Dict
from pydantic import Field, field_validator, ConfigDict

from .base import ThothBaseSettings, PathMixin


class GlypharSettings(ThothBaseSettings, PathMixin):
    """
    Configuration for Glyphar OCR integration and doctrine access.

    This class enables the Agent to locate the 'Doctrine' (YAML configs)
    needed to propose experiments and validate results.
    """

    # ---------------------------------------------------------------
    # GLYPHAR INFRASTRUCTURE PATHS
    # ---------------------------------------------------------------
    # Default relative path assumes standard Noosphera directory structure
    GLYPHAR_ROOT: Path = Field(
        default=Path(__file__).parents[5] / "tools" / "Glyphar",
        description="Root path to Glyphar OCR tool source and docs",
    )

    GLYPHAR_TESSDATA_DIR: Optional[Path] = Field(
        default=None,
        description="Optional override for Tesseract training data path",
    )

    @field_validator("GLYPHAR_ROOT", mode="before")
    @classmethod
    def validate_glyphar_root(cls, v):
        """Ensure GLYPHAR_ROOT is cast to a Path object."""
        return Path(v) if isinstance(v, (str, Path)) else v

    @property
    def glyphar_docs_dir(self) -> Path:
        """The source of truth for tool capabilities and strategies."""
        return self.GLYPHAR_ROOT / "docs"

    @property
    def capabilities_dir(self) -> Path:
        """Directory containing analysis.yaml and preprocessing.yaml."""
        return self.glyphar_docs_dir / "capabilities"

    @property
    def strategies_dir(self) -> Path:
        """Directory containing pre-defined strategy YAMLs."""
        return self.glyphar_docs_dir / "strategies"

    @property
    def tradeoffs_dir(self) -> Path:
        """Directory containing performance and memory tradeoff definitions."""
        return self.glyphar_docs_dir / "tradeoffs"

    @property
    def glyphar_tessdata(self) -> Path:
        """Resolved path to the Tesseract resources."""
        if self.GLYPHAR_TESSDATA_DIR:
            return self.resolve_path(self.GLYPHAR_TESSDATA_DIR)
        return self.GLYPHAR_ROOT / "src" / "glyphar" / "resources" / "tessdata"

    # ---------------------------------------------------------------
    # PREFECT ORCHESTRATION
    # ---------------------------------------------------------------
    # The name of the deployment the agent will trigger via Prefect client
    GLYPHAR_FLOW_DEPLOYMENT: str = Field(
        default="Glyphar Ephemeral Task/main-deployment",
        description="Prefect deployment name for isolated OCR runs",
    )

    # ---------------------------------------------------------------
    # TEST DATA & GROUND TRUTH
    # ---------------------------------------------------------------
    TEST_DATA_DIR: Path = Field(
        default=Path(__file__).parents[5] / "Test" / "Data",
        description="Directory containing evaluation PDFs and GT files",
    )

    @property
    def test_pdfs(self) -> List[Path]:
        """Returns a sorted list of available test PDF documents."""
        if not self.TEST_DATA_DIR.exists():
            return []
        return sorted(self.TEST_DATA_DIR.glob("PDF_*.pdf"))

    @property
    def test_ground_truth(self) -> Dict[str, Path]:
        """Maps PDF stems to their corresponding Ground Truth text files."""
        if not self.TEST_DATA_DIR.exists():
            return {}
        return {p.stem.replace("_GT", ""): p for p in self.TEST_DATA_DIR.glob("*_GT.txt")}

    model_config = ConfigDict(case_sensitive=True)


# ================================================================
# GLOBAL INSTANCE
# ================================================================
glyphar_settings = GlypharSettings()
