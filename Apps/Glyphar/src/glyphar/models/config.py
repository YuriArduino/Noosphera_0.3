"""
Declarative configuration schema for OCR pipeline.

Defines processing intent without exposing implementation details.
Enables reproducibility and API-driven pipeline control.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class OCRConfig(BaseModel):
    """
    Immutable pipeline configuration specification.

    Separates concerns:
        - WHAT to process (languages, DPI)
        - HOW to process (parallelism, timeout)
        - QUALITY targets (min confidence)

    Design constraints:
        - Frozen after creation (no runtime mutation)
        - Defaults optimized for Portuguese documents
        - Validation boundaries prevent pathological configs

    Example (API payload):
        {
            "engine": "tesseract",
            "languages": "por+eng",
            "dpi": 300,
            "parallel": true,
            "max_workers": 4,
            "min_confidence": 70.0
        }
    """

    engine: str = Field(default="tesseract", description="OCR engine identifier")
    languages: str = Field(default="por+eng", description="Tesseract language codes")
    dpi: int = Field(default=300, ge=150, le=600, description="Rendering DPI")

    min_confidence: float = Field(
        default=30.0, ge=0.0, le=100.0, description="Minimum acceptable word confidence"
    )

    parallel: bool = Field(default=False, description="Enable parallel page processing")
    max_workers: Optional[int] = Field(
        default=None, ge=1, le=16, description="Max thread pool size (parallel mode)"
    )
    timeout_per_page_s: int = Field(
        default=30, ge=10, le=300, description="Per-page processing timeout"
    )

    enable_quality_assessment: bool = Field(
        default=True, description="Activate adaptive preprocessing"
    )
    preprocessing_strategies: List[str] = Field(
        default_factory=list,
        description="Allowed preprocessing strategy names (e.g., 'grayscale', 'shadow')",
    )

    model_config = ConfigDict(
        extra="forbid",  # Strict validation — reject unknown fields
        frozen=True,
        json_schema_extra={
            "example": {
                "engine": "tesseract",
                "languages": "por+eng",
                "dpi": 300,
                "parallel": True,
                "max_workers": 4,
            }
        },
    )
