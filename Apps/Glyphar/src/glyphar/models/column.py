"""
OCR result for a single column or detected region.

Represents atomic unit of OCR output — no layout logic, just text + confidence.
Columns are aggregated into PageResult for document-level representation.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class ColumnResult(BaseModel):
    """
    Atomic OCR result for a single text region.

    Properties:
        - Immutable after creation (frozen config)
        - Minimal metadata (only what's needed for debugging/audit)
        - Confidence normalized to 0.0-100.0 scale (Tesseract-agnostic)
    """

    col_index: int = Field(..., ge=1, description="1-based column/region index")

    text: str = Field(
        default="", description="Extracted text (may be empty on failure)"
    )

    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Mean word confidence (0-100)"
    )

    word_count: int = Field(..., ge=0, description="Total words extracted")

    char_count: int = Field(..., ge=0, description="Total characters extracted")

    processing_time_s: float = Field(
        ..., ge=0.0, description="Region processing duration"
    )

    bbox: Optional[Dict[str, int]] = Field(
        default=None,
        description="Absolute bounding box {x, y, w, h} in page coordinates",
    )

    region_id: Optional[str] = Field(
        default=None,
        description="Unique region identifier (debugging)",
    )

    config_used: Optional[str] = Field(
        default=None,
        description="OCR config variant applied",
    )

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        json_schema_extra={
            "example": {
                "col_index": 1,
                "text": "Primeiras Publicações Psicanalíticas",
                "confidence": 93.8,
                "word_count": 4,
                "char_count": 42,
                "processing_time_s": 0.85,
            }
        },
    )
