"""
OCR result aggregated at page level.

Combines multiple ColumnResult instances with layout metadata.
Represents complete OCR output for a single document page.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from .column import ColumnResult
from .enums import LayoutType, PageQuality


class PageResult(BaseModel):
    """
    Consolidated OCR result for a single document page.

    Aggregates:
        - Multiple ColumnResult instances (one per detected region)
        - Layout classification (SINGLE/DOUBLE/MULTI)
        - Page-level confidence (mean of column confidences)
        - Processing metadata (duration, warnings)

    Design principles:
        - Columns preserve spatial order (left→right, top→bottom)
        - Confidence is arithmetic mean (simple, interpretable)
        - Warnings are non-blocking (pipeline continues on recoverable errors)

    Example:
        >>> page = PageResult(
        ...     page_number=1,
        ...     layout_type=LayoutType.DOUBLE,
        ...     columns=[col_left, col_right],
        ...     page_confidence_mean=92.3,
        ...     processing_time_s=2.4
        ... )
        >>> full_text = page.get_text(separator="\n\n")
    """

    page_number: int = Field(..., ge=1, description="1-based page number")
    layout_type: LayoutType = Field(default=LayoutType.UNKNOWN)
    columns: List[ColumnResult] = Field(default_factory=list)
    page_quality: PageQuality = Field(default=PageQuality.UNKNOWN)

    page_confidence_mean: float = Field(..., ge=0.0, le=100.0)
    processing_time_s: float = Field(..., ge=0.0)

    config_used: Optional[str] = Field(None, description="Dominant OCR config for page")
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    page_text_hash: Optional[str] = Field(None, description="SHA256 hash of page text")

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        json_schema_extra={
            "example": {
                "page_number": 1,
                "layout_type": "double",
                "columns": 2,
                "page_confidence_mean": 92.3,
                "processing_time_s": 2.4,
            }
        },
    )

    @property
    def total_words(self) -> int:
        """Total words across all columns."""
        return sum(c.word_count for c in self.columns)

    @property
    def total_chars(self) -> int:
        """Total characters across all columns."""
        return sum(c.char_count for c in self.columns)

    @property
    def is_high_quality(self) -> bool:
        """Page qualifies as high-quality (confidence ≥ 90%)."""
        return self.page_confidence_mean >= 90.0

    def get_text(self, separator: str = "\n\n") -> str:
        """
        Concatenate column texts with configurable separator.

        Args:
            separator: String inserted between columns (default: double newline).

        Returns:
            Unified page text preserving column order.

        Note:
            Does not apply post-processing (normalization, spellcheck) — pure OCR output.
        """
        return separator.join(c.text for c in self.columns if c.text.strip())
