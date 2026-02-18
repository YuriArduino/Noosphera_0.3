"""
OCR Domain Models — Glyphar Output Structure.

Pure business entities representing OCR processing results.
These models describe perception only (no decision logic).
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field

from .common import (
    PageQuality,
    LayoutType,
    HashSHA256,
    PageID,
    BoundingBox,
)


# ================================================================
# FILE METADATA
# ================================================================
class FileMetadata(BaseModel):
    """
    Input file metadata and identification.
    """

    path: str = Field(..., description="Full path to the file")
    name: str = Field(..., description="Filename with extension")
    extension: str = Field(..., description="File extension (e.g., 'pdf')")
    size_bytes: int = Field(..., ge=0)
    created_at: datetime = Field(...)
    modified_at: datetime = Field(...)
    hash_sha256: HashSHA256 = Field(...)
    pages_count: int = Field(..., ge=1)

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# COLUMN RESULT
# ================================================================
class ColumnResult(BaseModel):
    """
    OCR result for a single column/region within a page.
    """

    col_index: int = Field(..., ge=1)
    text: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=100.0)
    word_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    processing_time_s: float = Field(..., ge=0.0)
    bbox: BoundingBox = Field(...)
    region_id: str = Field(...)
    config_used: str = Field(...)

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def words_per_second(self) -> float:
        """
        Throughput metric.
        """
        if self.processing_time_s <= 0:
            return 0.0
        return self.word_count / self.processing_time_s


# ================================================================
# PAGE RESULT
# ================================================================
class PageResult(BaseModel):
    """
    OCR result for a single page.
    """

    id: PageID = Field(...)
    page_number: int = Field(..., ge=1)
    layout_type: LayoutType = Field(...)
    columns: List[ColumnResult] = Field(...)
    page_quality: PageQuality = Field(...)
    page_confidence_mean: float = Field(..., ge=0.0, le=100.0)
    processing_time_s: float = Field(..., ge=0.0)
    config_used: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    page_text_hash: HashSHA256 = Field(...)

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def full_text(self) -> str:
        """
        Docstring for full_text

        :param self: Description
        :return: Description
        :rtype: str
        """

        return "\n\n".join(col.text for col in self.columns)

    @computed_field
    @property
    def total_word_count(self) -> int:
        """
        Docstring for total_word_count"""

        return sum(col.word_count for col in self.columns)


# ================================================================
# OCR STATISTICS
# ================================================================
class OCRStatistics(BaseModel):
    """
    Aggregate statistics for the document.
    """

    total_pages: int = Field(..., ge=0)
    successful_pages: int = Field(..., ge=0)
    failed_pages: int = Field(..., ge=0)
    total_words: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)
    average_confidence: float = Field(..., ge=0.0, le=100.0)
    total_processing_time_s: float = Field(..., ge=0.0)
    pages_per_second: float = Field(..., ge=0.0)
    quality_distribution: dict[PageQuality, int] = Field(...)
    low_confidence_pages: List[int] = Field(default_factory=list)

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def success_rate(self) -> float:
        """Percentage of pages classified as successful (non-poor quality)."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100.0


# ================================================================
# OCR CONFIG
# ================================================================
class OCRConfig(BaseModel):
    """
    Configuration used during OCR processing.
    """

    engine: str = Field(...)
    languages: str = Field(...)
    dpi: int = Field(..., ge=72, le=600)
    min_confidence: float = Field(..., ge=0.0, le=100.0)
    parallel: bool = Field(...)
    max_workers: int = Field(..., ge=1, le=32)
    timeout_per_page_s: int = Field(..., ge=1)
    enable_quality_assessment: bool = Field(...)
    preprocessing_strategies: List[str] = Field(default_factory=list)

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# PROCESSING METADATA
# ================================================================
class ProcessingMetadata(BaseModel):
    """
    Technical processing metadata.
    """

    processor: str = Field(...)
    mode: str = Field(...)
    llm_ready: bool = Field(...)
    doc_prefix: str = Field(...)
    doc_date: str = Field(...)

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# OCR OUTPUT (ROOT ENTITY)
# ================================================================
class OCROutput(BaseModel):
    """
    Complete OCR result.

    Immutable perceptual snapshot produced by Glyphar.
    """

    file_metadata: FileMetadata
    pages: List[PageResult]
    full_text: str
    statistics: OCRStatistics
    config: OCRConfig
    metadata: ProcessingMetadata
    created_at: datetime

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def total_pages(self) -> int:
        """Total number of pages processed."""
        return len(self.pages)

    @computed_field
    @property
    def poor_quality_pages(self) -> List[PageResult]:
        """Pages with poor quality."""
        return [p for p in self.pages if p.page_quality == PageQuality.POOR]

    @computed_field
    @property
    def min_page_confidence(self) -> float:
        """Minimum confidence across all pages."""
        if not self.pages:
            return 0.0
        return min(p.page_confidence_mean for p in self.pages)

    @computed_field
    @property
    def max_page_confidence(self) -> float:
        """Maximum confidence across all pages."""
        if not self.pages:
            return 0.0
        return max(p.page_confidence_mean for p in self.pages)
