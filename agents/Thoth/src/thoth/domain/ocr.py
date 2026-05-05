"""
OCR Domain Models — Glyphar Output Structure.

Pure business entities representing OCR processing results.
These models describe perception only, providing the structured data
necessary for the Agent's decision-making logic.

This version strictly adheres to the original contract to ensure
compatibility with metrics and downstream consumers.
"""

from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, computed_field, ConfigDict

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
    Input file metadata and identification properties.
    Matches the Glyphar SST for file tracking.
    """

    path: str = Field(..., description="Full absolute path to the source file")
    name: str = Field(..., description="Filename including extension")
    extension: str = Field(..., description="File extension without dot")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(...)
    modified_at: datetime = Field(...)
    hash_sha256: HashSHA256 = Field(..., description="Deduplication hash")
    pages_count: int = Field(..., ge=1)

    model_config = ConfigDict(extra="ignore", frozen=True)


# ================================================================
# COLUMN RESULT
# ================================================================
class ColumnResult(BaseModel):
    """
    OCR result for a single detected text region within a page.
    """

    col_index: int = Field(..., ge=1)
    text: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=100.0)
    word_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    processing_time_s: float = Field(..., ge=0.0)
    bbox: BoundingBox = Field(..., description="Geometric box: (left, top, width, height)")
    region_id: str = Field(...)
    config_used: str = Field(...)

    model_config = ConfigDict(extra="ignore", frozen=True)

    @computed_field
    @property
    def words_per_second(self) -> float:
        """Throughput metric for this specific region."""
        if self.processing_time_s <= 0:
            return 0.0
        return self.word_count / self.processing_time_s


# ================================================================
# PAGE RESULT
# ================================================================
class PageResult(BaseModel):
    """
    OCR result for a single document page.
    """

    id: PageID = Field(..., description="Canonical Page ID")
    page_number: int = Field(..., ge=1)
    layout_type: LayoutType = Field(...)
    columns: List[ColumnResult] = Field(...)
    page_quality: PageQuality = Field(...)
    page_confidence_mean: float = Field(..., ge=0.0, le=100.0)
    processing_time_s: float = Field(..., ge=0.0)
    config_used: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    # RESTORED: This field is now strictly required (Field(...))
    page_text_hash: HashSHA256 = Field(..., description="SHA256 of the page text")

    model_config = ConfigDict(extra="ignore", frozen=True)

    @computed_field
    @property
    def full_text(self) -> str:
        """
        Concatenates all column texts into a unified page string.
        RESTORED: Preserves all columns, including empty ones, using simple join.
        """
        return "\n\n".join(col.text for col in self.columns)

    @computed_field
    @property
    def total_word_count(self) -> int:
        """Sum of words across all regions on the page."""
        return sum(col.word_count for col in self.columns)


# ================================================================
# OCR STATISTICS
# ================================================================
class OCRStatistics(BaseModel):
    """
    Aggregate statistics for the entire document processing job.
    """

    total_pages: int = Field(..., ge=0)
    successful_pages: int = Field(..., ge=0)
    failed_pages: int = Field(..., ge=0)
    total_words: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)
    average_confidence: float = Field(..., ge=0.0, le=100.0)
    total_processing_time_s: float = Field(..., ge=0.0)
    pages_per_second: float = Field(..., ge=0.0)
    quality_distribution: Dict[PageQuality, int] = Field(...)
    low_confidence_pages: List[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", frozen=True)

    @computed_field
    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed pages."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100.0


# ================================================================
# OCR CONFIG
# ================================================================
class OCRConfig(BaseModel):
    """
    Configuration specification used during OCR processing.
    """

    engine: str = Field(default="tesseract")
    languages: str = Field(default="por+eng")
    dpi: int = Field(default=300, ge=72, le=600)
    min_confidence: float = Field(default=30.0, ge=0.0, le=100.0)
    parallel: bool = Field(default=True)
    # RESTORED: Enforced as int, defaults to 4, rejects None.
    max_workers: int = Field(default=4, ge=1, le=32)
    timeout_per_page_s: int = Field(default=30, ge=1)
    enable_quality_assessment: bool = Field(default=True)
    preprocessing_strategies: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", frozen=True)


# ================================================================
# PROCESSING METADATA
# ================================================================
class ProcessingMetadata(BaseModel):
    """
    Technical and lineage metadata regarding the execution environment.
    """

    processor: str = Field(...)
    mode: str = Field(...)
    llm_ready: bool = Field(...)
    doc_prefix: str = Field(...)
    doc_date: str = Field(...)
    batch_id: Optional[str] = Field(default=None)

    model_config = ConfigDict(extra="ignore", frozen=True)


# ================================================================
# OCR OUTPUT (ROOT ENTITY)
# ================================================================
class OCROutput(BaseModel):
    """
    Complete OCR Result aggregate.
    An immutable perceptual snapshot produced by Glyphar.
    """

    file_metadata: FileMetadata
    pages: List[PageResult]
    full_text: str
    statistics: OCRStatistics
    config: OCRConfig
    metadata: ProcessingMetadata
    created_at: datetime

    model_config = ConfigDict(extra="ignore", frozen=True)

    @computed_field
    @property
    def total_pages(self) -> int:
        """Total number of pages processed."""
        return len(self.pages)

    @computed_field
    @property
    def poor_quality_pages(self) -> List[PageResult]:
        """Pages with POOR quality classification."""
        return [p for p in self.pages if p.page_quality == PageQuality.POOR]

    # RESTORED: Original computed fields for min/max confidence
    @computed_field
    @property
    def min_page_confidence(self) -> float:
        """Identification of the lowest page-level confidence."""
        if not self.pages:
            return 0.0
        return min(p.page_confidence_mean for p in self.pages)

    @computed_field
    @property
    def max_page_confidence(self) -> float:
        """Identification of the highest page-level confidence."""
        if not self.pages:
            return 0.0
        return max(p.page_confidence_mean for p in self.pages)
