"""
OCR Domain Models — Glyphar Output Structure.

Pure business entities representing OCR processing results.
Based on actual Glyphar output (PDF_A, PDF_B, PDF_C JSONs).
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field

from .common import (
    PageQuality,
    LayoutType,
    GlypharStrategy,
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

    Example from PDF_B_Digital.json:
        {
            "path": "/media/.../PDF_B_Digital.pdf",
            "name": "PDF_B_Digital.pdf",
            "size_bytes": 957465,
            "hash_sha256": "6c34a38c...",
            "pages_count": 6
        }
    """
    path: str = Field(..., description="Full path to the file")
    name: str = Field(..., description="Filename with extension")
    extension: str = Field(..., description="File extension (e.g., 'pdf')")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")
    created_at: datetime = Field(..., description="File creation timestamp")
    modified_at: datetime = Field(..., description="File modification timestamp")
    hash_sha256: HashSHA256 = Field(..., description="SHA256 hash of file content")
    pages_count: int = Field(..., ge=1, description="Total number of pages")

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# COLUMN RESULT (Text region within a page)
# ================================================================
class ColumnResult(BaseModel):
    """
    OCR result for a single column/region within a page.

    Example from PDF_B_Digital.json:
        {
            "col_index": 1,
            "text": "ELISABETH ROUDINESCO...",
            "confidence": 71.54,
            "word_count": 13,
            "char_count": 78,
            "bbox": {"left": 53, "top": 22, "width": 1148, "height": 1764}
        }
    """
    col_index: int = Field(..., ge=1, description="Column index (1-based)")
    text: str = Field(..., description="Extracted text from this column")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Word-level confidence mean")
    word_count: int = Field(..., ge=0, description="Number of words detected")
    char_count: int = Field(..., ge=0, description="Number of characters")
    processing_time_s: float = Field(..., ge=0.0, description="Time to process this column")
    bbox: dict[str, int] = Field(..., description="Bounding box: left, top, width, height")
    region_id: str = Field(..., description="Unique region identifier")
    config_used: str = Field(..., description="Tesseract config string used")

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def words_per_second(self) -> float:
        """Processing throughput in words/second."""
        if self.processing_time_s == 0:
            return 0.0
        return self.word_count / self.processing_time_s


# ================================================================
# PAGE RESULT
# ================================================================
class PageResult(BaseModel):
    """
    OCR result for a single page.

    Example from PDF_B_Digital.json:
        {
            "id": "pdf_b_digital_20260206_001",
            "page_number": 1,
            "layout_type": "single",
            "columns": [...],
            "page_quality": "excellent",
            "page_confidence_mean": 71.54,
            "processing_time_s": 1.01
        }
    """
    id: PageID = Field(..., description="Unique page identifier")
    page_number: int = Field(..., ge=1, description="Page number (1-based)")
    layout_type: LayoutType = Field(..., description="Detected layout structure")
    columns: List[ColumnResult] = Field(..., description="Text columns in this page")
    page_quality: PageQuality = Field(..., description="Quality classification")
    page_confidence_mean: float = Field(..., ge=0.0, le=100.0)
    processing_time_s: float = Field(..., ge=0.0)
    config_used: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    page_text_hash: HashSHA256 = Field(..., description="Hash of extracted text")

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def full_text(self) -> str:
        """Concatenated text from all columns."""
        return "\n\n".join(col.text for col in self.columns)

    @computed_field
    @property
    def total_word_count(self) -> int:
        """Total words across all columns."""
        return sum(col.word_count for col in self.columns)

    @computed_field
    @property
    def needs_correction(self) -> bool:
        """Check if page confidence is below correction threshold."""
        return self.page_confidence_mean < 92.0


# ================================================================
# STATISTICS
# ================================================================
class OCRStatistics(BaseModel):
    """
    Processing statistics for the entire document.

    Example from PDF_B_Digital.json:
        {
            "total_pages": 6,
            "successful_pages": 6,
            "average_confidence": 89.56,
            "total_processing_time_s": 9.81
        }
    """
    total_pages: int = Field(..., ge=0)
    successful_pages: int = Field(..., ge=0)
    failed_pages: int = Field(..., ge=0)
    total_words: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)
    average_confidence: float = Field(..., ge=0.0, le=100.0)
    total_processing_time_s: float = Field(..., ge=0.0)
    pages_per_second: float = Field(..., ge=0.0)
    quality_distribution: dict[str, int] = Field(...)
    low_confidence_pages: List[int] = Field(default_factory=list)

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed pages."""
        if self.total_pages == 0:
            return 0.0
        return (self.successful_pages / self.total_pages) * 100.0

    @computed_field
    @property
    def needs_llm_correction(self) -> bool:
        """Check if document confidence is below LLM correction threshold."""
        return self.average_confidence < 92.0


# ================================================================
# OCR CONFIG
# ================================================================
class OCRConfig(BaseModel):
    """
    Configuration used for OCR processing.

    Example from PDF_B_Digital.json:
        {
            "engine": "tesseract",
            "languages": "por+eng",
            "dpi": 200,
            "min_confidence": 20.0,
            "parallel": true,
            "max_workers": 4
        }
    """
    engine: str = Field(..., description="OCR engine name")
    languages: str = Field(..., description="Language codes (e.g., 'por+eng')")
    dpi: int = Field(..., ge=72, le=600)
    min_confidence: float = Field(..., ge=0.0, le=100.0)
    parallel: bool = Field(...)
    max_workers: int = Field(..., ge=1, le=32)
    timeout_per_page_s: int = Field(..., ge=1)
    enable_quality_assessment: bool = Field(...)
    preprocessing_strategies: List[str] = Field(default_factory=list)

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# METADATA (Processing info)
# ================================================================
class ProcessingMetadata(BaseModel):
    """Processing metadata and context."""
    processor: str = Field(..., description="Processor class name")
    mode: str = Field(..., description="Processing mode: sequential | parallel")
    llm_ready: bool = Field(..., description="Whether output is formatted for LLM")
    doc_prefix: str = Field(..., description="Document prefix for page IDs")
    doc_date: str = Field(..., description="Processing date: YYYYMMDD")

    model_config = {"extra": "ignore", "frozen": True}


# ================================================================
# OCR OUTPUT (Complete result)
# ================================================================
class OCROutput(BaseModel):
    """
    Complete OCR processing output.

    This is the main entity returned by Glyphar and consumed by Thoth.
    Immutable (frozen) to ensure data integrity across graph nodes.

    Example structure from PDF_B_Digital.json.
    """
    file_meta FileMetadata = Field(..., description="Input file information")
    pages: List[PageResult] = Field(..., description="Per-page OCR results")
    full_text: str = Field(..., description="Complete document text with markers")
    statistics: OCRStatistics = Field(..., description="Processing statistics")
    config: OCRConfig = Field(..., description="Configuration used")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    created_at: datetime = Field(..., description="Output generation timestamp")

    model_config = {"extra": "ignore", "frozen": True}

    @computed_field
    @property
    def total_pages(self) -> int:
        """Total number of pages processed."""
        return len(self.pages)

    @computed_field
    @property
    def average_confidence(self) -> float:
        """Document-level average confidence."""
        return self.statistics.average_confidence

    @computed_field
    @property
    def needs_llm_correction(self) -> bool:
        """Check if document needs LLM correction."""
        return self.statistics.needs_llm_correction

    @computed_field
    @property
    def poor_quality_pages(self) -> List[PageResult]:
        """List of pages with 'poor' quality."""
        return [p for p in self.pages if p.page_quality == PageQuality.POOR]

    def llm_ready_text(self) -> str:
        """
        Format text for LLM correction.

        Structure (from analysis.yaml):
            === OCR RESULTS - N PAGES ===
            === PAGE 1 | Confidence: XX.X% ===
            [text]
            === END OF DOCUMENT ===
        """
        lines = [f"=== OCR RESULTS - {self.total_pages} PAGES ===", ""]

        for page in self.pages:
            lines.append(f"=== PAGE {page.page_number} | Confidence: {page.page_confidence_mean:.1f}% ===")
            lines.append("")
            lines.append(page.full_text)
            lines.append("")

        lines.append("=== END OF DOCUMENT ===")
        return "\n".join(lines)

    def summary(self) -> dict:
        """
        Generate summary for dashboard/API response.

        Returns structure from summary.json:
            {
                "file": "PDF_B_Digital.pdf",
                "file_hash": "...",
                "pages": 6,
                "words": 1386,
                "avg_confidence": 89.6,
                "processing_time_s": 9.81,
                "needs_llm_correction": true
            }
        """
        return {
            "file": self.file_metadata.name,
            "file_hash": self.file_metadata.hash_sha256,
            "pages": self.total_pages,
            "words": self.statistics.total_words,
            "avg_confidence": round(self.average_confidence, 1),
            "processing_time_s": round(self.statistics.total_processing_time_s, 2),
            "needs_llm_correction": self.needs_llm_correction,
        }
