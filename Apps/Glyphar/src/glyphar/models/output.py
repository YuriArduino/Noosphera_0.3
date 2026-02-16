"""
Final OCR output schema — primary API contract for external consumers.

Designed for seamless integration with:
    - REST APIs (JSON-serializable)
    - LLM correction pipelines (llm_ready_text())
    - UI dashboards (summary() view)
    - Audit systems (immutable metadata)
"""

from typing import List, Dict, Any, cast
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from .file import FileMetadata
from .page import PageResult
from .stats import ProcessingStatistics
from .config import OCRConfig


class OCROutput(BaseModel):
    """
    Immutable final output of OCR pipeline processing.

    Represents complete document processing result ready for consumption.
    Structured for:
        - API responses (JSON serialization)
        - LLM ingestion (llm_ready_text())
        - Human review (summary view)
        - Long-term storage (audit trail)

    Immutability guarantees:
        - Frozen after creation (no mutation)
        - All nested objects frozen (FileMetadata, PageResult, etc.)
        - Deterministic serialization

    Example usage:
        # API response
        >>> output = pipeline.process("book.pdf")
        >>> return JSONResponse(output.model_dump())

        # LLM correction input
        >>> llm_input = output.llm_ready_text()
        >>> corrected = llm.correct(llm_input)

        # Dashboard summary
        >>> summary = output.summary()
        >>> print(f"Processed {summary['pages']} pages")
    """

    file_metadata: FileMetadata = Field(..., description="Source file metadata")
    pages: List[PageResult] = Field(..., description="Per-page OCR results")
    full_text: str = Field(..., description="Complete concatenated document text")
    statistics: ProcessingStatistics = Field(..., description="Aggregate metrics")
    config: OCRConfig = Field(..., description="Configuration used for processing")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="ignore", frozen=True)

    @property
    def total_pages(self) -> int:
        """Total pages in document."""
        return len(self.pages)

    def _statistics(self) -> ProcessingStatistics:
        """Typed accessor to avoid static-analysis confusion with FieldInfo."""
        return cast(ProcessingStatistics, self.statistics)

    def _file_metadata(self) -> FileMetadata:
        """Typed accessor to avoid static-analysis confusion with FieldInfo."""
        return cast(FileMetadata, self.file_metadata)

    @property
    def total_words(self) -> int:
        """Total words in document."""
        return self._statistics().total_words

    @property
    def average_confidence(self) -> float:
        """Document-level mean confidence."""
        return self._statistics().average_confidence

    @property
    def needs_llm_correction(self) -> bool:
        """
        Heuristic flag for LLM correction necessity.

        Returns True if average confidence < 90% — threshold where LLM correction
        typically provides net benefit for Portuguese text.
        """
        return self._statistics().average_confidence < 90.0

    @property
    def high_quality_pages(self) -> List[PageResult]:
        """Pages with confidence ≥ 90% (minimal LLM correction needed)."""
        return [p for p in self.pages if p.is_high_quality]

    def llm_ready_text(self) -> str:
        """
        Format text for LLM ingestion with minimal context.

        Structure:
            DOCUMENT: filename
            PAGES: N
            AVERAGE CONFIDENCE: XX.X%
            ----------------------------------------
            [full text]

        No content alteration — preserves OCR output fidelity for LLM correction.
        """
        header = [
            f"DOCUMENT: {self._file_metadata().name}",
            f"PAGES: {self.total_pages}",
            f"AVERAGE CONFIDENCE: {self.average_confidence:.1f}%",
            "-" * 40,
        ]
        return "\n".join(header) + "\n\n" + self.full_text

    def summary(self) -> Dict[str, Any]:
        """
        Lightweight executive summary for UI/logs.

        Returns dict with:
            - file: Filename only (not full path)
            - pages: Total page count
            - words: Total word count
            - average_confidence: Rounded to 1 decimal
            - processing_time_s: Rounded to 2 decimals
            - needs_llm_correction: Boolean flag
        """
        stats = self._statistics()
        return {
            "file": self._file_metadata().name,
            "pages": self.total_pages,
            "words": stats.total_words,
            "average_confidence": round(self.average_confidence, 1),
            "processing_time_s": round(stats.total_processing_time_s, 2),
            "needs_llm_correction": self.needs_llm_correction,
        }
