"""
Aggregated processing statistics for pipeline monitoring.

Pure data carrier — no business logic. Enables:
    - Performance benchmarking
    - Quality trend analysis
    - Cost estimation (pages/second)
"""

from typing import Dict, List
from pydantic import BaseModel, Field, ConfigDict
from .enums import PageQuality


class ProcessingStatistics(BaseModel):
    """
    Immutable aggregation of OCR processing metrics.

    Computed after full document processing completes.
    Used for monitoring, billing, and quality assurance.

    Key metrics:
        - success_rate: (successful / total) pages
        - pages_per_second: Throughput metric
        - quality_distribution: Histogram of PageQuality values
        - estimated_llm_tokens: Rough token count for LLM correction cost estimation

    Example:
        >>> stats = ProcessingStatistics(
        ...     total_pages=300,
        ...     successful_pages=295,
        ...     failed_pages=5,
        ...     total_words=150000,
        ...     average_confidence=89.7,
        ...     total_processing_time_s=45.2,
        ...     pages_per_second=6.64
        ... )
        >>> print(f"Success rate: {stats.success_rate:.1f}%")
    """

    total_pages: int = Field(..., ge=0)
    successful_pages: int = Field(..., ge=0)
    failed_pages: int = Field(..., ge=0)

    total_words: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)

    average_confidence: float = Field(..., ge=0.0, le=100.0)
    total_processing_time_s: float = Field(..., ge=0.0)
    pages_per_second: float = Field(..., ge=0.0)

    quality_distribution: Dict[PageQuality, int] = Field(default_factory=dict)
    low_confidence_pages: List[int] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", frozen=True)

    @property
    def success_rate(self) -> float:
        """Percentage of successfully processed pages."""
        total = self.successful_pages + self.failed_pages
        return (self.successful_pages / total * 100) if total > 0 else 0.0

    @property
    def estimated_llm_tokens(self) -> int:
        """
        Rough token estimate for LLM correction cost planning.

        Heuristic: 1.3 tokens per word (accounts for whitespace/punctuation).
        Conservative estimate — actual LLM tokenizers may differ.
        """
        return int(self.total_words * 1.3)
