"""
OCR Pipeline Configuration — Thoth Agent.

Defines the operational parameters for OCR execution, including parallelism,
resource limits, timeouts, and strategy fallback sequences.
"""

from typing import List
from pydantic import Field, ConfigDict
from .base import ThothBaseSettings
from ..domain.common import GlypharStrategy


class PipelineSettings(ThothBaseSettings):
    """
    Configuration for OCR pipeline execution and resource management.

    This class controls how the Agent instructs the Prefect workers to
    allocate hardware resources and manage time-to-completion.
    """

    # ---------------------------------------------------------------
    # PARALLELISM & CONCURRENCY
    # ---------------------------------------------------------------
    MAX_WORKERS: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum parallel threads/processes for OCR execution",
    )

    BATCH_SIZE: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of pages processed per atomic batch",
    )

    # ---------------------------------------------------------------
    # EXECUTION TIMEOUTS
    # ---------------------------------------------------------------
    TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=30,
        le=1200,
        description="Global timeout for an entire document processing job",
    )

    TIMEOUT_PER_PAGE: int = Field(
        default=45,
        ge=5,
        le=180,
        description="Maximum time allowed for a single page OCR run",
    )

    # ---------------------------------------------------------------
    # DATA LIMITS & RESOLUTION
    # ---------------------------------------------------------------
    MAX_PAGES: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Hard limit on the number of pages allowed per document",
    )

    MAX_FILE_SIZE_MB: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum input file size in Megabytes",
    )

    DEFAULT_DPI: int = Field(
        default=200,
        ge=72,
        le=600,
        description="Default rendering resolution for PDF rasterization",
    )

    # ---------------------------------------------------------------
    # DOCTRINE STRATEGIES (SST Alignment)
    # ---------------------------------------------------------------
    # These must correspond to filenames in tools/Glyphar/docs/strategies/*.yaml
    INITIAL_STRATEGY: GlypharStrategy = Field(
        default=GlypharStrategy.FAST,
        description="The starting doctrine used for unknown documents",
    )

    STRATEGY_RETRIAL_SEQUENCE: List[GlypharStrategy] = Field(
        default=[GlypharStrategy.FAST, GlypharStrategy.BALANCED, GlypharStrategy.AGGRESSIVE],
        description="The order in which doctrines are attempted during reprocessing",
    )

    model_config = ConfigDict(frozen=True, use_enum_values=False)


# ================================================================
# GLOBAL INSTANCE
# ================================================================
pipeline_settings = PipelineSettings()
