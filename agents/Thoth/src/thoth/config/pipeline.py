"""
OCR pipeline configuration for Thoth Agent.

Worker settings, batch sizes, timeouts, and processing limits.
"""

from pydantic import Field

from .base import ThothBaseSettings


class PipelineSettings(ThothBaseSettings):
    """
    Configuration for OCR pipeline execution.

    Controls parallelism, batching, and resource limits.

    Example:
        >>> from thoth.config import pipeline_settings
        >>> result = pipeline.process(
        ...     doc,
        ...     max_workers=pipeline_settings.MAX_WORKERS,
        ...     batch_size=pipeline_settings.BATCH_SIZE,
        ... )
    """

    # ---------------------------------------------------------------
    # PARALLELISM
    # ---------------------------------------------------------------
    MAX_WORKERS: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Maximum parallel workers for OCR processing",
    )

    BATCH_SIZE: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Pages per processing batch",
    )

    # ---------------------------------------------------------------
    # TIMEOUTS
    # ---------------------------------------------------------------
    TIMEOUT_SECONDS: int = Field(
        default=240,
        ge=30,
        le=600,
        description="Timeout per document in seconds",
    )

    TIMEOUT_PER_PAGE: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout per page in seconds",
    )

    # ---------------------------------------------------------------
    # LIMITS
    # ---------------------------------------------------------------
    MAX_PAGES: int = Field(
        default=500,
        ge=1,
        le=2000,
        description="Maximum pages per document",
    )

    MAX_FILE_SIZE_MB: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum file size in MB",
    )

    DPI: int = Field(
        default=200,
        ge=72,
        le=600,
        description="DPI for PDF rasterization",
    )

    # ---------------------------------------------------------------
    # GLYPHAR STRATEGIES
    # ---------------------------------------------------------------
    DEFAULT_STRATEGY: str = Field(
        default="fast_scan",
        description="Default Glyphar processing strategy",
    )

    STRATEGY_FALLBACK: list[str] = Field(
        default=["fast_scan", "high_accuracy", "noisy_documents"],
        description="Strategy fallback order",
    )


# ================================================================
# GLOBAL INSTANCE
# ================================================================
pipeline_settings = PipelineSettings()
