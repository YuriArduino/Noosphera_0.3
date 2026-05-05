# glyphar/core/__init__.py
"""
Public API for the glyphar.core package.

This module provides the primary orchestration components for the OCR pipeline.
It uses lazy import error handling to provide descriptive feedback if
dependencies are missing at runtime.
"""

__all__ = [
    "PageProcessor",
    "FileProcessor",
    "OCRPipeline",
    "ParallelProcessor",
    "ConfigStrategy",
    "EngineConfig",
    "OCRConfig",
    "ImagePreprocessor",
    "QualityAssessor",
]

# Capture import errors to raise them lazily with better context
_IMPORT_ERROR = None

try:
    from .page_processor import PageProcessor
    from .file_processor import FileProcessor
    from .pipeline import OCRPipeline
    from .parallel_processor import ParallelProcessor

    # Optimization sub-package imports
    from glyphar.optimization.config_strategy import ConfigStrategy, EngineConfig
    from glyphar.optimization.image_preprocessor import ImagePreprocessor

    # Domain models
    from glyphar.models.config import OCRConfig

    # Analysis sub-package imports
    from glyphar.analysis.quality_assessor import QualityAssessor

except ImportError as exc:
    _IMPORT_ERROR = exc


def __getattr__(name: str):
    """Raise descriptive import errors when a component is accessed."""
    if _IMPORT_ERROR is not None:
        raise ImportError(
            f"Failed to import core submodules when accessing '{name}'. "
            f"Original error: {_IMPORT_ERROR!r}. "
            "Ensure all dependencies (core, optimization, analysis, models) "
            "are correctly installed."
        ) from _IMPORT_ERROR

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
