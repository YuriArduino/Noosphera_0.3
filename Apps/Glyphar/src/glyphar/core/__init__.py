"""
Public API for the `core` package (Apps.Glyphar.core).

This module intentionally keeps imports minimal and explicit.
If there's an import error in a submodule, the error is captured and
re-raised only when the missing symbol is actually accessed, which
helps avoiding circular import crashes at package import time.

Exports (public):
  - PageProcessor
  - FileProcessor
  - OCRPipeline
  - ParallelProcessor
  - ConfigStrategy, OCRConfig (from optimization)
  - ImagePreprocessor
  - QualityAssessor (sibling analysis package)
"""

from typing import TYPE_CHECKING

__all__ = [
    "PageProcessor",
    "FileProcessor",
    "FileResult",
    "OCRPipeline",
    "ParallelProcessor",
    "ConfigStrategy",
    "OCRConfig",
    "ImagePreprocessor",
    "QualityAssessor",
]

# Try eager imports; if anything fails we capture the error and raise lazily.
_import_error = None

try:
    # Core processors / orchestrators (relative imports within package)
    from .page_processor import PageProcessor  # noqa: F401
    from .file_processor import FileProcessor, FileResult  # noqa: F401
    from .pipeline import OCRPipeline  # noqa: F401
    from .parallel_processor import ParallelProcessor  # noqa: F401

    # Optimization helpers
    from .optimization.config_strategy import ConfigStrategy, OCRConfig  # noqa: F401
    from .optimization.image_preprocessor import ImagePreprocessor  # noqa: F401

    # Sibling analysis component (quality assessor)
    # Use relative sibling import; this expects package layout:
    # Apps/Glyphar/analysis/quality_assessor.py
    from ..analysis.quality_assessor import QualityAssessor  # noqa: F401

except Exception as exc:  # pragma: no cover - keep runtime behavior
    # Capture the import error to re-raise when attributes are accessed.
    _import_error = exc


def __getattr__(name: str):
    """
    Lazy error raising: if a symbol is requested but initial imports failed,
    re-raise the original import error with helpful context.
    """
    if _import_error is not None:
        raise ImportError(
            f"Failed to import core submodules when accessing '{name}'. "
            f"Original error: {_import_error!r}. "
            "Check that all submodules (core.*, optimization.*, and analysis.*) "
            "are present and importable."
        ) from _import_error

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # Provide a clean dir() listing even if imports failed.
    return sorted(__all__)
