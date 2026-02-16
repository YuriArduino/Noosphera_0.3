"""Public API for the ``glyphar.core`` package."""

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

# Try eager imports; if anything fails we capture the error and raise lazily.
_import_error = None

try:
    from .page_processor import PageProcessor  # noqa: F401
    from .file_processor import FileProcessor  # noqa: F401
    from .pipeline import OCRPipeline  # noqa: F401
    from .parallel_processor import ParallelProcessor  # noqa: F401

    from ..optimization.config_strategy import ConfigStrategy, EngineConfig  # noqa: F401
    from ..optimization.image_preprocessor import ImagePreprocessor  # noqa: F401
    from ..models.config import OCRConfig  # noqa: F401

    from ..analysis.quality_assessor import QualityAssessor  # noqa: F401

except Exception as exc:  # pragma: no cover - keep runtime behavior
    _import_error = exc


def __getattr__(name: str):
    """Raise lazy import errors with context."""
    if _import_error is not None:
        raise ImportError(
            f"Failed to import core submodules when accessing '{name}'. "
            f"Original error: {_import_error!r}. "
            "Check that all submodules (core.*, optimization.*, analysis.*, models.*) "
            "are present and importable."
        ) from _import_error

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
