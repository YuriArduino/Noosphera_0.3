"""
Thoth Agent Configuration Module.

Centralized, type-safe configuration via pydantic-settings.
Modular design for maintainability and separation of concerns.

Usage:
    >>> from thoth.config import settings
    >>> print(settings.llm.LLMSTUDIO_BASE_URL)
    >>> print(settings.thresholds.LLM_CORRECTION_THRESHOLD)

    # Or import specific modules:
    >>> from thoth.config import llm_settings
    >>> print(llm_settings.LLMSTUDIO_MODEL)
"""

from typing import TYPE_CHECKING

# ================================================================
# RUNTIME IMPORTS — Actual instances used by the application
# ================================================================
from .base import ThothBaseSettings, PathMixin
from .glyphar import glyphar_settings
from .llm import llm_settings
from .thresholds import threshold_settings
from .pipeline import pipeline_settings
from .memory import memory_settings
from .api import api_settings
from .environment import env_settings

# ================================================================
# TYPE-CHECKING IMPORTS — Classes for IDE hints only
# ================================================================
if TYPE_CHECKING:
    from .glyphar import GlypharSettings
    from .llm import LLMSettings
    from .thresholds import ThresholdSettings
    from .pipeline import PipelineSettings
    from .memory import MemorySettings
    from .api import APISettings
    from .environment import EnvironmentSettings


class ThothSettings:
    """
    Unified settings object that aggregates all configuration modules.

    Provides a single entry point for all Thoth configuration.

    Example:
        >>> from thoth.config import settings
        >>> print(settings.llm.LLMSTUDIO_BASE_URL)
        >>> print(settings.thresholds.REPROCESS_THRESHOLD)
        >>> print(settings.pipeline.MAX_WORKERS)
    """

    def __init__(self) -> None:
        self._glyphar = glyphar_settings
        self._llm = llm_settings
        self._thresholds = threshold_settings
        self._pipeline = pipeline_settings
        self._memory = memory_settings
        self._api = api_settings
        self._environment = env_settings

    # ================================================================
    # MODULE ACCESSORS (Primary Interface)
    # ================================================================
    @property
    def glyphar(self):
        """Glyphar integration settings."""
        return self._glyphar

    @property
    def llm(self):
        """LLMStudio connection settings."""
        return self._llm

    @property
    def thresholds(self):
        """Decision threshold settings."""
        return self._thresholds

    @property
    def pipeline(self):
        """OCR pipeline settings."""
        return self._pipeline

    @property
    def memory(self):
        """Memory and learning settings."""
        return self._memory

    @property
    def api(self):
        """FastAPI server settings."""
        return self._api

    @property
    def environment(self):
        """Environment and logging settings."""
        return self._environment


# ================================================================
# GLOBAL SETTINGS INSTANCE
# ================================================================
settings = ThothSettings()


# ================================================================
# EXPORTS
# ================================================================
__all__ = [
    # Unified settings
    "settings",
    "ThothSettings",
    # Module-specific settings (instances)
    "glyphar_settings",
    "llm_settings",
    "threshold_settings",
    "pipeline_settings",
    "memory_settings",
    "api_settings",
    "env_settings",
    # Base classes
    "ThothBaseSettings",
    "PathMixin",
]
