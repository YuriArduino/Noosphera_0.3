"""
Thoth Agent Configuration Module.

Centralized, type-safe configuration via Pydantic Settings.
This module acts as the single entry point for all operational parameters,
merging environment variables, YAML defaults, and system constants.

Usage:
    >>> from thoth.config import settings
    >>> print(settings.llm.LLM_BASE_URL)
    >>> print(settings.thresholds.ACCEPTANCE_CEILING)

    # Or import specific modules directly:
    >>> from thoth.config import llm_settings
    >>> print(llm_settings.CHAT_MODEL)
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
    Unified settings facade that aggregates all configuration modules.

    Provides a clean, hierarchical interface to access any part of the
    Agent's configuration without multiple imports.
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
    # MODULE ACCESSORS (Interface)
    # ================================================================
    @property
    def glyphar(self):
        """Glyphar tool integration and doctrine paths."""
        return self._glyphar

    @property
    def llm(self):
        """Language Model and Embedding connection settings."""
        return self._llm

    @property
    def thresholds(self):
        """Strategic decision boundaries and quality gates."""
        return self._thresholds

    @property
    def pipeline(self):
        """Operational execution limits and worker settings."""
        return self._pipeline

    @property
    def memory(self):
        """SST persistence, pgvector, and cognitive learning settings."""
        return self._memory

    @property
    def api(self):
        """Agent's FastAPI server and identity metadata."""
        return self._api

    @property
    def environment(self):
        """Global switches, logging levels, and feature flags."""
        return self._environment


# ================================================================
# GLOBAL SETTINGS INSTANCE (Singleton Pattern)
# ================================================================
settings = ThothSettings()


# ================================================================
# EXPORTS
# ================================================================
__all__ = [
    # Unified entry point
    "settings",
    "ThothSettings",
    # Direct access to module instances
    "glyphar_settings",
    "llm_settings",
    "threshold_settings",
    "pipeline_settings",
    "memory_settings",
    "api_settings",
    "env_settings",
    # Base utilities
    "ThothBaseSettings",
    "PathMixin",
]
