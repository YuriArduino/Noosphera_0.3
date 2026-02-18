"""
Base configuration module for Thoth Agent.

Shared utilities and base classes for all config modules.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class ThothBaseSettings(BaseSettings):
    """
    Base class for all Thoth configuration modules.

    Provides common settings:
        - Environment file loading (.env)
        - THOTH_ prefix for env variables
        - Case-insensitive matching
        - UTF-8 encoding
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="THOTH_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


class PathMixin:
    """Mixin for path validation and resolution."""

    @staticmethod
    def resolve_path(path: str | Path, base: Path | None = None) -> Path:
        """
        Resolve a path, optionally relative to a base directory.

        Args:
            path: Path to resolve (string or Path object)
            base: Optional base directory for relative paths

        Returns:
            Resolved Path object
        """
        p = Path(path) if isinstance(path, str) else path
        if base and not p.is_absolute():
            p = base / p
        p.mkdir(parents=True, exist_ok=True)
        return p
