"""
Base Configuration Infrastructure — Thoth Agent.

Provides the core settings classes and utilities used by all configuration
sub-modules. Built on top of Pydantic Settings for robust environment management.
"""

from pathlib import Path
from typing import Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict


class NisabaBaseSettings(BaseSettings):
    """
    Base class for all Nisaba configuration modules.

    Features:
        - Automatic environment variable loading.
        - Mandatory 'NISABA_' prefix for system overrides.
        - Case-insensitive name matching.
        - UTF-8 encoding support for configuration files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NISABA_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


class PathMixin:
    """
    Utility mixin for standardizing path resolution across the Agent's filesystem.
    """

    @staticmethod
    def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
        """
        Resolves a path, ensuring it is absolute and that the directory structure exists.

        Args:
            path: The target path (string or Path object).
            base: An optional base directory to resolve relative paths against.

        Returns:
            A resolved, absolute Path object.
        """
        p = Path(path) if isinstance(path, str) else path

        if base and not p.is_absolute():
            p = base / p

        # Ensure the directory exists to prevent I/O errors during Agent execution
        p.mkdir(parents=True, exist_ok=True)
        return p
