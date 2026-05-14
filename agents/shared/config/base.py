"""
Base Configuration Infrastructure — Noosphera Shared.

Provides the core settings classes and utilities used by all agents and tools.
Built on top of Pydantic Settings for robust environment management across
the distributed architecture.
"""

import os
from pathlib import Path
from typing import Optional, Union

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

# -------------------------------------------------------------------------------
# PATH DISCOVERY
# -------------------------------------------------------------------------------
# Discovery of the project root to ensure the global .env (SST) is always found.
# Structure: agents/shared/config/base.py -> agents/shared/config -> agents/shared -> agents -> ROOT
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class SharedBaseSettings(BaseSettings):
    """
    Global Noosphera infrastructure settings.

    This class serves as the root for all configuration modules, ensuring
    consistent environment loading and validation rules.

    Features:
        - Automatic discovery of the global .env file in the project root.
        - Case-insensitive environment variable matching.
        - Strict validation of assignments during runtime.
        - Ignore extra fields to allow a single .env for multiple services.

    Design rationale:
        - Centralizing the .env location prevents path resolution errors when
          running agents from different subdirectories.
        - Case-insensitivity ensures compatibility across different OS shells.
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )


class PathMixin:
    """
    Utility mixin for standardizing path resolution and directory safety.
    """

    @staticmethod
    def resolve_path(
        path: Union[str, Path],
        base: Optional[Path] = None,
    ) -> Path:
        """
        Resolves a path, ensuring it is absolute and that the directory exists.

        Args:
            path: The target path (string or Path object).
            base: An optional base directory to resolve relative paths against.

        Returns:
            A resolved, absolute Path object.

        Use cases:
            - Initializing log directories during agent startup.
            - Setting up data storage paths for OCR or Audio processing.
            - Standardizing relative paths from the .env file.

        Design rationale:
            - Automatic directory creation (mkdir) prevents runtime I/O errors.
            - Supports both string and Path objects for flexibility in Field definitions.
        """
        p = Path(path) if isinstance(path, str) else path

        if base and not p.is_absolute():
            p = base / p

        # Ensure the directory structure is ready for use
        p.mkdir(parents=True, exist_ok=True)

        return p
