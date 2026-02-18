"""
Environment and logging configuration for Thoth Agent.

Deployment environment, debug mode, and log levels.
"""

from typing import Literal
from pydantic import Field

from .base import ThothBaseSettings


class EnvironmentSettings(ThothBaseSettings):
    """
    Configuration for deployment environment and logging.

    Example:
        >>> from thoth.config import env_settings
        >>> if env_settings.DEBUG:
        ...     logging.basicConfig(level=logging.DEBUG)
    """

    # ---------------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------------
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Current deployment environment",
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable debug logging and verbose output",
    )

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ---------------------------------------------------------------
    # MCP CONFIGURATION
    # ---------------------------------------------------------------
    MCP_ENABLED: bool = Field(
        default=True,
        description="Enable MCP integration with Glyphar",
    )

    # ---------------------------------------------------------------
    # PREFECT ORCHESTRATION (Fase 2)
    # ---------------------------------------------------------------
    PREFECT_ENABLED: bool = Field(
        default=False,
        description="Enable Prefect orchestration",
    )

    PREFECT_FLOW_NAME: str = Field(
        default="thoth-orchestration",
        description="Prefect flow name",
    )

    PREFECT_API_URL: str | None = Field(
        default=None,
        description="Prefect API URL (optional)",
    )

    # ---------------------------------------------------------------
    # HELPER METHODS
    # ---------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"


# ================================================================
# GLOBAL INSTANCE
# ================================================================
env_settings = EnvironmentSettings()
