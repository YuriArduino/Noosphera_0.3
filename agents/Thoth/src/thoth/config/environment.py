"""
Environment and Orchestration Configuration — Thoth Agent.

Defines the deployment context (dev/prod), logging verbosity, and
high-level feature flags for MCP and Prefect integration.
"""

import os
from typing import Literal, Optional
from pydantic import Field, ConfigDict

from .base import ThothBaseSettings


class EnvironmentSettings(ThothBaseSettings):
    """
    Configuration for deployment environment, logging, and external orchestrators.

    This class acts as the master switchboard for Thoth's operational capabilities.
    """

    # ---------------------------------------------------------------
    # DEPLOYMENT CONTEXT
    # ---------------------------------------------------------------
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Current deployment tier",
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable verbose output and debug-level execution traces",
    )

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Global logging threshold for the Thoth package",
    )

    # ---------------------------------------------------------------
    # MCP INTEGRATION (Model Context Protocol)
    # ---------------------------------------------------------------
    # Enables communication with the Glyphar MCP server if used
    MCP_ENABLED: bool = Field(
        default=True,
        description="Enable MCP bridge for tool-server communication",
    )

    # ---------------------------------------------------------------
    # PREFECT ORCHESTRATION (Phase 2 Integration)
    # ---------------------------------------------------------------
    PREFECT_ENABLED: bool = Field(
        default=True,  # Activated for the current architecture phase
        description="Enable remote flow triggering via Prefect API",
    )

    PREFECT_FLOW_NAME: str = Field(
        default="thoth-orchestration",
        description="Identifier for the main Agent orchestration flow",
    )

    # Essential for 'get_client()' calls in the Agent Tools
    PREFECT_API_URL: Optional[str] = Field(
        default=os.environ.get("PREFECT_API_URL"),
        description="Remote Prefect Server API endpoint",
    )

    # ---------------------------------------------------------------
    # OPERATIONAL HELPERS
    # ---------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Helper to check if the agent is in a restricted production state."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Helper to enable experimental features or local paths."""
        return self.ENVIRONMENT == "development"

    model_config = ConfigDict(frozen=True, extra="ignore")


# ================================================================
# GLOBAL INSTANCE
# ================================================================
env_settings = EnvironmentSettings()
