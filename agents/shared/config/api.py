"""
Global FastAPI Runtime Configuration — Noosphera Shared.

Provides standardized networking, security, and documentation settings
for all REST interfaces within the Noosphera ecosystem.
"""

from pydantic import Field, computed_field, ConfigDict, AliasChoices
from agents.shared.config.base import SharedBaseSettings


class SharedAPISettings(SharedBaseSettings):
    """
    Global API and Networking infrastructure settings.

    This class standardizes how agents bind to the network and how
    their documentation is exposed.

    Features:
        - Support for container-friendly host binding (0.0.0.0).
        - Configurable CORS policies for cross-agent communication.
        - Toggleable Swagger (OpenAPI) and ReDoc interfaces.
        - Automatic URL construction for internal service discovery.

    Design rationale:
        - Using 'API_' as a generic shared prefix allows for global networking
          rules, while AliasChoices allow for 'AGENT_'-level overrides.
        - Computed properties ensure that internal links (like /docs)
          are always accurate regardless of the port assigned.
    """

    # ---------------------------------------------------------------------------
    # NETWORK SERVER
    # ---------------------------------------------------------------------------

    API_HOST: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("AGENT_API_HOST", "API_HOST"),
        description="The network interface the server will bind to.",
    )

    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("AGENT_API_PORT", "API_PORT"),
        description="The TCP port for the REST API.",
    )

    # ---------------------------------------------------------------------------
    # SECURITY & ACCESS
    # ---------------------------------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="List of origins allowed for Cross-Origin Resource Sharing.",
    )

    # ---------------------------------------------------------------------------
    # DOCUMENTATION TABS
    # ---------------------------------------------------------------------------

    API_DOCS_ENABLED: bool = Field(
        default=True,
        description="Master switch to enable/disable Swagger UI (/docs).",
    )

    API_REDOC_ENABLED: bool = Field(
        default=False,
        description="Master switch to enable/disable ReDoc interface (/redoc).",
    )

    # ---------------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # ---------------------------------------------------------------------------

    @computed_field
    @property
    def api_base_url(self) -> str:
        """Constructs the base URL for the server, resolving 0.0.0.0 to localhost."""
        host = "localhost" if self.API_HOST == "0.0.0.0" else self.API_HOST
        return f"http://{host}:{self.API_PORT}"

    @computed_field
    @property
    def docs_url(self) -> str:
        """Generates the direct link to the Swagger UI."""
        return f"{self.api_base_url}/docs"

    model_config = ConfigDict(frozen=True, extra="ignore")


# Global instance for generic tools/services
api_settings = SharedAPISettings()
