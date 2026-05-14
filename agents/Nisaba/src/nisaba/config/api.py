"""
API Configuration Overrides — Nisaba Agent.

Refines the API identity, metadata, and port allocation specifically
for the Nisaba Agent orchestration service.
"""

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from agents.shared.config.api import SharedAPISettings


class NisabaAPISettings(SharedAPISettings):
    """
    Nisaba-specific API metadata and port settings.

    Inherits networking logic from SharedAPISettings while defining
    Nisaba's unique identity in the agent fleet.

    Features:
        - Unique port assignment (8001) to prevent local collisions.
        - Customized API title and description for Swagger documentation.
        - Service-level isolation via 'NISABA_' environment prefix.

    Use cases:
        - Differentiating Nisaba from Glyphar (8000) or Thoth (8002) in logs.
        - Providing psychoanalytic context in the OpenAPI metadata.

    Design rationale:
        - Hardcoding the port to 8001 here establishes a 'Static Fleet Map',
          making it easier to manage the docker-compose routing.
    """

    # ---------------------------------------------------------------------------
    # NETWORK OVERRIDES
    # ---------------------------------------------------------------------------

    API_PORT: int = Field(
        default=8001,
        description="Nisaba's dedicated port in the Noosphera fleet.",
    )

    # ---------------------------------------------------------------------------
    # API IDENTITY
    # ---------------------------------------------------------------------------

    API_TITLE: str = Field(
        default="Nisaba Agent",
        description="Title displayed in the OpenAPI/Swagger documentation.",
    )

    API_VERSION: str = Field(
        default="0.3.0",
        description="Current version aligned with Noosphera 0.3 project scope.",
    )

    API_DESCRIPTION: str = Field(
        default=(
            "Autonomous Agent for psychoanalytic environment orchestration "
            "and Human-in-the-Loop (HITL) interaction."
        ),
        description="Extended description for the API gateway.",
    )

    # Apply Nisaba prefix to capture NISABA_API_PORT etc. from the .env
    model_config = SettingsConfigDict(env_prefix="NISABA_", frozen=True)


# Global instance for the Nisaba service
nisaba_api_settings = NisabaAPISettings()
