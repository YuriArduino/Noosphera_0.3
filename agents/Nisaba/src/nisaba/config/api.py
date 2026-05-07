"""
FastAPI Server Configuration — Nisaba Agent.

Defines the server networking parameters, CORS policies, and API
metadata for the Nisaba Agent's web interface.
"""

from pydantic import Field, computed_field, ConfigDict
from .base import NisabaBaseSettings


class APISettings(NisabaBaseSettings):
    """
    Configuration for the FastAPI REST interface.

    This interface serves as the primary gateway for external systems
    to interact with the Nisaba LangGraph.
    """

    # ---------------------------------------------------------------
    # NETWORK SERVER
    # ---------------------------------------------------------------
    FASTAPI_HOST: str = Field(
        default="0.0.0.0",
        description="The network interface the server will bind to",
    )

    FASTAPI_PORT: int = Field(
        default=8002,  # Glyphar (8000), Nisaba (8001)
        ge=1,
        le=65535,
        description="The TCP port for the REST API",
    )

    # ---------------------------------------------------------------
    # API IDENTITY (English Standard)
    # ---------------------------------------------------------------
    FASTAPI_TITLE: str = Field(
        default="Nisaba Agent",
        description="Title displayed in the OpenAPI/Swagger documentation",
    )

    FASTAPI_VERSION: str = Field(
        default="0.3.0",  # Aligned with Noosphera 0.3
        description="Current version of the Nisaba Agent service",
    )

    FASTAPI_DESCRIPTION: str = Field(
        default="Autonomous Agent for psychoanalytic environment management and orchestration."
        "And human interaction(HITL).",
        description="Extended description for the API documentation",
    )

    # ---------------------------------------------------------------
    # SECURITY & ACCESS
    # ---------------------------------------------------------------
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="List of origins allowed to perform Cross-Origin Resource Sharing",
    )

    # ---------------------------------------------------------------
    # COMPUTED PROPERTIES
    # ---------------------------------------------------------------
    @computed_field
    @property
    def api_base_url(self) -> str:
        """Constructs the base URL for the server."""
        host = "localhost" if self.FASTAPI_HOST == "0.0.0.0" else self.FASTAPI_HOST
        return f"http://{host}:{self.FASTAPI_PORT}"

    @computed_field
    @property
    def docs_url(self) -> str:
        """Generates the direct link to the Swagger UI."""
        return f"{self.api_base_url}/docs"

    model_config = ConfigDict(frozen=True)


# ================================================================
# GLOBAL INSTANCE
# ================================================================
api_settings = APISettings()
