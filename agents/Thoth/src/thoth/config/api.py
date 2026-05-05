"""
FastAPI Server Configuration — Thoth Agent.

Defines the server networking parameters, CORS policies, and API
metadata for the Thoth Agent's web interface.
"""

from pydantic import Field, computed_field, ConfigDict
from .base import ThothBaseSettings


class APISettings(ThothBaseSettings):
    """
    Configuration for the FastAPI REST interface.

    This interface serves as the primary gateway for external systems
    to interact with the Thoth LangGraph.
    """

    # ---------------------------------------------------------------
    # NETWORK SERVER
    # ---------------------------------------------------------------
    FASTAPI_HOST: str = Field(
        default="0.0.0.0",
        description="The network interface the server will bind to",
    )

    FASTAPI_PORT: int = Field(
        default=8001,  # Isolated from Glyphar (usually 8000)
        ge=1,
        le=65535,
        description="The TCP port for the REST API",
    )

    # ---------------------------------------------------------------
    # API IDENTITY (English Standard)
    # ---------------------------------------------------------------
    FASTAPI_TITLE: str = Field(
        default="Thoth Agent",
        description="Title displayed in the OpenAPI/Swagger documentation",
    )

    FASTAPI_VERSION: str = Field(
        default="0.3.0",  # Aligned with Noosphera 0.3
        description="Current version of the Thoth Agent service",
    )

    FASTAPI_DESCRIPTION: str = Field(
        default="Autonomous Agent for psychoanalytic document processing and OCR optimization.",
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
