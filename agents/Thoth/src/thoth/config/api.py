"""
FastAPI configuration for Thoth Agent.

Server settings, host, port, and API metadata.
"""

from pydantic import Field, computed_field

from .base import ThothBaseSettings


class APISettings(ThothBaseSettings):
    """
    Configuration for FastAPI server.

    Example:
        >>> from thoth.config import api_settings
        >>> uvicorn.run("app:app", host=api_settings.FASTAPI_HOST, port=api_settings.FASTAPI_PORT)
    """

    # ---------------------------------------------------------------
    # SERVER
    # ---------------------------------------------------------------
    FASTAPI_HOST: str = Field(
        default="0.0.0.0",
        description="FastAPI server host",
    )

    FASTAPI_PORT: int = Field(
        default=8001,
        ge=1,
        le=65535,
        description="FastAPI server port",
    )

    # ---------------------------------------------------------------
    # API METADATA
    # ---------------------------------------------------------------
    FASTAPI_TITLE: str = Field(
        default="Thoth OCR Agent",
        description="API title for OpenAPI docs",
    )

    FASTAPI_VERSION: str = Field(
        default="0.1.0",
        description="API version",
    )

    FASTAPI_DESCRIPTION: str = Field(
        default="Agente OCR autônomo para documentos psicanalíticos",
        description="API description for OpenAPI docs",
    )

    # ---------------------------------------------------------------
    # CORS
    # ---------------------------------------------------------------
    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # ---------------------------------------------------------------
    # COMPUTED FIELDS
    # ---------------------------------------------------------------
    @computed_field
    @property
    def api_base_url(self) -> str:
        """Returns base URL for the FastAPI server."""
        host = "localhost" if self.FASTAPI_HOST == "0.0.0.0" else self.FASTAPI_HOST
        return f"http://{host}:{self.FASTAPI_PORT}"

    @computed_field
    @property
    def docs_url(self) -> str:
        """Returns URL for Swagger docs."""
        return f"{self.api_base_url}/docs"


# ================================================================
# GLOBAL INSTANCE
# ================================================================
api_settings = APISettings()
