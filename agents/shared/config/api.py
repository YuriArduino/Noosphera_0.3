"""
Global FastAPI Runtime Configuration — Noosphera
"""

from pydantic import (
    Field,
    computed_field,
    ConfigDict,
)

from shared.config.base import SharedBaseSettings


class APISettings(SharedBaseSettings):

    # ---------------------------------------------------------
    # NETWORK
    # ---------------------------------------------------------

    API_HOST: str = Field(
        default="0.0.0.0",
        description="Server bind interface",
    )

    API_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="HTTP server port",
    )

    # ---------------------------------------------------------
    # SECURITY
    # ---------------------------------------------------------

    CORS_ORIGINS: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # ---------------------------------------------------------
    # DOCUMENTATION
    # ---------------------------------------------------------

    API_DOCS_ENABLED: bool = Field(
        default=True,
    )

    API_REDOC_ENABLED: bool = Field(
        default=False,
    )

    # ---------------------------------------------------------
    # COMPUTED
    # ---------------------------------------------------------

    @computed_field
    @property
    def api_base_url(self) -> str:

        host = "localhost" if self.API_HOST == "0.0.0.0" else self.API_HOST

        return f"http://{host}:{self.API_PORT}"

    @computed_field
    @property
    def docs_url(self) -> str:

        return f"{self.api_base_url}/docs"

    model_config = ConfigDict(frozen=True)


api_settings = APISettings()
