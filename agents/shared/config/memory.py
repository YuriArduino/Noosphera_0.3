from pydantic import Field, ConfigDict

from .base import SharedBaseSettings


class MemorySettings(SharedBaseSettings):

    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/noosphera"
    )

    MEMORY_ENABLED: bool = Field(default=True)

    CHECKPOINT_ENABLED: bool = Field(default=True)

    VECTORSTORE_ENABLED: bool = Field(default=True)

    LEDGER_ENABLED: bool = Field(default=True)

    SEMANTIC_SEARCH_TOP_K: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
    )


memory_settings = MemorySettings()
