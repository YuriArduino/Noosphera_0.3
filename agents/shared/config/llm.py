"""
Global LLM Runtime Configuration — Noosphera
"""

from typing import Literal
from pydantic import Field, computed_field, ConfigDict

from shared.config.base import SharedBaseSettings


class SharedLLMSettings(SharedBaseSettings):

    # ---------------------------------------------------------
    # INFRASTRUCTURE
    # ---------------------------------------------------------

    LLM_BASE_URL: str = Field(default="http://127.0.0.1:1234/v1")

    LLM_API_KEY: str = Field(default="sk-local")

    LLM_TIMEOUT: int = Field(default=180)

    LLM_MAX_RETRIES: int = Field(default=3)

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------

    CHAT_MODEL: str = Field(default="nvidia/nemotron-3-nano-4b")

    EMBEDDING_MODEL: str = Field(default="text-embedding-all-minilm-l6-v2-embedding")

    EMBEDDING_DIMENSION: int = Field(default=384)

    # ---------------------------------------------------------
    # LANGUAGE
    # ---------------------------------------------------------

    PROMPT_LANGUAGE: Literal["pt-BR", "en-US"] = Field(default="pt-BR")

    # ---------------------------------------------------------
    # ENDPOINTS
    # ---------------------------------------------------------

    @computed_field
    @property
    def llm_chat_endpoint(self) -> str:
        return f"{self.LLM_BASE_URL.rstrip('/')}/chat/completions"

    @computed_field
    @property
    def llm_embedding_endpoint(self) -> str:
        return f"{self.LLM_BASE_URL.rstrip('/')}/embeddings"

    model_config = ConfigDict(frozen=True, extra="ignore")


llm_settings = SharedLLMSettings()
