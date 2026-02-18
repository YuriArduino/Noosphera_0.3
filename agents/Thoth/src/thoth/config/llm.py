"""
LLMStudio integration configuration for Thoth Agent.

Connection settings, model selection, and API endpoints.
"""

from typing import Literal
from pydantic import Field, computed_field

from .base import ThothBaseSettings


class LLMSettings(ThothBaseSettings):
    """
    Configuration for LLMStudio integration.

    Supports both chat completion and embedding models.

    Example:
        >>> from thoth.config.llm import llm_settings
        >>> print(llm_settings.llm_full_endpoint)
        'http://127.0.0.1:1234/v1/chat/completions'
    """

    # ---------------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------------
    LLMSTUDIO_BASE_URL: str = Field(
        default="http://127.0.0.1:1234",
        description="LLMStudio API base URL",
    )

    # ✅ CORREÇÃO: Endpoints completos já no .env
    LLMSTUDIO_CHAT_ENDPOINT: str = Field(
        default="http://127.0.0.1:1234/v1/chat/completions",
        description="Full chat completion endpoint URL",
    )

    LLMSTUDIO_EMBEDDING_ENDPOINT: str = Field(
        default="http://127.0.0.1:1234/v1/embeddings",
        description="Full embeddings endpoint URL",
    )

    LLMSTUDIO_TIMEOUT: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )

    LLMSTUDIO_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )

    # ---------------------------------------------------------------
    # CHAT MODEL
    # ---------------------------------------------------------------
    LLMSTUDIO_MODEL: str = Field(
        default="meta-llama-3.1-8b-instruct",
        description="Primary model for text correction",
    )

    LLMSTUDIO_MODEL_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for chat completions (low for OCR consistency)",
    )

    LLMSTUDIO_MAX_TOKENS: int = Field(
        default=8000,
        ge=100,
        le=32000,
        description="Maximum tokens for chat completion",
    )

    # ---------------------------------------------------------------
    # EMBEDDING MODEL (Local via LLMStudio)
    # ---------------------------------------------------------------
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-nomic-embed-text-v1.5@q8_0",
        description="Local embedding model via LLMStudio",
    )

    EMBEDDING_VIA_LLMSTUDIO: bool = Field(
        default=True,
        description="Use LLMStudio for embeddings instead of HuggingFace",
    )

    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Embedding dimension (depends on model)",
    )

    # ---------------------------------------------------------------
    # PROMPT LANGUAGE
    # ---------------------------------------------------------------
    PROMPT_LANGUAGE: Literal["pt-BR", "en-US"] = Field(
        default="pt-BR",
        description="Language for agent prompts and responses",
    )

    # ---------------------------------------------------------------
    # COMPUTED FIELDS — Apenas aliases para conveniência
    # ---------------------------------------------------------------
    @computed_field
    @property
    def llm_full_endpoint(self) -> str:
        """Alias for LLMSTUDIO_CHAT_ENDPOINT (backward compatibility)."""
        return self.LLMSTUDIO_CHAT_ENDPOINT

    @computed_field
    @property
    def embedding_full_endpoint(self) -> str:
        """Alias for LLMSTUDIO_EMBEDDING_ENDPOINT (backward compatibility)."""
        return self.LLMSTUDIO_EMBEDDING_ENDPOINT


# ================================================================
# GLOBAL INSTANCE
# ================================================================
llm_settings = LLMSettings()
