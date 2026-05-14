"""
Global LLM Runtime Configuration — Noosphera Shared.

Manages connection parameters, model selection, and endpoint generation
for OpenAI-compatible providers (Ollama, LM Studio, vLLM).
"""

from typing import Literal
from pydantic import Field, computed_field, ConfigDict, AliasChoices
from .base import SharedBaseSettings


class SharedLLMSettings(SharedBaseSettings):
    """
    Core LLM and Embedding infrastructure settings.

    This class defines the shared connectivity and model defaults used
    across the entire Noosphera ecosystem.

    Features:
        - Support for OpenAI-compatible API specifications.
        - Unified embedding dimension management to prevent vector store collisions.
        - Automatic endpoint construction via computed fields.
        - Fallback support for global 'AGENT_' environment variables.

    Design rationale:
        - Centralizing EMBEDDING_DIMENSION is critical; if one agent indexes
          at 384 and another at 768, the pgvector database will reject the query.
        - AliasChoices allow for a 'Global Fallback' (e.g., AGENT_LLM_BASE_URL)
          if a service-specific override is not provided.
    """

    # ---------------------------------------------------------------------------
    # INFRASTRUCTURE & CONNECTIVITY
    # ---------------------------------------------------------------------------

    LLM_BASE_URL: str = Field(
        default="http://127.0.0.1:1234/v1",
        validation_alias=AliasChoices("AGENT_LLM_BASE_URL", "LLM_BASE_URL"),
        description="Base URL for the inference server.",
    )

    LLM_API_KEY: str = Field(
        default="sk-local",
        validation_alias=AliasChoices("AGENT_LLM_API_KEY", "LLM_API_KEY"),
        description="API key for authentication (use 'sk-local' for local dev).",
    )

    LLM_TIMEOUT: int = Field(
        default=180,
        description="Request timeout in seconds for long reasoning chains.",
    )

    LLM_MAX_RETRIES: int = Field(
        default=3,
        description="Number of retry attempts for failed API calls.",
    )

    # ---------------------------------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------------------------------

    CHAT_MODEL: str = Field(
        default="nvidia/nemotron-3-nano-4b",
        validation_alias=AliasChoices("AGENT_CHAT_MODEL", "CHAT_MODEL"),
        description="The primary model for agentic reasoning and dialogue.",
    )

    EMBEDDING_MODEL: str = Field(
        default="text-embedding-all-minilm-l6-v2-embedding",
        validation_alias=AliasChoices("SHARED_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
        description="The model used for semantic vector generation.",
    )

    EMBEDDING_DIMENSION: int = Field(
        default=384,  # Matches all-minilm-l6-v2
        description="The fixed vector size for the pgvector database.",
    )

    # ---------------------------------------------------------------------------
    # LANGUAGE & LOCALIZATION
    # ---------------------------------------------------------------------------

    PROMPT_LANGUAGE: Literal["pt-BR", "en-US"] = Field(
        default="pt-BR",
        description="Standard language for system prompts and reasoning logs.",
    )

    # ---------------------------------------------------------------------------
    # COMPUTED ENDPOINTS
    # ---------------------------------------------------------------------------

    @computed_field
    @property
    def llm_chat_endpoint(self) -> str:
        """Constructs the full OpenAI-style chat completions endpoint."""
        return f"{self.LLM_BASE_URL.rstrip('/')}/chat/completions"

    @computed_field
    @property
    def llm_embedding_endpoint(self) -> str:
        """Constructs the full OpenAI-style embeddings endpoint."""
        return f"{self.LLM_BASE_URL.rstrip('/')}/embeddings"

    model_config = ConfigDict(frozen=True, extra="ignore")


# Global instance for shared access
llm_settings = SharedLLMSettings()
