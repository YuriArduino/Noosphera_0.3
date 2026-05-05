"""
LLM Integration Configuration — Thoth Agent.

Manages connection settings, model selection, and API endpoints for
Large Language Models (LLM) and Embedding providers.

Designed to interface with OpenAI-compatible APIs (LLM Studio, LM Studio, Ollama).
"""

import os
from typing import Literal
from pydantic import Field, computed_field, ConfigDict

from .base import ThothBaseSettings


class LLMSettings(ThothBaseSettings):
    """
    Configuration for LLM and Embedding integration.

    This class enables Thoth to communicate with local or remote inference
    servers for text correction and semantic indexing.
    """

    # ---------------------------------------------------------------
    # CONNECTION & INFRASTRUCTURE
    # ---------------------------------------------------------------

    # Priority: Env Var 'OPENAI_API_BASE' from Noosphera Compose, then default
    LLM_BASE_URL: str = Field(
        default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:1234/v1"),
        description="Base URL for the OpenAI-compatible API (e.g., LLM Studio/Ollama)",
    )

    LLM_API_KEY: str = Field(
        default=os.environ.get("OPENAI_API_KEY", "sk-local"),
        description="API Key for the LLM provider (use 'sk-local' for local servers)",
    )

    LLM_TIMEOUT: int = Field(
        default=180,
        ge=10,
        le=600,
        description="Request timeout in seconds for long OCR correction tasks",
    )

    LLM_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed LLM requests",
    )

    # ---------------------------------------------------------------
    # CHAT MODEL (Text Correction & Reasoning)
    # ---------------------------------------------------------------

    CHAT_MODEL: str = Field(
        default="meta-llama-3.1-8b-instruct",
        description="Primary model used for OCR text refinement and correction",
    )

    CHAT_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for completions (kept low for high OCR fidelity)",
    )

    CHAT_MAX_TOKENS: int = Field(
        default=8000,
        ge=100,
        le=128000,
        description="Maximum tokens allowed for a single correction response",
    )

    # ---------------------------------------------------------------
    # EMBEDDING MODEL (Semantic Memory)
    # ---------------------------------------------------------------

    EMBEDDING_MODEL: str = Field(
        default="text-embedding-nomic-embed-text-v1.5@q8_0",
        description="Model used to generate vectors for pgvector and memory recall",
    )

    # Matches the dimension defined in memory.py (SST)
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Vector dimensions produced by the chosen embedding model",
    )

    # ---------------------------------------------------------------
    # AGENT LOCALIZATION
    # ---------------------------------------------------------------

    PROMPT_LANGUAGE: Literal["pt-BR", "en-US"] = Field(
        default="pt-BR",
        description="Primary language for Agent system prompts and thought process",
    )

    # ---------------------------------------------------------------
    # COMPUTED ENDPOINTS (Aliases)
    # ---------------------------------------------------------------

    @computed_field
    @property
    def llm_chat_endpoint(self) -> str:
        """Returns the full OpenAI-style chat completions endpoint."""
        return f"{self.LLM_BASE_URL.rstrip('/')}/chat/completions"

    @computed_field
    @property
    def llm_embedding_endpoint(self) -> str:
        """Returns the full OpenAI-style embeddings endpoint."""
        return f"{self.LLM_BASE_URL.rstrip('/')}/embeddings"

    # Backward compatibility aliases
    @computed_field
    @property
    def llm_full_endpoint(self) -> str:
        return self.llm_chat_endpoint

    @computed_field
    @property
    def embedding_full_endpoint(self) -> str:
        return self.llm_embedding_endpoint

    model_config = ConfigDict(frozen=True, extra="ignore")


# ================================================================
# GLOBAL INSTANCE
# ================================================================
llm_settings = LLMSettings()
