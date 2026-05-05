"""
Memory and Learning Configuration — Thoth Agent.

Configures the hybrid memory architecture:
    1. Operational State: LangGraph Checkpoints (PostgreSQL).
    2. Cognitive Ledger: Historical decisions and corrections (PostgreSQL).
    3. Semantic Memory: Vectorized insights and patterns (pgvector).
    4. Reflection Layer: LangMem background consolidation.
"""

import os
from pathlib import Path
from pydantic import Field, ConfigDict
from .base import ThothBaseSettings, PathMixin


class MemorySettings(ThothBaseSettings, PathMixin):
    """
    Configuration for Thoth's multi-layered persistent memory system.
    Transitioned from local SQLite/files to centralized PostgreSQL (SST).
    """

    # ===============================================================
    # PERSISTENCE BACKEND (PostgreSQL SST)
    # ===============================================================

    # URL pointing to the Thoth-specific database (thoth_db)
    THOTH_DATABASE_URL: str = Field(
        default=os.environ.get(
            "THOTH_DATABASE_URL", "postgresql://yuri:3759@localhost:5432/thoth_db"
        ),
        description="Main connection string for Agent state and memory",
    )

    MEMORY_ENABLED: bool = Field(
        default=True,
        description="Master switch for the long-term memory system",
    )

    # ===============================================================
    # OPERATIONAL MEMORY (LangGraph Checkpoints)
    # ===============================================================

    CHECKPOINT_ENABLED: bool = Field(
        default=True,
        description="Enable persistence for LangGraph state transitions",
    )

    # ===============================================================
    # COGNITIVE LEDGER (Audit & Learning)
    # ===============================================================

    LEDGER_ENABLED: bool = Field(
        default=True,
        description="Enable logging of every strategic decision and correction",
    )

    LEDGER_AUTO_MIGRATE: bool = Field(
        default=True,
        description="Automatically initialize SQLModel tables on startup",
    )

    # ===============================================================
    # SEMANTIC MEMORY (pgvector)
    # ===============================================================

    EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text-v1.5",
        description="Model identifier for generating semantic vectors",
    )

    EMBEDDING_PROVIDER: str = Field(
        default="ollama",  # or 'openai', 'anthropic'
        description="LLM provider for the embedding model",
    )

    EMBEDDING_DIMENSIONS: int = Field(
        default=768,  # Nomic standard. Use 1536 for OpenAI.
        description="Vector dimensions in pgvector",
    )

    VECTORSTORE_ENABLED: bool = Field(
        default=True,
        description="Enable vectorized experience retrieval",
    )

    # ===============================================================
    # REFLECTION LAYER (LangMem)
    # ===============================================================

    REFLECTION_MODEL: str = Field(
        default="claude-3-5-sonnet-latest",
        description="Reasoning model used to extract insights from trajectories",
    )

    REFLECTION_PROVIDER: str = Field(
        default="anthropic", description="Provider for the reflection reasoning engine"
    )

    # ===============================================================
    # OPERATIONAL PARAMETERS
    # ===============================================================

    HITL_THRESHOLD: float = Field(
        default=50.0,
        description="Confidence threshold that forces Human-in-the-Loop intervention",
    )

    MEMORY_WINDOW_SIZE: int = Field(
        default=10,
        description="Number of recent experiences to keep in immediate reasoning context",
    )

    SEMANTIC_SEARCH_TOP_K: int = Field(
        default=3,
        description="Number of similar cases to retrieve during strategy selection",
    )

    # ===============================================================
    # LEGACY / LOCAL PATHS (Optional Logging)
    # ===============================================================

    # Useful for local debugging or plain-text logs
    LOCAL_DATA_DIR: Path = Field(
        default=Path("./data"), description="Base path for local logs and cache"
    )

    model_config = ConfigDict(frozen=True)


# ================================================================
# GLOBAL INSTANCE
# ================================================================
memory_settings = MemorySettings()
