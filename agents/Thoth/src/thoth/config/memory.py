"""
Memory and learning configuration for Thoth Agent.

Supports:
    - LangGraph checkpoint persistence (operational state)
    - Separate SQLite Ledger (learning memory)
    - VectorStore embeddings (semantic memory)
"""

from pathlib import Path
from pydantic import Field, field_validator

from .base import ThothBaseSettings, PathMixin


class MemorySettings(ThothBaseSettings, PathMixin):
    """
    Configuration for Thoth hybrid memory system.

    Layers:
        1. Checkpoint DB → Operational graph state (LangGraph)
        2. Ledger DB → Cognitive learning records (decisions/corrections)
        3. VectorStore → Semantic embedding memory
    """

    # ===============================================================
    # GLOBAL MEMORY SWITCH
    # ===============================================================

    MEMORY_ENABLED: bool = Field(
        default=True,
        description="Enable long-term memory system",
    )

    # ===============================================================
    # CHECKPOINTING (Operational Memory - LangGraph)
    # ===============================================================

    CHECKPOINT_ENABLED: bool = Field(
        default=True,
        description="Enable checkpoint persistence for graph state",
    )

    CHECKPOINT_DB_PATH: Path = Field(
        default=Path("./data/checkpoints/thoth_checkpoint.db"),
        description="SQLite path for LangGraph checkpoint persistence",
    )

    @field_validator("CHECKPOINT_DB_PATH", mode="before")
    @classmethod
    def ensure_checkpoint_dir(cls, v):
        """Ensure checkpoint persistence directory exists."""
        path = Path(v) if isinstance(v, (str, Path)) else v
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ===============================================================
    # LEDGER (Cognitive Learning Memory - Separate SQLite)
    # ===============================================================

    LEDGER_ENABLED: bool = Field(
        default=True,
        description="Enable cognitive ledger for decisions and corrections",
    )

    LEDGER_DB_PATH: Path = Field(
        default=Path("./data/ledger/thoth_ledger.db"),
        description="SQLite path for cognitive learning ledger",
    )

    LEDGER_AUTO_MIGRATE: bool = Field(
        default=True,
        description="Automatically create ledger tables if missing",
    )

    @field_validator("LEDGER_DB_PATH", mode="before")
    @classmethod
    def ensure_ledger_dir(cls, v):
        """Ensure ledger persistence directory exists."""
        path = Path(v) if isinstance(v, (str, Path)) else v
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ===============================================================
    # EMBEDDINGS (Semantic Memory)
    # ===============================================================

    EMBEDDING_MODEL: str = Field(
        default="text-embedding-nomic-embed-text-v1.5@q8_0",
        description="Model used for generating embeddings",
    )

    EMBEDDING_COLLECTION_NAME: str = Field(
        default="thoth_semantic_memory",
        description="Collection name for document embeddings",
    )

    EMBEDDING_DIMENSIONS: int = Field(
        default=768,
        description="Embedding vector dimensions",
    )

    # ===============================================================
    # VECTORSTORE BACKEND
    # ===============================================================

    VECTORSTORE_ENABLED: bool = Field(
        default=True,
        description="Enable vectorstore semantic memory",
    )

    VECTORSTORE_TYPE: str = Field(
        default="chromadb",
        description="VectorStore backend: chromadb | faiss",
    )

    VECTORSTORE_MAX_DOCUMENTS: int = Field(
        default=10000,
        description="Maximum documents stored in vector memory",
    )

    MEMORY_PERSISTENCE_PATH: Path = Field(
        default=Path("./data/memory"),
        description="Path for vectorstore persistence",
    )

    @field_validator("MEMORY_PERSISTENCE_PATH", mode="before")
    @classmethod
    def ensure_memory_path(cls, v):
        """Ensure memory persistence directory exists."""
        path = Path(v) if isinstance(v, (str, Path)) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ===============================================================
    # HUMAN-IN-THE-LOOP
    # ===============================================================

    HITL_ENABLED: bool = Field(
        default=True,
        description="Enable Human-in-the-Loop for low confidence cases",
    )

    HITL_THRESHOLD: float = Field(
        default=50.0,
        description="Confidence threshold for HITL intervention",
    )


# ================================================================
# GLOBAL INSTANCE
# ================================================================

memory_settings = MemorySettings()
