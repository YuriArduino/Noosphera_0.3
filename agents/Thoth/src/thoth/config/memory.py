"""
Memory and learning configuration for Thoth Agent.

LangMem, VectorStore, and checkpoint persistence settings.
"""

from pathlib import Path
from pydantic import Field, field_validator

from .base import ThothBaseSettings, PathMixin


class MemorySettings(ThothBaseSettings, PathMixin):
    """
    Configuration for agent memory and learning.

    Supports:
        - LangMem for conversation memory
        - ChromaDB/FAISS for vector embeddings
        - SQLite for checkpoint persistence

    Example:
        >>> from thoth.config import memory_settings
        >>> if memory_settings.MEMORY_ENABLED:
        ...     store = ChromaDB(memory_settings.MEMORY_PERSISTENCE_PATH)
    """

    # ---------------------------------------------------------------
    # MEMORY ENABLED
    # ---------------------------------------------------------------
    MEMORY_ENABLED: bool = Field(
        default=True,
        description="Enable long-term memory with vectorstore",
    )

    MEMORY_PERSISTENCE_PATH: Path = Field(
        default=Path("./data/memory"),
        description="Path for memory persistence",
    )

    @field_validator("MEMORY_PERSISTENCE_PATH", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """Ensure persistence path is a Path object."""
        return Path(v) if isinstance(v, (str, Path)) else v

    # ---------------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------------
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-nomic-embed-text-v1.5@q8_0",
        description="Model for generating embeddings",
    )

    EMBEDDING_COLLECTION_NAME: str = Field(
        default="thoth_documents",
        description="ChromaDB collection name for document embeddings",
    )

    # ---------------------------------------------------------------
    # CHECKPOINTING (LangGraph)
    # ---------------------------------------------------------------
    CHECKPOINT_DB_PATH: Path = Field(
        default=Path("./data/checkpoints/thoth.db"),
        description="SQLite path for LangGraph checkpoint persistence",
    )

    CHECKPOINT_ENABLED: bool = Field(
        default=True,
        description="Enable checkpoint persistence for graph state",
    )

    @field_validator("CHECKPOINT_DB_PATH", mode="before")
    @classmethod
    def ensure_checkpoint_dir(cls, v):
        """Ensure checkpoint directory exists."""
        path = Path(v) if isinstance(v, (str, Path)) else v
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------------
    # HUMAN-IN-THE-LOOP
    # ---------------------------------------------------------------
    HITL_ENABLED: bool = Field(
        default=True,
        description="Enable Human-in-the-Loop for edge cases",
    )

    HITL_THRESHOLD: float = Field(
        default=50.0,
        description="Confidence threshold for HITL intervention",
    )

    # ---------------------------------------------------------------
    # VECTORSTORE
    # ---------------------------------------------------------------
    VECTORSTORE_TYPE: str = Field(
        default="chromadb",
        description="VectorStore backend: chromadb | faiss",
    )

    VECTORSTORE_MAX_DOCUMENTS: int = Field(
        default=10000,
        description="Maximum documents in vector store",
    )


# ================================================================
# GLOBAL INSTANCE
# ================================================================
memory_settings = MemorySettings()
