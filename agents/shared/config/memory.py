"""
Global Persistence Infrastructure — Noosphera Shared.

Manages the connection to the centralized PostgreSQL backend, controlling
state checkpoints, vector storage, and the cognitive ledger.
"""

from pydantic import Field, ConfigDict, AliasChoices, computed_field
from agents.shared.config.base import SharedBaseSettings


class SharedMemorySettings(SharedBaseSettings):
    """
    Global Database and Persistence infrastructure settings.

    Standardizes how agents connect to the Noosphera SST (Single Source of Truth)
    for both relational and vector data.

    Features:
        - Centralized PostgreSQL connection string management.
        - Granular toggles for Checkpoints, Vectorstores, and Ledgers.
        - Standardized Top-K parameters for semantic retrieval.

    Design rationale:
        - Moving the DATABASE_URL to Shared ensures that if the DB credentials
          change (e.g., in a Docker migration), all agents remain in sync.
        - AliasChoices allow for 'AGENT_'-level fallbacks from the global .env.
    """

    # ---------------------------------------------------------------------------
    # DATABASE CONNECTIVITY
    # ---------------------------------------------------------------------------

    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/noosphera",
        validation_alias=AliasChoices("AGENT_DATABASE_URL", "DATABASE_URL"),
        description="Main connection string for the PostgreSQL/pgvector instance.",
    )

    @computed_field
    @property
    def NISABA_DATABASE_URL(self) -> str:
        """Backward-compatible alias for older Nisaba modules."""
        return self.DATABASE_URL

    # ---------------------------------------------------------------------------
    # FEATURE TOGGLES
    # ---------------------------------------------------------------------------

    MEMORY_ENABLED: bool = Field(
        default=True,
        description="Master switch for the long-term memory system.",
    )

    CHECKPOINT_ENABLED: bool = Field(
        default=True,
        description="Enables LangGraph state persistence for thread recovery.",
    )

    VECTORSTORE_ENABLED: bool = Field(
        default=True,
        description="Enables semantic search capabilities via pgvector.",
    )

    KNOWLEDGE_GRAPH_ENABLED: bool = Field(
        default=False,
        validation_alias=AliasChoices("KNOWLEDGE_GRAPH_ENABLED", "NEO4J_ENABLED"),
        description="Enables relationship retrieval through Neo4j/Cypher.",
    )

    LEDGER_ENABLED: bool = Field(
        default=True,
        description="Enables the historical decision/correction audit log.",
    )

    # ---------------------------------------------------------------------------
    # KNOWLEDGE GRAPH CONNECTIVITY
    # ---------------------------------------------------------------------------

    NEO4J_URI: str = Field(
        default="bolt://localhost:7687",
        validation_alias=AliasChoices("NEO4J_URI", "NEO4J_URL"),
        description="Bolt URI for the Neo4j knowledge graph.",
    )

    NEO4J_USER: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("NEO4J_USER", "NEO4J_USERNAME"),
        description="Neo4j username.",
    )

    NEO4J_PASSWORD: str = Field(
        default="neo4j",
        validation_alias=AliasChoices("NEO4J_PASSWORD", "NEO4J_PASS"),
        description="Neo4j password.",
    )

    # ---------------------------------------------------------------------------
    # RETRIEVAL PARAMETERS
    # ---------------------------------------------------------------------------

    SEMANTIC_SEARCH_TOP_K: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant documents to retrieve in semantic queries.",
    )

    model_config = ConfigDict(frozen=True, extra="ignore")


# Global instance for shared access
memory_settings = SharedMemorySettings()
