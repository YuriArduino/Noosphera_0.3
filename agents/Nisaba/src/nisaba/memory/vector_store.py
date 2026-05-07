"""
Semantic Memory: pgvector Integration for Similarity Search.
Enables Nisaba to retrieve similar past experiences.
"""

from typing import List, Optional
from datetime import datetime, timezone
from sqlmodel import Session, select, create_engine, update
from sqlalchemy import text
from nisaba.config.memory import memory_settings
from nisaba.config.llm import llm_settings
from nisaba.schema.tables import SemanticExperienceTable


# Lazy import to avoid circular dependencies
def get_embedding(text: str) -> List[float]:
    """Generate embedding using configured provider."""
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model=llm_settings.EMBEDDING_MODEL,
        openai_api_key=llm_settings.LLM_API_KEY,
        openai_api_base=llm_settings.LLM_BASE_URL,
        dimensions=llm_settings.EMBEDDING_DIMENSION,
    )
    return embeddings.embed_query(text)


class VectorStore:
    """Manages semantic memory with pgvector similarity search."""

    def __init__(self):
        self.enabled = memory_settings.VECTORSTORE_ENABLED
        self.db_url = memory_settings.NISABA_DATABASE_URL
        self.dimensions = llm_settings.EMBEDDING_DIMENSION

        if not self.enabled:
            self.engine = None
            return

        self.engine = create_engine(self.db_url, pool_pre_ping=True)
        self._ensure_vector_extension()

    def _ensure_vector_extension(self):
        """Ensure pgvector extension is installed."""
        if not self.engine:
            return
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    def add_experience(
        self,
        content: str,
        session_id: str,
        title: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Add a semantic experience with optional auto-embedding."""
        if not self.enabled or not self.engine:
            return -1

        # Generate embedding if not provided
        if embedding is None:
            embedding = get_embedding(content)

        with Session(self.engine) as session:
            experience = SemanticExperienceTable(
                session_id=session_id,
                title=title,
                content=content,
                category=category,
                tags=tags or [],
                metadata_json=metadata or {},
                embedding=embedding,
            )
            session.add(experience)
            session.commit()
            return experience.id

    def search_similar(
        self,
        query: str,
        limit: int = None,
        category: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> List[SemanticExperienceTable]:
        """
        Search for similar experiences using vector similarity.
        Returns results ordered by relevance (L2 distance).
        """
        if not self.enabled or not self.engine:
            return []

        limit = limit or memory_settings.SEMANTIC_SEARCH_TOP_K
        query_embedding = get_embedding(query)

        with Session(self.engine) as session:
            # Build query with optional filters
            stmt = (
                select(SemanticExperienceTable)
                .order_by(SemanticExperienceTable.embedding.l2_distance(query_embedding))
                .limit(limit)
            )

            if category:
                stmt = stmt.where(SemanticExperienceTable.category == category)

            results = session.exec(stmt).all()

            # Filter by relevance if needed (cosine similarity approximation)
            if min_relevance > 0 and results:
                filtered = []
                for r in results:
                    if r.embedding:
                        relevance = self._calculate_relevance(query_embedding, r.embedding)
                        if relevance >= min_relevance:
                            filtered.append(r)
                results = filtered

            return results

    def _calculate_relevance(self, embedding_a: List[float], embedding_b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding_a or not embedding_b or len(embedding_a) != len(embedding_b):
            return 0.0
        dot = sum(a * b for a, b in zip(embedding_a, embedding_b))
        norm_a = sum(a * a for a in embedding_a) ** 0.5
        norm_b = sum(b * b for b in embedding_b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def increment_usage(self, experience_id: int):
        """Track usage of an experience for relevance ranking."""
        if not self.enabled or not self.engine:
            return
        with Session(self.engine) as session:
            exp = session.get(SemanticExperienceTable, experience_id)
            if exp:
                exp.usage_count += 1
                exp.updated_at = datetime.now(timezone.utc)
                session.add(exp)
                session.commit()
