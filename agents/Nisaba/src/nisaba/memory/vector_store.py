"""
Semantic Memory: pgvector Integration for Similarity Search.
Enables Nisaba to retrieve similar past experiences.
"""

import logging
from typing import List, Optional, Union
from datetime import datetime, timezone

import numpy as np
from sqlmodel import Session, select, create_engine
from sqlalchemy import text

from agents.shared.config.memory import memory_settings
from nisaba.config.llm import llm_settings
from nisaba.schema.tables import SemanticExperienceTable

logger = logging.getLogger(__name__)


def _safe_text_for_embedding(text: Optional[str], fallback: str = "empty") -> str:
    if text is None:
        return fallback
    text = str(text).strip()
    return text if text else fallback


def _to_float_list(vec: Optional[Union[List[float], np.ndarray]]) -> Optional[List[float]]:
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        return vec.tolist()
    if isinstance(vec, list):
        return [float(x) for x in vec]
    return None


def get_embedding(text: str) -> List[float]:
    """Generate embedding using configured provider."""
    from langchain_openai import OpenAIEmbeddings

    text = _safe_text_for_embedding(text)
    embeddings = OpenAIEmbeddings(
        model=llm_settings.EMBEDDING_MODEL,
        openai_api_key=llm_settings.LLM_API_KEY,
        openai_api_base=llm_settings.LLM_BASE_URL,
        dimensions=llm_settings.EMBEDDING_DIMENSION,
        check_embedding_ctx_length=False,
    )
    result = embeddings.embed_query(text)
    return [float(x) for x in result]


class VectorStore:
    """Manages semantic memory with pgvector similarity search. Singleton pattern."""

    _instance = None
    _engine = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.enabled = memory_settings.VECTORSTORE_ENABLED
        self.db_url = memory_settings.DATABASE_URL
        self.dimensions = llm_settings.EMBEDDING_DIMENSION

        if not self.enabled:
            self._initialized = True
            return

        self.engine = create_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self._ensure_vector_extension()
        self._initialized = True

    def _ensure_vector_extension(self):
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
        if not self.enabled or not self.engine:
            return -1

        content = _safe_text_for_embedding(content, fallback="empty_experience")
        if embedding is None:
            embedding = get_embedding(content)

        with Session(self.engine) as session:
            experience = SemanticExperienceTable(
                session_id=session_id,
                title=_safe_text_for_embedding(title, fallback=""),
                content=content,
                category=_safe_text_for_embedding(category, fallback="general"),
                tags=tags or [],
                metadata_json=metadata or {},
                embedding=embedding,  # List[float] → Vector(384) automaticamente
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
        if not self.enabled or not self.engine:
            return []

        query = _safe_text_for_embedding(query)
        if len(query.strip()) < 3:
            return []

        limit = limit or memory_settings.SEMANTIC_SEARCH_TOP_K
        query_embedding = get_embedding(query)

        with Session(self.engine) as session:
            # Usa o método da coluna: embedding.l2_distance(query_embedding)
            stmt = (
                select(SemanticExperienceTable)
                .order_by(SemanticExperienceTable.embedding.l2_distance(query_embedding))
                .limit(limit)
            )

            if category:
                stmt = stmt.where(SemanticExperienceTable.category == category)

            results = session.exec(stmt).all()

            # Filtro opcional por similaridade cosseno (calculada em Python)
            if min_relevance > 0.0 and results:
                filtered = []
                for r in results:
                    emb = _to_float_list(r.embedding)
                    if emb and len(emb) > 0:
                        relevance = self._calculate_relevance(query_embedding, emb)
                        if relevance >= min_relevance:
                            filtered.append(r)
                results = filtered

            return results

    def _calculate_relevance(self, embedding_a: List[float], embedding_b: List[float]) -> float:
        """Cosine similarity between two embeddings. Returns 0.0 if invalid."""
        if not embedding_a or not embedding_b:
            return 0.0
        if len(embedding_a) != len(embedding_b):
            return 0.0

        dot = sum(a * b for a, b in zip(embedding_a, embedding_b))
        norm_a = sum(a * a for a in embedding_a) ** 0.5
        norm_b = sum(b * b for b in embedding_b) ** 0.5

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def increment_usage(self, experience_id: int):
        if not self.enabled or not self.engine:
            return
        with Session(self.engine) as session:
            exp = session.get(SemanticExperienceTable, experience_id)
            if exp:
                exp.usage_count += 1
                exp.updated_at = datetime.now(timezone.utc)
                session.add(exp)
                session.commit()
