"""
Semantic Memory: pgvector Integration for Similarity Search.
Enables Nisaba to retrieve similar past experiences.
"""

import logging
import json
import re
from typing import List, Optional, Union
from datetime import datetime, timezone

import numpy as np
from sqlmodel import Session, create_engine
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


def _safe_text_for_bm25_query(text: Optional[str], fallback: str = "empty") -> str:
    """
    Convert arbitrary chat text into a conservative ParadeDB/Lucene query.

    Raw user turns may contain Cypher snippets, JSON, markdown, quotes, colons,
    and parentheses. Those characters are meaningful to ParadeDB's query parser,
    so the lexical side of hybrid retrieval receives only simple terms joined by
    explicit OR operators. The original text is still used for embeddings.
    """
    text = _safe_text_for_embedding(text, fallback=fallback).lower()
    terms = re.findall(r"[0-9a-zA-ZÀ-ÿ_]{3,}", text)

    stopwords = {
        "aos",
        "com",
        "das",
        "dos",
        "essa",
        "esse",
        "está",
        "esta",
        "para",
        "por",
        "que",
        "sem",
        "uma",
        "você",
        "voce",
    }
    deduped_terms: list[str] = []
    seen = set()
    for term in terms:
        if term in stopwords or term in seen:
            continue
        seen.add(term)
        deduped_terms.append(term)
        if len(deduped_terms) >= 12:
            break

    return " OR ".join(deduped_terms) if deduped_terms else fallback


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
        self.table_schema = self._resolve_experience_schema()
        self._initialized = True

    def _ensure_vector_extension(self):
        if not self.engine:
            return
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

    def _resolve_experience_schema(self) -> str:
        """
        Detect where semantic_experience currently lives.
        Preference order: public -> nisaba.
        """
        if not self.engine:
            return "public"

        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT table_schema
                    FROM information_schema.tables
                    WHERE table_name = 'semantic_experience'
                      AND table_schema IN ('public', 'nisaba')
                    ORDER BY CASE WHEN table_schema = 'public' THEN 0 ELSE 1 END
                    LIMIT 1
                    """
                )
            ).fetchone()
            if result and result[0]:
                return str(result[0])
        return "public"

    def _has_bm25_index(self) -> bool:
        """Return True when semantic_experience has a ParadeDB BM25 index."""
        if not self.engine:
            return False

        schema = self.table_schema if self.table_schema in {"public", "nisaba"} else "public"
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT 1
                    FROM pg_indexes
                    WHERE schemaname = :schema
                      AND tablename = 'semantic_experience'
                      AND indexdef ILIKE '%USING bm25%'
                    LIMIT 1
                    """
                ),
                {"schema": schema},
            ).fetchone()
            return row is not None

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

        schema = self.table_schema if self.table_schema in {"public", "nisaba"} else "public"
        with Session(self.engine) as session:
            stmt = text(
                f"""
                INSERT INTO {schema}.semantic_experience
                    (created_at, updated_at, session_id, content, title, category, tags, metadata_json, embedding, relevance_score, usage_count)
                VALUES
                    (CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :session_id, :content, :title, :category, CAST(:tags AS jsonb), CAST(:metadata_json AS jsonb), CAST(:embedding AS vector), NULL, 0)
                RETURNING id
                """
            ).bindparams(
                session_id=session_id,
                content=content,
                title=_safe_text_for_embedding(title, fallback=""),
                category=_safe_text_for_embedding(category, fallback="general"),
                tags=json.dumps(tags or [], ensure_ascii=False),
                metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
                embedding=embedding,
            )
            row = session.exec(stmt).first()
            session.commit()
            if row is None:
                return -1
            if hasattr(row, "_mapping"):
                mapping = row._mapping
                if "id" in mapping:
                    return int(mapping["id"])
                if len(mapping) > 0:
                    return int(next(iter(mapping.values())))
            if isinstance(row, (list, tuple)):
                return int(row[0])
            return int(row)

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

        schema = self.table_schema if self.table_schema in {"public", "nisaba"} else "public"

        with Session(self.engine) as session:
            stmt = text(
                f"""
                SELECT e.*
                FROM {schema}.semantic_experience e
                WHERE (:category IS NULL OR e.category = :category)
                ORDER BY e.embedding <-> CAST(:embedding AS vector)
                LIMIT :limit
                """
            ).bindparams(
                embedding=query_embedding,
                category=category,
                limit=limit,
            )
            rows = session.exec(stmt).fetchall()
            results = [SemanticExperienceTable(**dict(row._mapping)) for row in rows]

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

    def search_hybrid(
        self,
        query: str,
        limit: int = None,
        category: Optional[str] = None,
    ) -> List[SemanticExperienceTable]:
        """
        Hybrid retrieval using pgvector + BM25 (ParadeDB) fused with RRF.

        Falls back to vector-only search if BM25 extension/index is unavailable.
        """
        if not self.enabled or not self.engine:
            return []

        query = _safe_text_for_embedding(query)
        if len(query.strip()) < 3:
            return []

        limit = limit or memory_settings.SEMANTIC_SEARCH_TOP_K
        query_embedding = get_embedding(query)
        bm25_query = _safe_text_for_bm25_query(query)

        schema = self.table_schema if self.table_schema in {"public", "nisaba"} else "public"

        if not self._has_bm25_index():
            return self.search_similar(query=query, limit=limit, category=category)

        with Session(self.engine) as session:
            stmt = text(
                f"""
                WITH vector_results AS (
                    SELECT id,
                           ROW_NUMBER() OVER (ORDER BY embedding <-> CAST(:embedding AS vector)) AS rank
                    FROM {schema}.semantic_experience
                    WHERE (:category IS NULL OR category = :category)
                    ORDER BY embedding <-> CAST(:embedding AS vector)
                    LIMIT :vec_limit
                ),
                bm25_results AS (
                    SELECT id,
                           ROW_NUMBER() OVER (ORDER BY paradedb.score(id) DESC) AS rank
                    FROM {schema}.semantic_experience
                    WHERE (content @@@ :query OR title @@@ :query OR category @@@ :query)
                      AND (:category IS NULL OR category = :category)
                    ORDER BY paradedb.score(id) DESC
                    LIMIT :bm25_limit
                ),
                rrf_scores AS (
                    SELECT COALESCE(v.id, b.id) AS id,
                           (COALESCE(1.0 / (60 + v.rank), 0.0) +
                            COALESCE(1.0 / (60 + b.rank), 0.0)) AS rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN bm25_results b ON v.id = b.id
                )
                SELECT e.*
                FROM {schema}.semantic_experience e
                JOIN rrf_scores rs ON e.id = rs.id
                ORDER BY rs.rrf_score DESC, e.id DESC
                LIMIT :final_limit
                """
            ).bindparams(
                embedding=query_embedding,
                query=bm25_query,
                category=category,
                vec_limit=limit * 2,
                bm25_limit=limit * 2,
                final_limit=limit,
            )

            try:
                result = session.exec(stmt)
                rows = result.fetchall()
                return [SemanticExperienceTable(**dict(row._mapping)) for row in rows]
            except Exception as exc:
                logger.warning(
                    "Hybrid retrieval unavailable, falling back to vector-only search: %s",
                    exc,
                )
                return self.search_similar(query=query, limit=limit, category=category)

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
        schema = self.table_schema if self.table_schema in {"public", "nisaba"} else "public"
        with Session(self.engine) as session:
            stmt = text(
                f"""
                UPDATE {schema}.semantic_experience
                SET usage_count = COALESCE(usage_count, 0) + 1,
                    updated_at = :updated_at
                WHERE id = :experience_id
                """
            ).bindparams(
                updated_at=datetime.now(timezone.utc),
                experience_id=experience_id,
            )
            session.exec(stmt)
            session.commit()
