"""add paradedb bm25 index

Revision ID: 5c43f7b9e2ab
Revises: a7f288ba602e
Create Date: 2026-05-15 18:20:00

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers
revision: str = '5c43f7b9e2ab'
down_revision: Union[str, None] = 'a7f288ba602e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Install pg_search and create BM25 index on semantic_experience."""
    op.execute(
        """
        DO $$
        BEGIN
            BEGIN
                CREATE SCHEMA IF NOT EXISTS paradedb;
                CREATE EXTENSION IF NOT EXISTS pg_search SCHEMA paradedb;
            EXCEPTION
                WHEN feature_not_supported OR undefined_file OR insufficient_privilege THEN
                    RAISE NOTICE 'pg_search extension unavailable or not permitted on this server. Skipping BM25 setup.';
                    RETURN;
            END;
        END $$;
        """
    )

    op.execute(
        """
        DO $$
        DECLARE
            target_schema text;
            bm25_exists boolean;
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_search') THEN
                RAISE NOTICE 'pg_search extension is not installed in this database. Skipping BM25 index creation.';
                RETURN;
            END IF;

            SELECT table_schema
              INTO target_schema
              FROM information_schema.tables
             WHERE table_name = 'semantic_experience'
               AND table_schema IN ('public', 'nisaba')
             ORDER BY CASE WHEN table_schema = 'public' THEN 0 ELSE 1 END
             LIMIT 1;

            IF target_schema IS NULL THEN
                RAISE NOTICE 'semantic_experience not found in public/nisaba, skipping BM25 index creation';
                RETURN;
            END IF;

            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = target_schema
                  AND tablename = 'semantic_experience'
                  AND indexdef ILIKE '%USING bm25%'
            ) INTO bm25_exists;

            IF bm25_exists THEN
                RAISE NOTICE 'BM25 index already exists on %.semantic_experience', target_schema;
                RETURN;
            END IF;

            EXECUTE format(
                'CREATE INDEX IF NOT EXISTS idx_experience_bm25 ON %I.semantic_experience USING bm25 (id, content, title, category) WITH (key_field=''id'')',
                target_schema
            );
        END $$;
        """
    )


def downgrade() -> None:
    """Drop BM25 index when present in known schemas."""
    op.execute("DROP INDEX IF EXISTS public.idx_experience_bm25")
    op.execute("DROP INDEX IF EXISTS nisaba.idx_experience_bm25")
