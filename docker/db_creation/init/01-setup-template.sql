-- =============================================================================
-- NOÖSPHERA — GLOBAL CLUSTER DNA
-- -----------------------------------------------------------------------------
-- File:
--   01-setup-template.sql
--
-- Purpose:
--   Defines the foundational PostgreSQL capabilities inherited by all future
--   databases created from template1.
--
-- Architectural Role:
--   This script establishes the cognitive substrate of the Noösphera cluster:
--
--   - Vector memory (pgvector)
--   - Hybrid semantic retrieval (pg_search + BM25)
--   - Fuzzy semantic matching (pg_trgm)
--   - JSONB indexing acceleration
--   - Cryptographic identity
--   - Query observability
--
-- IMPORTANT:
--   Everything installed into template1 propagates to all future databases.
--
-- Requirements:
--   - pgvector installed in PostgreSQL image
--   - pg_search installed in PostgreSQL image
--   - shared_preload_libraries configured correctly
--
-- =============================================================================

\connect template1

-- =============================================================================
-- SAFETY VALIDATION
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Initializing Noösphera cluster template...';
END $$;

-- =============================================================================
-- AI & HYBRID SEARCH
-- =============================================================================

-- pgvector:
-- Vector embeddings + HNSW/IVFFlat ANN search
CREATE EXTENSION IF NOT EXISTS vector
SCHEMA public;

-- pg_search:
-- BM25 hybrid semantic search (ParadeDB)
CREATE EXTENSION IF NOT EXISTS pg_search
SCHEMA public;

-- =============================================================================
-- TEXT ANALYSIS & INDEXING
-- =============================================================================

-- Trigram fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm
SCHEMA public;

-- Optimized indexing for JSONB and compound queries
CREATE EXTENSION IF NOT EXISTS btree_gin
SCHEMA public;

CREATE EXTENSION IF NOT EXISTS btree_gist
SCHEMA public;

-- =============================================================================
-- IDENTITY & CRYPTOGRAPHY
-- =============================================================================

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"
SCHEMA public;

-- Cryptographic functions
CREATE EXTENSION IF NOT EXISTS pgcrypto
SCHEMA public;

-- =============================================================================
-- OBSERVABILITY
-- =============================================================================

-- Query monitoring and profiling
CREATE EXTENSION IF NOT EXISTS pg_stat_statements
SCHEMA public;

-- =============================================================================
-- GLOBAL DATABASE DEFAULTS
-- =============================================================================

-- Distributed-system safe timezone
ALTER DATABASE template1
SET timezone TO 'UTC';

-- Better multilingual semantic handling
ALTER DATABASE template1
SET default_text_search_config TO 'pg_catalog.simple';

-- =============================================================================
-- VALIDATION
-- =============================================================================

DO $$
BEGIN

    RAISE NOTICE '';
    RAISE NOTICE '==============================================';
    RAISE NOTICE ' Noosphera Cluster DNA Initialized';
    RAISE NOTICE '==============================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Capabilities Enabled:';
    RAISE NOTICE '  - pgvector (HNSW/IVFFlat)';
    RAISE NOTICE '  - pg_search (BM25)';
    RAISE NOTICE '  - pg_trgm';
    RAISE NOTICE '  - JSONB indexing';
    RAISE NOTICE '  - UUID + crypto';
    RAISE NOTICE '  - Query observability';
    RAISE NOTICE '';
    RAISE NOTICE 'Template inheritance active.';
    RAISE NOTICE '';

END $$;
