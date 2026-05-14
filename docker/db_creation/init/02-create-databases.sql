-- =============================================================================
-- NOÖSPHERA — PHYSICAL DATABASE PROVISIONING
-- -----------------------------------------------------------------------------
-- File:
--   02-create-databases.sql
--
-- Purpose:
--   Creates the physical databases composing the Noösphera ecosystem.
--
-- Architectural Model:
--
--   DATABASE = Ontological Domain
--
--   Each database represents an independent cognitive or operational domain.
--
-- =============================================================================

-- =============================================================================
-- PREFECT — ORCHESTRATION LAYER
-- =============================================================================

SELECT 'CREATE DATABASE prefect_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'prefect_db'
)\gexec

COMMENT ON DATABASE prefect_db IS
'Workflow orchestration and temporal execution layer.';

-- =============================================================================
-- AGENT HUB — COGNITIVE CORE
-- =============================================================================

SELECT 'CREATE DATABASE noosphera_agents_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'noosphera_agents_db'
)\gexec

COMMENT ON DATABASE noosphera_agents_db IS
'Central cognitive substrate for distributed AI agents.';

-- =============================================================================
-- GLYPHAR — OCR & DOCUMENT EXTRACTION
-- =============================================================================

SELECT 'CREATE DATABASE glyphar_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'glyphar_db'
)\gexec

COMMENT ON DATABASE glyphar_db IS
'Document extraction, OCR, and symbolic ingestion domain.';

-- =============================================================================
-- LYRA — AUDIO & SIGNAL PROCESSING
-- =============================================================================

SELECT 'CREATE DATABASE lyra_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'lyra_db'
)\gexec

COMMENT ON DATABASE lyra_db IS
'Audio cognition and multimodal signal processing domain.';

-- =============================================================================
-- NOMOS — SEMANTIC INDEXING
-- =============================================================================

SELECT 'CREATE DATABASE nomos_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'nomos_db'
)\gexec

COMMENT ON DATABASE nomos_db IS
'Semantic indexing, retrieval, and conceptual navigation domain.';

-- =============================================================================
-- ARKHE — PROVENANCE & MEMORY TRACEABILITY
-- =============================================================================

SELECT 'CREATE DATABASE arkhe_db TEMPLATE template1'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'arkhe_db'
)\gexec

COMMENT ON DATABASE arkhe_db IS
'Provenance, lineage tracking, and memory traceability domain.';

-- =============================================================================
-- VALIDATION
-- =============================================================================

DO $$
BEGIN

    RAISE NOTICE '';
    RAISE NOTICE '==============================================';
    RAISE NOTICE ' Noosphera Physical Databases Created';
    RAISE NOTICE '==============================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Domains:';
    RAISE NOTICE '  - prefect_db';
    RAISE NOTICE '  - noosphera_agents_db';
    RAISE NOTICE '  - glyphar_db';
    RAISE NOTICE '  - lyra_db';
    RAISE NOTICE '  - nomos_db';
    RAISE NOTICE '  - arkhe_db';
    RAISE NOTICE '';

END $$;
