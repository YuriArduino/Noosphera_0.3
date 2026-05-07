-- ============================================
-- NOOSPHERA AGENT DATABASE — BASE INFRASTRUCTURE
-- ============================================
-- Purpose: Shared database foundation for all Noosphera Agents
-- Pattern: Mirrors 02-tool-base.sql for architectural consistency
-- Version: 0.3.0 | Noosphera Architecture

-- 1. CREATE SHARED AGENT DATABASE
-- ============================================
CREATE DATABASE "noosphera_agents_db"
    WITH
    OWNER = yuri
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

\c "noosphera_agents_db"

-- 2. INSTALL SHARED EXTENSIONS (DB-level, accessible to all schemas)
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- 3. CREATE AGENT SCHEMAS (Logical Isolation)
-- ============================================
-- Each agent operates in its own schema to avoid table collisions
-- while sharing the same database & extensions
CREATE SCHEMA IF NOT EXISTS nisaba AUTHORIZATION yuri;
-- CREATE SCHEMA IF NOT EXISTS agent_alpha AUTHORIZATION yuri; -- Future agents

-- 4. CONFIGURE SEARCH PATH & PERMISSIONS
-- ============================================
ALTER DATABASE "noosphera_agents_db" SET search_path TO public, nisaba, extensions;
GRANT ALL PRIVILEGES ON DATABASE "noosphera_agents_db" TO yuri;
GRANT USAGE ON SCHEMA nisaba TO yuri;
GRANT CREATE ON SCHEMA nisaba TO yuri;

-- 5. VALIDATION & SUMMARY
-- ============================================
DO $$
BEGIN
    RAISE NOTICE '🧠 Noosphera Agents Database Base Created!';
    RAISE NOTICE '✅ Database: "noosphera_agents_db"';
    RAISE NOTICE '✅ Shared Extensions: vector, uuid-ossp, pg_trgm, btree_gin';
    RAISE NOTICE '✅ Schemas: public (shared), nisaba (ready for tables)';
    RAISE NOTICE '✅ Next: Configure Alembic with schema="nisaba" and run migrations';
END $$;
