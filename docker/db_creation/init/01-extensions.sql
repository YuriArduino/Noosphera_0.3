-- ============================================
-- NOOSPHERA DATABASE EXTENSIONS
-- ============================================

-- Extensões para IA e desenvolvimento
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS hstore;

-- Mensagem de sucesso
DO $$
BEGIN
    RAISE NOTICE '✅ Extensões instaladas com sucesso!';
    RAISE NOTICE '📊 Bancos existentes:';
END $$;

-- Listar extensões instaladas
SELECT
    extname AS "Extensão",
    extversion AS "Versão"
FROM pg_extension
ORDER BY extname;

-- Mostrar bancos atuais
SELECT datname FROM pg_database ORDER BY datname;
