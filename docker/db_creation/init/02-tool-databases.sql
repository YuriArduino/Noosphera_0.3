-- ============================================
-- NOOSPHERA TOOL-SPECIFIC DATABASES
-- ============================================

-- 1. Glyphar (OCR Tool)
-- Create a dedicated database for OCR metadata and processing state.
CREATE DATABASE glyphar_db;

-- 2. Audio (Future Tool)
-- Reserved for audio transcription and embeddings.
-- CREATE DATABASE audio_db;

-- ============================================
-- PER-DATABASE EXTENSION SETUP
-- ============================================

-- Switch to Glyphar Database to install required extensions
\c glyphar_db

-- pgvector is essential for future OCR-to-Vector integrations
CREATE EXTENSION IF NOT EXISTS vector;
-- uuid-ossp for unique resource identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- pg_trgm for fuzzy text searching within OCR results
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Summary Message
DO $$
BEGIN
    RAISE NOTICE '✅ Tool databases and specific extensions created successfully!';
END $$;
