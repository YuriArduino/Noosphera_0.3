#!/bin/bash

# =============================================================================
# NOÖSPHERA — AGENT & SEMANTIC TOPOLOGY INITIALIZATION
# -----------------------------------------------------------------------------
# File:
#   03-setup-agents.sh
#
# Purpose:
#   Initializes the semantic, cognitive, and operational topology of the
#   Noösphera ecosystem.
#
# Architectural Model:
#
#   DATABASE  → Ontological Domain
#   SCHEMA    → Cognitive Persona / Functional Layer
#
# Domains:
#   - noosphera_agents_db → Cognitive substrate
#   - nomos_db            → Semantic retrieval
#   - glyphar_db          → Document ingestion
#   - lyra_db             → Audio cognition
#   - arkhe_db            → Provenance & lineage
#   - prefect_db          → Temporal orchestration
#
# =============================================================================

set -e

echo ""
echo "=========================================================="
echo " Noosphera Semantic Topology Initialization"
echo "=========================================================="
echo ""

# =============================================================================
# [1/6] AGENT HUB — COGNITIVE SUBSTRATE
# =============================================================================

echo "[1/6] Initializing cognitive agent substrate..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "noosphera_agents_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_search;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- CORE COGNITIVE PERSONAS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS nisaba;
    CREATE SCHEMA IF NOT EXISTS thoth;
    CREATE SCHEMA IF NOT EXISTS euterpe;
    CREATE SCHEMA IF NOT EXISTS hermes;
    CREATE SCHEMA IF NOT EXISTS janus;
    CREATE SCHEMA IF NOT EXISTS eris;

    -- =========================================================================
    -- SHARED COGNITIVE LAYERS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS shared_memory;
    CREATE SCHEMA IF NOT EXISTS embeddings;
    CREATE SCHEMA IF NOT EXISTS event_bus;
    CREATE SCHEMA IF NOT EXISTS audit;

    -- =========================================================================
    -- OWNERSHIP
    -- =========================================================================

    ALTER SCHEMA nisaba OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA thoth OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA euterpe OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA hermes OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA janus OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA eris OWNER TO "$POSTGRES_USER";

    ALTER SCHEMA shared_memory OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA embeddings OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA event_bus OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA audit OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- SCHEMA DOCUMENTATION
    -- =========================================================================

    COMMENT ON SCHEMA nisaba IS
    'Central cognitive orchestration layer.';

    COMMENT ON SCHEMA thoth IS
    'Extraction and symbolic ingestion layer.';

    COMMENT ON SCHEMA euterpe IS
    'Audio cognition and multimodal interpretation layer.';

    COMMENT ON SCHEMA hermes IS
    'Semantic routing and retrieval coordination layer.';

    COMMENT ON SCHEMA janus IS
    'Temporal and provenance mediation layer.';

    COMMENT ON SCHEMA eris IS
    'Audit, perturbation, and anomaly tracking layer.';

    COMMENT ON SCHEMA shared_memory IS
    'Shared associative memory substrate between agents.';

    COMMENT ON SCHEMA embeddings IS
    'Central vector embedding repository.';

    COMMENT ON SCHEMA event_bus IS
    'Distributed event communication layer.';

    COMMENT ON SCHEMA audit IS
    'System-wide audit and observability layer.';

    -- =========================================================================
    -- DATABASE SEARCH PATH
    -- =========================================================================

    ALTER DATABASE noosphera_agents_db
    SET search_path TO nisaba, shared_memory, public;

EOSQL

# =============================================================================
# [2/6] NOMOS — SEMANTIC INDEXING DOMAIN
# =============================================================================

echo "[2/6] Initializing semantic retrieval domain..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "nomos_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_search;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- SEMANTIC LAYERS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS semantic_index;
    CREATE SCHEMA IF NOT EXISTS retrieval;
    CREATE SCHEMA IF NOT EXISTS ranking;

    -- =========================================================================
    -- OWNERSHIP
    -- =========================================================================

    ALTER SCHEMA semantic_index OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA retrieval OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA ranking OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- DOCUMENTATION
    -- =========================================================================

    COMMENT ON SCHEMA semantic_index IS
    'Hybrid semantic indexing layer.';

    COMMENT ON SCHEMA retrieval IS
    'Retrieval orchestration and semantic querying.';

    COMMENT ON SCHEMA ranking IS
    'Ranking, scoring, and semantic relevance evaluation.';

    -- =========================================================================
    -- SEARCH PATH
    -- =========================================================================

    ALTER DATABASE nomos_db
    SET search_path TO semantic_index, retrieval, public;

EOSQL

# =============================================================================
# [3/6] GLYPHAR — DOCUMENT INGESTION DOMAIN
# =============================================================================

echo "[3/6] Initializing document ingestion domain..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "glyphar_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- INGESTION LAYERS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS processor;
    CREATE SCHEMA IF NOT EXISTS staging;
    CREATE SCHEMA IF NOT EXISTS extraction;
    CREATE SCHEMA IF NOT EXISTS ocr;
    CREATE SCHEMA IF NOT EXISTS metadata;

    -- =========================================================================
    -- OWNERSHIP
    -- =========================================================================

    ALTER SCHEMA processor OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA staging OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA extraction OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA ocr OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA metadata OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- DOCUMENTATION
    -- =========================================================================

    COMMENT ON SCHEMA processor IS
    'Operational processing pipelines.';

    COMMENT ON SCHEMA staging IS
    'Raw ingestion and transient document staging.';

    COMMENT ON SCHEMA extraction IS
    'Structured extraction and normalization layer.';

    COMMENT ON SCHEMA ocr IS
    'OCR intermediary processing layer.';

    COMMENT ON SCHEMA metadata IS
    'Document provenance and metadata storage.';

    -- =========================================================================
    -- SEARCH PATH
    -- =========================================================================

    ALTER DATABASE glyphar_db
    SET search_path TO processor, extraction, public;

EOSQL

# =============================================================================
# [4/6] LYRA — AUDIO COGNITION DOMAIN
# =============================================================================

echo "[4/6] Initializing audio cognition domain..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "lyra_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- AUDIO COGNITIVE LAYERS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS signal;
    CREATE SCHEMA IF NOT EXISTS transcriptions;
    CREATE SCHEMA IF NOT EXISTS diarization;
    CREATE SCHEMA IF NOT EXISTS audio_embeddings;
    CREATE SCHEMA IF NOT EXISTS prosody;

    -- =========================================================================
    -- OWNERSHIP
    -- =========================================================================

    ALTER SCHEMA signal OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA transcriptions OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA diarization OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA audio_embeddings OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA prosody OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- DOCUMENTATION
    -- =========================================================================

    COMMENT ON SCHEMA signal IS
    'Raw audio signal processing layer.';

    COMMENT ON SCHEMA transcriptions IS
    'Speech-to-text transcription layer.';

    COMMENT ON SCHEMA diarization IS
    'Speaker segmentation and identity layer.';

    COMMENT ON SCHEMA audio_embeddings IS
    'Acoustic and multimodal embedding storage.';

    COMMENT ON SCHEMA prosody IS
    'Prosodic, affective, and temporal speech analysis.';

    -- =========================================================================
    -- SEARCH PATH
    -- =========================================================================

    ALTER DATABASE lyra_db
    SET search_path TO signal, transcriptions, public;

EOSQL

# =============================================================================
# [5/6] ARKHE — PROVENANCE & LINEAGE DOMAIN
# =============================================================================

echo "[5/6] Initializing provenance domain..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "arkhe_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- PROVENANCE LAYERS
    -- =========================================================================

    CREATE SCHEMA IF NOT EXISTS provenance;
    CREATE SCHEMA IF NOT EXISTS lineage;
    CREATE SCHEMA IF NOT EXISTS temporal_memory;

    -- =========================================================================
    -- OWNERSHIP
    -- =========================================================================

    ALTER SCHEMA provenance OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA lineage OWNER TO "$POSTGRES_USER";
    ALTER SCHEMA temporal_memory OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- DOCUMENTATION
    -- =========================================================================

    COMMENT ON SCHEMA provenance IS
    'Entity provenance and source traceability layer.';

    COMMENT ON SCHEMA lineage IS
    'Transformation lineage and derivation tracking.';

    COMMENT ON SCHEMA temporal_memory IS
    'Temporal persistence and historical cognition layer.';

    -- =========================================================================
    -- SEARCH PATH
    -- =========================================================================

    ALTER DATABASE arkhe_db
    SET search_path TO provenance, lineage, public;

EOSQL

# =============================================================================
# [6/6] PREFECT — TEMPORAL ORCHESTRATION DOMAIN
# =============================================================================

echo "[6/6] Initializing orchestration domain..."

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "prefect_db" <<-EOSQL

    -- =========================================================================
    -- EXTENSIONS
    -- =========================================================================

    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    -- =========================================================================
    -- PERMISSIONS
    -- =========================================================================

    GRANT ALL PRIVILEGES ON DATABASE prefect_db TO "$POSTGRES_USER";

    GRANT ALL PRIVILEGES ON SCHEMA public TO "$POSTGRES_USER";

    ALTER SCHEMA public OWNER TO "$POSTGRES_USER";

    -- =========================================================================
    -- DEFAULT PRIVILEGES
    -- =========================================================================

    ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON TABLES TO "$POSTGRES_USER";

    ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON SEQUENCES TO "$POSTGRES_USER";

    ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT ALL ON FUNCTIONS TO "$POSTGRES_USER";

EOSQL

# =============================================================================
# FINALIZATION
# =============================================================================

echo ""
echo "=========================================================="
echo " Noosphera Semantic Topology Initialized"
echo "=========================================================="
echo ""
echo "Domains Initialized:"
echo ""
echo "  [1] noosphera_agents_db"
echo "      - Cognitive substrate"
echo "      - Shared memory"
echo "      - Event bus"
echo ""
echo "  [2] nomos_db"
echo "      - Semantic retrieval"
echo "      - Hybrid indexing"
echo ""
echo "  [3] glyphar_db"
echo "      - OCR"
echo "      - Symbolic ingestion"
echo ""
echo "  [4] lyra_db"
echo "      - Audio cognition"
echo "      - Prosodic analysis"
echo ""
echo "  [5] arkhe_db"
echo "      - Provenance"
echo "      - Temporal lineage"
echo ""
echo "  [6] prefect_db"
echo "      - Workflow orchestration"
echo "      - Temporal execution"
echo ""
echo "Initialization completed successfully."
echo ""
