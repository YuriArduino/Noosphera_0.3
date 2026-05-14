# =============================================================================
# NOÖSPHERA — POSTGRESQL 18 AI STACK
# -----------------------------------------------------------------------------
# PostgreSQL 18 + pgvector + pg_search (BM25)
# Optimized for:
#   - AI workloads
#   - Hybrid search
#   - JSONB
#   - Vector embeddings
#   - Semantic retrieval
#
# Base OS:
#   Debian Bookworm
# =============================================================================

FROM postgres:18-bookworm

# =============================================================================
# METADATA
# =============================================================================

LABEL maintainer="yuri@noosphera"
LABEL org.opencontainers.image.title="Noosphera PostgreSQL AI Stack"
LABEL org.opencontainers.image.description="PostgreSQL 18 optimized for AI, pgvector and BM25 search"
LABEL org.opencontainers.image.version="1.0"

ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# PGVECTOR INSTALLATION
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-18-pgvector \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# PARADEDB / PG_SEARCH INSTALLATION
# -----------------------------------------------------------------------------
# Official self-hosted installation using .deb package.
# Avoids the interactive install.sh script.
# =============================================================================

RUN curl -L \
    "https://github.com/paradedb/paradedb/releases/download/v0.23.4/postgresql-18-pg-search_0.23.4-1PARADEDB-bookworm_amd64.deb" \
    -o /tmp/pg_search.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/pg_search.deb \
    && rm /tmp/pg_search.deb \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# POSTGRESQL CONFIGURATION
# =============================================================================

COPY --chown=postgres:postgres \
    config/postgresql.conf \
    /etc/postgresql/postgresql.conf

# =============================================================================
# HEALTHCHECK
# =============================================================================

HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD \
    pg_isready -U "${POSTGRES_USER:-postgres}" -d postgres

# =============================================================================
# SECURITY
# =============================================================================

USER postgres

# =============================================================================
# STARTUP
# =============================================================================

CMD ["postgres", "-c", "config_file=/etc/postgresql/postgresql.conf"]
