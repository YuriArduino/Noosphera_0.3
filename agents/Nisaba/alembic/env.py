"""
Alembic Environment Configuration — Nisaba Agent.

Purpose:
    - Load database connection from environment (secure, no hardcoded creds)
    - Import SSOT models from nisaba.schema.tables
    - Configure Alembic for auto-generate migrations with schema support

Pattern: Mirrors Glyphar's alembic/env.py for consistency.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# =============================================================================
# 1. PATH RESOLUTION
# =============================================================================
# Ensure src/ is in sys.path so we can import nisaba.* modules
# __file__ = /agents/Nisaba/alembic/env.py
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
SRC_DIR = BASE_DIR / "src"
for path in [str(REPO_ROOT), str(SRC_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# =============================================================================
# 2. IMPORT SSOT MODELS
# =============================================================================
# Single Source of Truth for all table definitions
from agents.shared.config.memory import memory_settings  # type: ignore
from nisaba.schema.tables import Base  # type: ignore

# =============================================================================
# 3. DATABASE CONNECTION
# =============================================================================
# Database URL comes from the shared/global config (.env at repository root).
DATABASE_URL = memory_settings.DATABASE_URL

# =============================================================================
# 4. ALEMBIC CONFIGURATION
# =============================================================================
config = context.config

# Load logging config from alembic.ini if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for auto-detecting model changes
target_metadata = Base.metadata

# =============================================================================
# 5. MIGRATION EXECUTION
# =============================================================================


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (no active DB connection).
    Useful for generating SQL scripts without connecting.
    """
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,  # SQLite/Postgres compatibility
        include_schemas=True,  # Detect changes in custom schemas (e.g., 'nisaba')
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (with active DB connection).
    This is the default mode for apply/upgrade operations.
    """
    # Override alembic.ini URL with env var (security best practice)
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = DATABASE_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        pool_pre_ping=True,  # Auto-reconnect on stale connections
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
