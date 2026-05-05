"""Alembic environment configuration for Glyphar Tool."""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# 1. RESOLUÇÃO DE CAMINHO
# __file__ é /tools/Glyphar/alembic/env.py
# O objetivo é adicionar /tools/Glyphar/src ao sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# 2. IMPORTS DO PROJETO:
# Agora o Python consegue encontrar seus módulos
from glyphar.schema.tables import metadata
from glyphar.database.db import DATABASE_URL

# 3. CONFIGURAÇÃO DO ALEMBIC:
config = context.config

# Interpreta o arquivo de configuração para logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Define o SST (Single Source of Truth) para o autogenerate
target_metadata = metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Sobrescrevemos a URL do alembic.ini pela URL injetada via Environment Variable
    # Isso evita expor senhas no arquivo .ini
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = DATABASE_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
