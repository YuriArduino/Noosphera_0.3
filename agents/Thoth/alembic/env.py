# agents/Thoth/alembic/env.py
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# 1. Resolução de Caminho (Sobe para agents/Thoth/ e entra em src/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# 2. Imports do Agente
# Importamos o SQLModel que contém todas as tabelas do ThothLedger
from thoth.infrastructure.ledger import SQLModel

# Importamos a URL de conexão (que deve estar no seu .env)
import os

DATABASE_URL = os.environ.get("THOTH_DATABASE_URL")

target_metadata = SQLModel.metadata

# ... o restante das funções run_migrations_online/offline permanecem
# o padrão do Alembic, apenas garantindo que usem a DATABASE_URL acima.
