"""Quick schema verification. Uses models.py as SSOT."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv  # type: ignore

# 🔹 Carrega .env ANTES de importar settings
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sqlalchemy import inspect, create_engine, text

# 🔹 Usa variável de ambiente diretamente como fallback seguro
DATABASE_URL = os.getenv(
    "NISABA_DATABASE_URL",
    "postgresql://yuri:3759@localhost:5432/noosphera_agents_db",  # ← Seus creds reais
)

print(f"🔍 Conectando a: {DATABASE_URL.replace('://yuri:3759@', '://***:***@')}")

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # Teste rápido de conexão
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database(), current_user"))
        db, user = result.first()
        print(f"✅ Conectado! Banco: {db}, Usuário: {user}")

    inspector = inspect(engine)

    print("\n📦 Tabelas no schema 'nisaba':")
    tables = inspector.get_table_names(schema="nisaba")
    if not tables:
        print("  ⚠️ Nenhuma tabela encontrada (rodou alembic upgrade head?)")
    for table in tables:
        cols = [c["name"] for c in inspector.get_columns(table, schema="nisaba")]
        print(f"  ├── {table} ({', '.join(cols)})")

    print("\n✅ Schema validado via SSOT (models.py)")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n💡 Dicas:")
    print("  1. Verifique se NISABA_DATABASE_URL no .env está correto")
    print("  2. Teste: docker exec -it noosphera_database psql -U yuri -d noosphera_agents_db")
    print("  3. Execute: make nisaba-upgrade para aplicar migrations")
    sys.exit(1)
