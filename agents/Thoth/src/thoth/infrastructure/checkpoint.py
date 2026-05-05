# agents/Thoth/src/thoth/infrastructure/checkpoint.py
from langgraph.checkpoint.postgres import PostgresSaver
from thoth.config import settings

# Conecta ao SST do Noosphera
checkpointer = PostgresSaver.from_conn_string(settings.memory.THOTH_DATABASE_URL)
# checkpointer.setup()  # Executado uma única vez para criar tabelas do LangGraph
