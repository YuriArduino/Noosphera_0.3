#!/bin/sh
set -e

# Caminho do banco de configuração do pgAdmin
PGADMIN_CONFIG_DB="/var/lib/pgadmin/pgadmin.db"
# Caminho do script Python de importação
SETUP_SCRIPT="/pgadmin4/setup.py"

echo "🔍 Verificando configuração de servidores do pgAdmin..."

if [ -f "/pgadmin4/servers.json" ]; then
    echo "📂 servers.json encontrado."

    # Se o banco de configuração já existe, força a importação
    if [ -f "$PGADMIN_CONFIG_DB" ]; then
        echo "🔄 Banco de configuração existente. Forçando importação do servers.json..."
        python3 "$SETUP_SCRIPT" load-servers /pgadmin4/servers.json --replace --sqlite-path "$PGADMIN_CONFIG_DB"
        if [ $? -eq 0 ]; then
            echo "✅ Servidores importados com sucesso (modo replace)."
        else
            echo "❌ Erro ao importar servidores."
        fi
    else
        echo "🆕 Primeira inicialização detectada. pgAdmin carregará servers.json automaticamente."
    fi
else
    echo "ℹ️ servers.json não encontrado. Nenhuma configuração automática será feita."
fi

# Executa o entrypoint original do pgAdmin
exec /entrypoint.sh "$@"
