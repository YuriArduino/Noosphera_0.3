#!/bin/sh

set -e

PGADMIN_CONFIG_DB="/var/lib/pgadmin/pgadmin.db"
SERVERS_FILE="/pgadmin4/servers.json"
SETUP_SCRIPT="/pgadmin4/setup.py"

echo "[Noosphera] Checking pgAdmin server configuration..."

if [ -f "$SERVERS_FILE" ]; then

    echo "[Noosphera] servers.json found."

    if [ -f "$PGADMIN_CONFIG_DB" ]; then

        echo "[Noosphera] Existing pgAdmin database detected."
        echo "[Noosphera] Reloading server definitions..."

        python3 "$SETUP_SCRIPT" \
            load-servers "$SERVERS_FILE" \
            --replace \
            --sqlite-path "$PGADMIN_CONFIG_DB"

        echo "[Noosphera] pgAdmin servers successfully reloaded."

    else

        echo "[Noosphera] First startup detected."
        echo "[Noosphera] pgAdmin will auto-import servers.json."

    fi

else

    echo "[Noosphera] No servers.json found."

fi
