include .env
export

# --- Path Configuration ---
# Path to the Glyphar tool relative to this Makefile
GLYPHAR_DIR = ../../tools/Glyphar
# Database URL for host-side operations (Alembic runs on the host, DB runs in Docker)
GLYPHAR_DB_URL = postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/glyphar_db

.PHONY: help build up down restart logs status shell backup restore clean config \
        glyphar-migrate glyphar-upgrade glyphar-history

help:
	@echo "Noosphera Database Commands"
	@echo "============================"
	@echo "make build    - Build custom PostgreSQL image"
	@echo "make up       - Start all services"
	@echo "make down     - Stop all services"
	@echo "make restart  - Restart services"
	@echo "make logs     - View logs (all services)"
	@echo "make logs-db  - View PostgreSQL logs only"
	@echo ""
	@echo "Glyphar Tool Migrations (Alembic):"
	@echo "  make glyphar-migrate m=\"message\" - Generate new migration from models"
	@echo "  make glyphar-upgrade             - Apply all pending migrations to DB"
	@echo "  make glyphar-history             - Show migration history"
	@echo ""
	@echo "make status   - Check services status"
	@echo "make shell    - Access psql shell"
	@echo "make backup   - Create full database backup"
	@echo "make restore  - Restore from backup"
	@echo "make config   - Show current configuration"
	@echo "make clean    - Remove everything (WARNING!)"

# --- Infrastructure Targets ---

build:
	@echo "Building custom PostgreSQL 18 image..."
	docker-compose -f docker-compose.db.yml build --no-cache

up:
	@echo "Starting Noosphera Database..."
	docker-compose -f docker-compose.db.yml up -d
	@echo "✅ PostgreSQL: localhost:$(POSTGRES_PORT)"
	@echo "✅ pgAdmin: http://localhost:$(PGADMIN_PORT)"
	@echo "   Email: $(PGADMIN_EMAIL) / Password: $(PGADMIN_PASSWORD)"

down:
	docker-compose -f docker-compose.db.yml down

restart:
	docker-compose -f docker-compose.db.yml restart

logs:
	docker-compose -f docker-compose.db.yml logs -f

logs-db:
	docker logs -f $(CONTAINER_NAME)

# --- Glyphar Migration Targets ---

glyphar-migrate:
	@echo "Generating Glyphar migration..."
	@if [ -z "$(m)" ]; then echo "❌ Error: Use 'make glyphar-migrate m=\"your message\"'"; exit 1; fi
	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini revision --autogenerate -m "$(m)"

glyphar-upgrade:
	@echo "Applying Glyphar migrations..."
	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini upgrade head
	@echo "✅ Glyphar database schema is up to date."

glyphar-history:
	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini history --verbose

# --- Maintenance Targets --

status:
	@docker-compose -f docker-compose.db.yml ps
	@echo ""
	@echo "Data Directory: $(DATA_DIR)"
	@if [ -d "$(DATA_DIR)" ]; then \
		du -sh $(DATA_DIR); \
	else \
		echo "Directory not found (OK - using Docker volume)"; \
	fi

shell:
	docker exec -it $(CONTAINER_NAME) psql -U $(POSTGRES_USER) -d glyphar_db

backup:
	@mkdir -p ./backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	docker exec $(CONTAINER_NAME) pg_dumpall -U $(POSTGRES_USER) > ./backups/backup_$$timestamp.sql; \
	echo "✅ Backup created: ./backups/backup_$$timestamp.sql"

restore:
	@read -p "Backup file path: " file; \
	docker exec -i $(CONTAINER_NAME) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) < $$file; \
	echo "✅ Restore completed from $$file"

config:
	@echo "Current Configuration:"
	@echo "======================"
	@echo "PostgreSQL User:     $(POSTGRES_USER)"
	@echo "PostgreSQL Database: $(POSTGRES_DB)"
	@echo "PostgreSQL Port:     $(POSTGRES_PORT)"
	@echo "pgAdmin Port:        $(PGADMIN_PORT)"
	@echo "Data Directory:      $(DATA_DIR)"
	@echo "Shared Buffers:      $(PG_SHARED_BUFFERS)"
	@echo "Work Memory:         $(PG_WORK_MEM)"

clean:
	@echo "⚠️  WARNING: This will delete ALL data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		docker-compose -f docker-compose.db.yml down -v; \
		rm -rf ./backups/*.sql; \
		echo "✅ Cleanup complete."; \
	else \
		echo "❌ Cancelled."; \
	fi
