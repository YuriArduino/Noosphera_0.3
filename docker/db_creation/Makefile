include .env
export

.PHONY: help build up down restart logs status shell backup restore clean config

help:
	@echo "Noosphera Database Commands"
	@echo "============================"
	@echo "make build    - Build custom PostgreSQL image"
	@echo "make up       - Start all services"
	@echo "make down     - Stop all services"
	@echo "make restart  - Restart services"
	@echo "make logs     - View logs (all services)"
	@echo "make logs-db  - View PostgreSQL logs only"
	@echo "make status   - Check services status"
	@echo "make shell    - Access psql shell"
	@echo "make backup   - Create full database backup"
	@echo "make restore  - Restore from backup"
	@echo "make config   - Show current configuration"
	@echo "make clean    - Remove everything (WARNING!)"

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
	docker exec -it $(CONTAINER_NAME) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

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
