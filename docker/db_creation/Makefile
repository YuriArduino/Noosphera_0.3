# =============================================================================
# NOÖSPHERA — Infrastructure Control Interface
# -----------------------------------------------------------------------------
# Purpose:
#   Unified operational interface for:
#   - Docker infrastructure lifecycle
#   - PostgreSQL administration
#   - Alembic migrations
#   - Agent database management
#   - Backup & recovery
#   - Development workflows
#
# Architecture:
#   - PostgreSQL 18 + pgvector
#   - Neo4j semantic graph layer
#   - Prefect orchestration
#   - pgAdmin auditing
#   - Multi-agent persistence domains
#
# Recommended:
#   GNU Make 4+
#   Docker Compose V2
#
# Author:
#   Project Noösphera
# =============================================================================

# =============================================================================
# ENVIRONMENT
# =============================================================================

include .env
export

.DEFAULT_GOAL := help

# =============================================================================
# COMPOSE CONFIGURATION
# =============================================================================

COMPOSE_FILE = docker-compose.db.yml
DOCKER_COMPOSE = docker compose -f $(COMPOSE_FILE)

# =============================================================================
# PROJECT PATHS
# =============================================================================

# -----------------------------------------------------------------------------
# Tool Modules
# -----------------------------------------------------------------------------

GLYPHAR_DIR = ../../tools/Glyphar
NISABA_DIR  = ../../agents/Nisaba
NISABA_ALEMBIC = ../../Venvs/nisaba/bin/alembic

# =============================================================================
# DATABASE URLS
# =============================================================================

# -----------------------------------------------------------------------------
# Host-side connections
# Used by Alembic running outside Docker
# -----------------------------------------------------------------------------

GLYPHAR_DB_URL = postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/glyphar_db

NISABA_DB_URL = postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/noosphera_agents_db

# =============================================================================
# BACKUP CONFIGURATION
# =============================================================================

BACKUP_DIR = ./backups
TIMESTAMP  = $(shell date +%Y%m%d_%H%M%S)

# =============================================================================
# PHONY TARGETS
# =============================================================================

.PHONY: \
	help \
	infra-build \
	infra-up \
	infra-down \
	infra-restart \
	infra-rebuild \
	infra-logs \
	infra-logs-db \
	infra-status \
	infra-ps \
	infra-config \
	infra-clean \
	wait-db \
	db-shell \
	db-shell-glyphar \
	db-shell-nisaba \
	db-backup \
	db-backup-all \
	db-restore \
	metrics \
	bootstrap \
	glyphar-migrate \
	glyphar-upgrade \
	glyphar-history \
	nisaba-migrate \
	nisaba-upgrade \
	nisaba-history

# =============================================================================
# HELP
# =============================================================================

help:
	@echo ""
	@echo "============================================================================="
	@echo " NOÖSPHERA — Infrastructure Commands"
	@echo "============================================================================="
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-build       Build infrastructure images"
	@echo "  make infra-up          Start all services"
	@echo "  make infra-down        Stop all services"
	@echo "  make infra-restart     Restart services"
	@echo "  make infra-rebuild     Rebuild and restart infrastructure"
	@echo "  make infra-logs        View infrastructure logs"
	@echo "  make infra-logs-db     View PostgreSQL logs only"
	@echo "  make infra-status      Show infrastructure status"
	@echo "  make infra-ps          Show running containers"
	@echo "  make infra-config      Show runtime configuration"
	@echo "  make infra-clean       Destroy infrastructure and volumes"
	@echo ""
	@echo "Database:"
	@echo "  make db-shell          Open PostgreSQL shell"
	@echo "  make db-shell-glyphar  Open glyphar_db shell"
	@echo "  make db-shell-nisaba   Open noosphera_agents_db shell"
	@echo "  make db-backup         Backup application databases"
	@echo "  make db-backup-all     Full cluster backup"
	@echo "  make db-restore        Restore database backup"
	@echo ""
	@echo "Glyphar:"
	@echo "  make glyphar-migrate m=\"message\""
	@echo "  make glyphar-upgrade"
	@echo "  make glyphar-history"
	@echo ""
	@echo "Nisaba:"
	@echo "  make nisaba-migrate m=\"message\""
	@echo "  make nisaba-upgrade"
	@echo "  make nisaba-history"
	@echo ""
	@echo "Operations:"
	@echo "  make bootstrap         Full infrastructure bootstrap"
	@echo "  make metrics           Show container resource usage"
	@echo ""

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

infra-build:
	@echo ""
	@echo "Building Noosphera infrastructure..."
	@$(DOCKER_COMPOSE) build --no-cache

infra-up:
	@echo ""
	@echo "Starting Noosphera infrastructure..."
	@$(DOCKER_COMPOSE) up -d
	@$(MAKE) wait-db

infra-down:
	@echo ""
	@echo "Stopping infrastructure..."
	@$(DOCKER_COMPOSE) down

infra-restart:
	@echo ""
	@echo "Restarting infrastructure..."
	@$(DOCKER_COMPOSE) restart

infra-rebuild:
	@echo ""
	@echo "Rebuilding infrastructure..."
	@$(DOCKER_COMPOSE) down
	@$(DOCKER_COMPOSE) build --no-cache
	@$(DOCKER_COMPOSE) up -d
	@$(MAKE) wait-db

infra-logs:
	@$(DOCKER_COMPOSE) logs -f

infra-logs-db:
	@docker logs -f $(POSTGRES_CONTAINER_NAME)

infra-status:
	@echo ""
	@echo "============================================================================="
	@echo " Infrastructure Status"
	@echo "============================================================================="
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "Data Directory:"
	@echo "  $(DATA_DIR)"
	@echo ""
	@if [ -d "$(DATA_DIR)" ]; then \
		du -sh $(DATA_DIR); \
	else \
		echo "Directory not found."; \
	fi

infra-ps:
	@$(DOCKER_COMPOSE) ps

infra-config:
	@echo ""
	@echo "============================================================================="
	@echo " Runtime Configuration"
	@echo "============================================================================="
	@echo ""
	@echo "POSTGRES_USER              = $(POSTGRES_USER)"
	@echo "POSTGRES_DB                = $(POSTGRES_DB)"
	@echo "POSTGRES_PORT              = $(POSTGRES_PORT)"
	@echo "PGADMIN_PORT               = $(PGADMIN_PORT)"
	@echo "DATA_DIR                   = $(DATA_DIR)"
	@echo "NETWORK_NAME               = $(NETWORK_NAME)"
	@echo "POSTGRES_CONTAINER_NAME    = $(POSTGRES_CONTAINER_NAME)"
	@echo ""
	@echo "Performance:"
	@echo "  SHARED_BUFFERS           = $(PG_SHARED_BUFFERS)"
	@echo "  WORK_MEM                 = $(PG_WORK_MEM)"
	@echo "  MAINTENANCE_WORK_MEM     = $(PG_MAINTENANCE_WORK_MEM)"
	@echo "  EFFECTIVE_CACHE_SIZE     = $(PG_EFFECTIVE_CACHE_SIZE)"
	@echo ""

infra-clean:
	@echo ""
	@echo "============================================================================="
	@echo " WARNING — DESTRUCTIVE OPERATION"
	@echo "============================================================================="
	@echo ""
	@read -p "Destroy all containers and volumes? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		$(DOCKER_COMPOSE) down -v; \
		find $(BACKUP_DIR) -name "*.sql" -delete 2>/dev/null || true; \
		find $(BACKUP_DIR) -name "*.dump" -delete 2>/dev/null || true; \
		echo ""; \
		echo "Infrastructure destroyed."; \
	else \
		echo ""; \
		echo "Operation cancelled."; \
	fi

# =============================================================================
# HEALTHCHECKS
# =============================================================================

wait-db:
	@echo ""
	@echo "Waiting for PostgreSQL healthcheck..."

	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' $(POSTGRES_CONTAINER_NAME))" = "healthy" ]; do \
		sleep 2; \
	done

	@echo "PostgreSQL is healthy."

	@echo "PostgreSQL is ready."
	@echo ""
	@echo "============================================================================="
	@echo " Infrastructure Online"
	@echo "============================================================================="
	@echo ""
	@echo "PostgreSQL : localhost:$(POSTGRES_PORT)"
	@echo "pgAdmin    : http://localhost:$(PGADMIN_PORT)"
	@echo "Neo4j      : http://localhost:7474"
	@echo "Prefect    : http://localhost:4200"
	@echo ""

# =============================================================================
# DATABASE SHELLS
# =============================================================================

db-shell:
	@docker exec -it $(POSTGRES_CONTAINER_NAME) \
		psql -U $(POSTGRES_USER)

db-shell-glyphar:
	@docker exec -it $(POSTGRES_CONTAINER_NAME) \
		psql -U $(POSTGRES_USER) -d glyphar_db

db-shell-nisaba:
	@docker exec -it $(POSTGRES_CONTAINER_NAME) \
		psql -U $(POSTGRES_USER) -d noosphera_agents_db

# =============================================================================
# BACKUP & RESTORE
# =============================================================================

db-backup:
	@echo ""
	@echo "Creating application database backups..."
	@mkdir -p $(BACKUP_DIR)

	@docker exec $(POSTGRES_CONTAINER_NAME) \
		pg_dump -U $(POSTGRES_USER) -Fc glyphar_db \
		> $(BACKUP_DIR)/glyphar_$(TIMESTAMP).dump

	@docker exec $(POSTGRES_CONTAINER_NAME) \
		pg_dump -U $(POSTGRES_USER) -Fc noosphera_agents_db \
		> $(BACKUP_DIR)/nisaba_$(TIMESTAMP).dump

	@echo ""
	@echo "Backups created:"
	@echo "  $(BACKUP_DIR)"

db-backup-all:
	@echo ""
	@echo "Creating full PostgreSQL cluster backup..."
	@mkdir -p $(BACKUP_DIR)

	@docker exec $(POSTGRES_CONTAINER_NAME) \
		pg_dumpall -U $(POSTGRES_USER) \
		> $(BACKUP_DIR)/cluster_$(TIMESTAMP).sql

	@echo ""
	@echo "Full cluster backup created."

db-restore:
	@echo ""
	@read -p "Backup file path: " file; \
	cat $$file | docker exec -i $(POSTGRES_CONTAINER_NAME) \
	psql -U $(POSTGRES_USER) -d $(POSTGRES_DB); \
	echo ""; \
	echo "Restore completed."

# =============================================================================
# GLYPHAR — ALEMBIC
# =============================================================================

glyphar-migrate:
	@echo ""
	@echo "Generating Glyphar migration..."
	@if [ -z "$(m)" ]; then \
		echo "ERROR: Use make glyphar-migrate m=\"message\""; \
		exit 1; \
	fi

	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src \
	DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini \
	revision --autogenerate -m "$(m)"

glyphar-upgrade:
	@echo ""
	@echo "Applying Glyphar migrations..."

	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src \
	DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini \
	upgrade head

	@echo ""
	@echo "Glyphar schema updated."

glyphar-history:
	@echo ""
	@echo "Glyphar migration history..."

	cd $(GLYPHAR_DIR) && \
	PYTHONPATH=src \
	DATABASE_URL=$(GLYPHAR_DB_URL) \
	alembic -c alembic/alembic.ini \
	history --verbose

# =============================================================================
# NISABA — ALEMBIC
# =============================================================================

nisaba-migrate:
	@echo ""
	@echo "Generating Nisaba migration..."

	@if [ -z "$(m)" ]; then \
		echo "ERROR: Use make nisaba-migrate m=\"message\""; \
		exit 1; \
	fi

	cd $(NISABA_DIR) && \
	PYTHONPATH=src \
	NISABA_DATABASE_URL=$(NISABA_DB_URL) \
	$(NISABA_ALEMBIC) -c alembic/alembic.ini \
	revision --autogenerate -m "$(m)"

nisaba-upgrade:
	@echo ""
	@echo "Applying Nisaba migrations..."

	cd $(NISABA_DIR) && \
	PYTHONPATH=src \
	NISABA_DATABASE_URL=$(NISABA_DB_URL) \
	$(NISABA_ALEMBIC) -c alembic/alembic.ini \
	upgrade head

	@echo ""
	@echo "Nisaba schema updated."

nisaba-history:
	@echo ""
	@echo "Nisaba migration history..."

	cd $(NISABA_DIR) && \
	PYTHONPATH=src \
	NISABA_DATABASE_URL=$(NISABA_DB_URL) \
	$(NISABA_ALEMBIC) -c alembic/alembic.ini \
	history --verbose

# =============================================================================
# OBSERVABILITY
# =============================================================================

metrics:
	@docker stats

# =============================================================================
# BOOTSTRAP
# =============================================================================

bootstrap:
	@echo ""
	@echo "============================================================================="
	@echo " Noosphera Bootstrap"
	@echo "============================================================================="
	@echo ""

	@$(MAKE) infra-up

	@echo ""
	@echo "Running Glyphar migrations..."
	@$(MAKE) glyphar-upgrade

	@echo ""
	@echo "Running Nisaba migrations..."
	@$(MAKE) nisaba-upgrade

	@echo ""
	@echo "============================================================================="
	@echo " Bootstrap completed successfully"
	@echo "============================================================================="
