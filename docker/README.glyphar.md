# Glyphar via Docker Compose (ephemeral)

## Build
```bash
docker compose -f docker/docker-compose.glyphar.yml build
```

## Run once (default sample PDF)
```bash
docker compose -f docker/docker-compose.glyphar.yml run --rm glyphar
```

## Run with custom file and model profile
```bash
GLYPHAR_INPUT=/data/input/PDF_B_Digital.pdf \
GLYPHAR_MODEL_TYPE=best \
docker compose -f docker/docker-compose.glyphar.yml run --rm glyphar
```

## Outputs
Artifacts are written to `docker/glyphar-output/`:
- `<name>.txt`
- `<name>.json`
- `<name>.summary.json`

The runner loads and merges:
1. `tools/Glyphar/config/runtime.yaml`
2. `tools/Glyphar/config/environment.yaml` (`overrides` section)
