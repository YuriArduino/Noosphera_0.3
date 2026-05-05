from prefect import flow, task
from typing import Dict, Any

# Glyphar é uma lib instalada no container do Worker
from glyphar.core.pipeline import OCRPipeline
from glyphar.database import persist_results


@task(name="SST Persistence")
def save_to_postgres(ocr_output, batch_id):
    """Garante que o resultado do container efêmero morra mas o dado viva."""
    return persist_results(ocr_output, batch_id)


@flow(name="Glyphar Ephemeral Execution", log_prints=True)
def run_glyphar_flow(file_path: str, strategy_manifest: Dict[str, Any]):
    """
    O Prefect Worker executa este flow dentro de um container Docker novo.
    """
    # 1. Setup do motor (configura o Glyphar conforme o pedido do Agente)
    # Aqui usamos os overrides que o Thoth enviou no manifest
    pipeline = OCRPipeline(overrides=strategy_manifest.get("overrides"))

    # 2. Execução (Motor Glyphar em ação)
    result = pipeline.process(file_path=file_path)

    # 3. Persistência (SST no Postgres)
    db_id = save_to_postgres(result, strategy_manifest.get("batch_id"))

    return {"db_file_id": db_id, "confidence": result.average_confidence}
