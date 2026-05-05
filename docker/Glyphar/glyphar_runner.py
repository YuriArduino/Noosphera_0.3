#!/usr/bin/env python3
"""
Ephemeral Glyphar runner for Docker/Compose with Database persistence.

Orchestrates the OCR pipeline, handles configuration merging, and persists
results into a structured SQL database using SQLModel.

Design Rationale:
    - Data Persistence: Replaces ephemeral JSON files with structured DB records.
    - Robustness: Ensures schema existence before processing starts.
    - Auditability: Groups files under batches for tracking execution runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, create_engine, select

# Glyphar Core and Domain Models
from glyphar.core.pipeline import OCRPipeline
from glyphar.engines.managed.tesseract_managed import TesseractManagedEngine
from glyphar.engines.validation import _resolve_default_tessdata
from glyphar.layout.column_detector import ColumnLayoutDetector
from glyphar.models.config import OCRConfig
from glyphar.models.output import OCROutput

# Glyphar Persistence Schema (SST) -- now fixed MRO
from glyphar.schema import metadata, BatchTable, FileTable, PageTable

# --- Environment & Paths Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:pass@db:5432/glyphar")
engine_db = create_engine(DATABASE_URL)

ROOT = Path("/workspace")
DEFAULT_RUNTIME = ROOT / "tools" / "Glyphar" / "config" / "runtime.yaml"
DEFAULT_ENV = ROOT / "tools" / "Glyphar" / "config" / "environment.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Safely load YAML configuration files."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _build_config(runtime: dict[str, Any]) -> OCRConfig:
    """Construct OCRConfig from merged runtime and environment configurations."""
    pipeline_cfg = runtime.get("pipeline", {})
    analysis_cfg = runtime.get("analysis", {})
    quality_cfg = analysis_cfg.get("quality_assessment", {})
    limits_cfg = runtime.get("limits", {})

    return OCRConfig(
        languages=runtime.get("engine", {}).get("language", "por+eng"),
        min_confidence=float(analysis_cfg.get("confidence_threshold", 30.0)),
        parallel=bool(pipeline_cfg.get("enable_parallelism", True)),
        max_workers=int(pipeline_cfg.get("max_workers", 4)),
        timeout_per_page_s=int(limits_cfg.get("timeout_seconds", 300)),
        enable_quality_assessment=bool(quality_cfg.get("enabled", True)),
    )


def init_db(retries: int = 5, delay: int = 3):
    """Ensure database tables exist, with basic retry logic for DB boot-up."""
    print(f"[*] Connecting to database at {DATABASE_URL.split('@')[-1]}...")
    for i in range(retries):
        try:
            metadata.create_all(engine_db)
            print("[+] Database schema is ready.")
            return
        except OperationalError:
            print(f"[!] DB not ready (attempt {i+1}/{retries}). Waiting {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to database.")


def persist_results(result: OCROutput, batch_ext_id: str | None = None) -> int:
    """
    Map the in-memory OCROutput DTO to the SQLModel database tables.

    Returns:
        int: The primary key ID of the persisted file record.
    """
    with Session(engine_db) as session:
        # 1. Handle Batch Context
        db_batch = None
        if batch_ext_id:
            # Check if batch exists or create new
            db_batch = session.exec(
                select(BatchTable).where(BatchTable.task_id == batch_ext_id)
            ).first()

            if not db_batch:
                db_batch = BatchTable(
                    task_id=batch_ext_id,
                    file_path=result.file_metadata.path,
                    status="completed",
                    config=result.config.model_dump(),
                )
                session.add(db_batch)
                session.commit()
                session.refresh(db_batch)

        # 2. Persist File Metadata and Statistics
        db_file = FileTable(
            batch_id=db_batch.id if db_batch else None,
            path=result.file_metadata.path,
            name=result.file_metadata.name,
            extension=result.file_metadata.extension,
            size_bytes=result.file_metadata.size_bytes,
            created_at=result.file_metadata.created_at,
            modified_at=result.file_metadata.modified_at,
            hash_sha256=result.file_metadata.hash_sha256,
            pages_count=result.total_pages,
            full_text=result.full_text,
            statistics=result.statistics.model_dump(),
        )
        session.add(db_file)
        session.commit()
        session.refresh(db_file)

        # 3. Persist Individual Pages (Bulk)
        for page in result.pages:
            db_page = PageTable(
                file_id=db_file.id,
                canonical_id=page.id,
                page_number=page.page_number,
                layout_type=page.layout_type,
                page_quality=page.page_quality,
                page_confidence_mean=page.page_confidence_mean,
                processing_time_s=page.processing_time_s,
                page_text_hash=page.page_text_hash,
                columns=[col.model_dump() for col in page.columns],
                warnings=page.warnings,
            )
            session.add(db_page)

        session.commit()
        return db_file.id


def main() -> None:
    """Main execution orchestrator."""
    # Initialization
    init_db()

    # Path Resolution
    input_path = Path(os.environ.get("GLYPHAR_INPUT", "/data/input/document.pdf"))
    output_dir = Path(os.environ.get("GLYPHAR_OUTPUT_DIR", "/data/output"))
    default_batch_name = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    batch_id = os.environ.get("GLYPHAR_BATCH_ID", default_batch_name)

    if not input_path.exists():
        raise FileNotFoundError(f"Input document not found: {input_path}")

    # Configuration Loading
    runtime_path = Path(os.environ.get("GLYPHAR_RUNTIME_CONFIG", str(DEFAULT_RUNTIME)))
    env_path = Path(os.environ.get("GLYPHAR_ENV_CONFIG", str(DEFAULT_ENV)))

    runtime = _load_yaml(runtime_path)
    env_cfg = _load_yaml(env_path)
    merged_runtime = _merge(runtime, env_cfg.get("overrides", {}))

    config = _build_config(merged_runtime)
    model_type = os.environ.get(
        "GLYPHAR_MODEL_TYPE",
        merged_runtime.get("engine", {}).get("model_type", "standard"),
    )

    # Engine and Pipeline Setup
    engine = TesseractManagedEngine(
        tessdata_dir=str(_resolve_default_tessdata()),
        languages=config.languages,
        model_type=model_type,
        config=config,
    )

    pipeline = OCRPipeline(
        engine=engine,
        layout_detector=ColumnLayoutDetector(),
        _preprocessing_strategies=[],
        config=config,
        include_llm_input=True,
    )

    # Execution (In-Memory Processing)
    print(f"[*] Processing: {input_path.name}")
    result = pipeline.process(
        file_path=str(input_path),
        parallel=config.parallel,
        max_workers=config.max_workers,
        batch_size=int(merged_runtime.get("pipeline", {}).get("batch_size", 8)),
        show_progress=True,
    )

    # Persistence (SQL Database)
    print("[*] Persisting results to database...")
    db_id = persist_results(result, batch_ext_id=batch_id)

    # Output Artifacts (Legacy text support for LLMs)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    (output_dir / f"{stem}.txt").write_text(result.full_text, encoding="utf-8")

    # Final Summary (Standard Output)
    print(
        json.dumps(
            {
                "status": "success",
                "db_file_id": db_id,
                "batch_id": batch_id,
                "input": str(input_path),
                "pages": result.total_pages,
                "avg_confidence": round(result.average_confidence, 2),
                "output_txt": str(output_dir / f"{stem}.txt"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        sys.exit(1)
