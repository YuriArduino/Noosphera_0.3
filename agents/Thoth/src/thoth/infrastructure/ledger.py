"""
Cognitive Ledger for Thoth Agent.

Responsible for:
- Logging decisions
- Logging corrections
- Recording semantic experiences

Separate from LangGraph checkpoint DB.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from thoth.config import memory_settings


class ThothLedger:
    """Cognitive learning ledger (passive memory)."""

    def __init__(self):
        self.enabled: bool = memory_settings.LEDGER_ENABLED
        self.db_path: Path = memory_settings.LEDGER_DB_PATH

        if not self.enabled:
            # Create a dummy in-memory connection to satisfy type safety
            self.conn: sqlite3.Connection = sqlite3.connect(":memory:")
            return

        self.conn: sqlite3.Connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        if memory_settings.LEDGER_AUTO_MIGRATE:
            self._create_tables()

    # ==========================================================
    # TABLE CREATION
    # ==========================================================

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS decision_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            action TEXT NOT NULL,
            strategy TEXT,
            avg_confidence REAL,
            attempts INTEGER,
            execution_step TEXT,
            hitl_triggered INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS correction_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            model_name TEXT NOT NULL,
            original_confidence REAL,
            final_confidence REAL,
            processing_time REAL,
            urgency TEXT,
            success INTEGER,
            created_at TEXT NOT NULL
        );
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS semantic_experience (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            layout_type TEXT,
            page_quality TEXT,
            error_type TEXT,
            snippet TEXT,
            strategy_used TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        );
        """
        )

        self.conn.commit()

    # ==========================================================
    # LOGGING METHODS
    # ==========================================================

    def log_decision(
        self,
        document_id: str,
        document_hash: str,
        action: str,
        strategy: Optional[str],
        avg_confidence: float,
        attempts: int,
        execution_step: str,
        hitl_triggered: bool,
    ):
        """Log a Thoth decision event to the ledger."""

        if not self.conn:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT INTO decision_ledger (
            document_id,
            document_hash,
            action,
            strategy,
            avg_confidence,
            attempts,
            execution_step,
            hitl_triggered,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
            (
                document_id,
                document_hash,
                action,
                strategy,
                avg_confidence,
                attempts,
                execution_step,
                int(hitl_triggered),
                datetime.utcnow().isoformat(),
            ),
        )

        self.conn.commit()

    def log_correction(
        self,
        document_id: str,
        document_hash: str,
        model_name: str,
        original_confidence: float,
        final_confidence: float,
        processing_time: float,
        urgency: str,
        success: bool,
    ):
        """Log an LLM correction event to the ledger."""
        if not self.conn:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT INTO correction_ledger (
            document_id,
            document_hash,
            model_name,
            original_confidence,
            final_confidence,
            processing_time,
            urgency,
            success,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
            (
                document_id,
                document_hash,
                model_name,
                original_confidence,
                final_confidence,
                processing_time,
                urgency,
                int(success),
                datetime.utcnow().isoformat(),
            ),
        )

        self.conn.commit()

    def log_semantic_experience(
        self,
        document_id: str,
        document_hash: str,
        layout_type: Optional[str],
        page_quality: Optional[str],
        error_type: Optional[str],
        snippet: Optional[str],
        strategy_used: Optional[str],
        confidence: Optional[float],
    ):
        """Log a semantic experience for future learning and retrieval."""
        if not self.conn:
            return

        cursor = self.conn.cursor()
        cursor.execute(
            """
        INSERT INTO semantic_experience (
            document_id,
            document_hash,
            layout_type,
            page_quality,
            error_type,
            snippet,
            strategy_used,
            confidence,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
            (
                document_id,
                document_hash,
                layout_type,
                page_quality,
                error_type,
                snippet,
                strategy_used,
                confidence,
                datetime.utcnow().isoformat(),
            ),
        )

        self.conn.commit()

    def close(self):
        """Close ledger database connection."""
        self.conn.close()
