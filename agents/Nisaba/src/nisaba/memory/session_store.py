"""
Short-Term Memory: PostgreSQL JSONB Session Store.
Manages conversation state with TTL-based cleanup.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from sqlmodel import Session, select, create_engine, update
from sqlalchemy import text
from agents.shared.config.memory import memory_settings
from nisaba.schema.tables import SessionStateTable  # modelo físico correto

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages short-term conversation state in PostgreSQL."""

    def __init__(self):
        self.enabled = memory_settings.MEMORY_ENABLED
        self.db_url = memory_settings.DATABASE_URL

        if not self.enabled:
            self.engine = None
            return

        self.engine = create_engine(self.db_url, pool_pre_ping=True)

    def save_state(
        self,
        session_id: str,
        state_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Save or update session state. Returns session ID."""
        if not self.enabled or not self.engine:
            return session_id

        with Session(self.engine) as session:
            # Tenta atualizar primeiro (mais performático)
            stmt = (
                update(SessionStateTable)
                .where(SessionStateTable.session_id == session_id)
                .values(
                    state_data=state_data,
                    updated_at=datetime.now(timezone.utc),
                    ttl_seconds=(
                        ttl_seconds if ttl_seconds is not None else SessionStateTable.ttl_seconds
                    ),
                )
            )
            result = session.exec(stmt)
            # Se nenhuma linha foi afetada, insere uma nova
            if result.rowcount == 0:
                new_state = SessionStateTable(
                    session_id=session_id,
                    user_id=user_id,
                    state_data=state_data,
                    ttl_seconds=ttl_seconds or 3600,
                )
                session.add(new_state)

            session.commit()
            return session_id

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session state by ID."""
        if not self.enabled or not self.engine:
            return None

        with Session(self.engine) as session:
            record = session.exec(
                select(SessionStateTable).where(SessionStateTable.session_id == session_id)
            ).first()

            if not record:
                return None

            # Verifica TTL usando SQL nativo
            if record.ttl_seconds and self._is_expired(record):
                self.delete_state(session_id)
                return None

            return record.state_data

    def _is_expired(self, record: SessionStateTable) -> bool:
        """Verifica se o registro expirou (consulta ao banco)."""
        with Session(self.engine) as session:
            stmt = text(
                "SELECT 1 FROM nisaba.session_state WHERE session_id = :sid "
                "AND created_at + (ttl_seconds * INTERVAL '1 second') < NOW()"
            )
            result = session.exec(stmt, {"sid": record.session_id}).first()
            return result is not None

    def delete_state(self, session_id: str) -> bool:
        """Delete a session state."""
        if not self.enabled or not self.engine:
            return False

        with Session(self.engine) as session:
            record = session.exec(
                select(SessionStateTable).where(SessionStateTable.session_id == session_id)
            ).first()
            if record:
                session.delete(record)
                session.commit()
                return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of deleted records."""
        if not self.enabled or not self.engine:
            return 0

        with Session(self.engine) as session:
            stmt = text(
                "DELETE FROM nisaba.session_state "
                "WHERE ttl_seconds IS NOT NULL "
                "AND created_at + (ttl_seconds * INTERVAL '1 second') < NOW()"
            )
            result = session.exec(stmt)
            session.commit()
            return result.rowcount
