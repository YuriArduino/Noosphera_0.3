"""
Short-Term Memory: PostgreSQL JSONB Session Store.
Manages conversation state with TTL-based cleanup.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlmodel import Session, select, create_engine, update
from nisaba.config.memory import memory_settings
from nisaba.memory.models import SessionState


class SessionStore:
    """Manages short-term conversation state in PostgreSQL."""

    def __init__(self):
        self.enabled = memory_settings.MEMORY_ENABLED
        self.db_url = memory_settings.NISABA_DATABASE_URL

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
    ) -> int:
        """Save or update session state. Returns session ID."""
        if not self.enabled or not self.engine:
            return session_id

        with Session(self.engine) as session:
            existing = session.exec(
                select(SessionState).where(SessionState.session_id == session_id)
            ).first()

            if existing:
                # Update existing
                existing.state_data = state_data
                existing.updated_at = datetime.now(timezone.utc)
                if ttl_seconds is not None:
                    existing.ttl_seconds = ttl_seconds
                session.add(existing)
            else:
                # Create new
                new_state = SessionState(
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
                select(SessionState).where(SessionState.session_id == session_id)
            ).first()

            if not record:
                return None

            # Check TTL
            if record.ttl_seconds:
                expiry = record.created_at + timedelta(seconds=record.ttl_seconds)
                if datetime.now(timezone.utc) > expiry:
                    self.delete_state(session_id)
                    return None

            return record.state_data

    def delete_state(self, session_id: str) -> bool:
        """Delete a session state."""
        if not self.enabled or not self.engine:
            return False

        with Session(self.engine) as session:
            record = session.exec(
                select(SessionState).where(SessionState.session_id == session_id)
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
            # Find expired sessions
            expired = session.exec(
                select(SessionState).where(
                    SessionState.ttl_seconds.is_not(None),
                    SessionState.created_at + timedelta(seconds=SessionState.ttl_seconds)
                    < datetime.now(timezone.utc),
                )
            ).all()

            count = len(expired)
            for record in expired:
                session.delete(record)
            session.commit()
            return count
