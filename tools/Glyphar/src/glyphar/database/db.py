"""
Database connection and session management.

Configures the SQLAlchemy engine and provides session factories for
synchronous database operations.
"""

import os
from sqlmodel import Session, create_engine

# DATABASE_URL será injetada pelo Compose/Prefect
DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)


def get_session():
    """Provide a transactional scope around a series of operations."""
    with Session(engine) as session:
        yield session
