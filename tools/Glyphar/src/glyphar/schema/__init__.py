# glyphar/schema/__init__.py
"""
Public interface for the Glyphar database schema.

This module exposes the official SQLAlchemy/SQLModel table classes and the
shared `metadata` object. All database interactions, sessions, and migrations
should import from this module to ensure a single, consistent view of the
database schema.

Example (SQLAlchemy Session):
    >>> from glyphar.schema import FileTable, metadata
    >>> session.query(FileTable).all()

Example (Alembic `env.py`):
    >>> from glyphar.schema import metadata
    >>> target_metadata = metadata
"""

from .tables import metadata, BatchTable, FileTable, PageTable

__all__ = [
    # The central registry for SQLAlchemy and Alembic
    "metadata",
    # Table Models
    "BatchTable",
    "FileTable",
    "PageTable",
]
