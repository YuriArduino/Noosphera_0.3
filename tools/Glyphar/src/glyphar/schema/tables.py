# glyphar/schema/tables.py
"""
Single Source of Truth (SST) for the Glyphar Database Schema.

This module defines the physical database tables by inheriting from the logical
domain models in `glyphar.models`. It is the sole entry point for database-
aware components like SQLAlchemy sessions and Alembic migrations.

Design Principles:
    - DRY (Don't Repeat Yourself): Field definitions are inherited from models.
    - Mutability: Table classes override the 'frozen' constraint of domain models
      to allow the database to manage IDs and timestamps.
    - SQLAlchemy Native: Uses JSONB for complex nested data structures.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import ConfigDict  # Required to override the 'frozen' constraint
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB

# Import base logical classes from models
from glyphar.models.batch import BatchTask
from glyphar.models.file import FileMetadata
from glyphar.models.page import PageResult
from glyphar.models.stats import ProcessingStatistics
from glyphar.models.config import OCRConfig

# Registry for SQLAlchemy and Alembic
metadata = SQLModel.metadata


class TimestampMixin(SQLModel):
    """Mixin to provide automatic audit timestamps for all tables."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": text("CURRENT_TIMESTAMP"),
        },
    )


# ---------------------------------------------------------------------------
# Corrected MRO: Do NOT inherit directly from SQLModel again.
# Each domain model (BatchTask, FileMetadata, PageResult) already inherits
# from SQLModel, so adding it explicitly would create an inconsistent
# Method Resolution Order (MRO) and raise TypeError.
# ---------------------------------------------------------------------------


class BatchTable(BatchTask, TimestampMixin, table=True):
    """
    Represents the `ocr_batches` table.
    Materializes BatchTask into a mutable database record.
    """

    __tablename__ = "ocr_batches"

    # CRITICAL: Overrides the frozen status of BatchTask to allow DB updates
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)

    # Prevent storing file paths in the batch table, as they are now stored
    # in the FileTable.
    file_path: Optional[str] = Field(default=None)

    # Store OCRConfig as JSONB for high flexibility
    config: OCRConfig = Field(..., sa_column=Column(JSONB))

    # Relationship with files
    files: List["FileTable"] = Relationship(back_populates="batch")


class FileTable(FileMetadata, TimestampMixin, table=True):
    """
    Represents the `ocr_files` table.
    Aggregates FileMetadata with statistics and full text results.
    """

    __tablename__ = "ocr_files"
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: Optional[int] = Field(
        default=None, foreign_key="ocr_batches.id", index=True
    )

    # Extended result data
    full_text: Optional[str] = Field(default=None)
    statistics: ProcessingStatistics = Field(..., sa_column=Column(JSONB))

    # Relationships
    batch: Optional[BatchTable] = Relationship(back_populates="files")
    pages: List["PageTable"] = Relationship(back_populates="file")


class PageTable(PageResult, TimestampMixin, table=True):
    """
    Represents the `ocr_pages` table.
    Stores individual page results with nested column data as JSONB.
    """

    __tablename__ = "ocr_pages"
    model_config = ConfigDict(frozen=False, extra="ignore")

    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="ocr_files.id", index=True)

    # Persistence overrides for complex nested types
    columns: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSONB))
    warnings: List[str] = Field(default=[], sa_column=Column(JSONB))

    # Relationship back to file
    file: "FileTable" = Relationship(back_populates="pages")
