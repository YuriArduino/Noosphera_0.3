"""
File metadata schema for OCR pipeline input tracking.

Captures immutable properties of input documents for auditability,
deduplication, and processing context. No business logic — pure data carrier.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class FileMetadata(BaseModel):
    """
    Immutable metadata extracted from input file.

    Used for:
        - Deduplication (SHA256 hash)
        - Audit trails (timestamps)
        - Processing context (DPI decisions based on size)
        - Output provenance (source file tracking)

    Design constraints:
        - Frozen after creation (no mutation)
        - Minimal fields (only what's needed for pipeline decisions)
        - JSON-serializable (datetime → ISO 8601)

    Example:
        >>> metadata = FileMetadata(
        ...     path="/docs/freud.pdf",
        ...     name="freud.pdf",
        ...     extension="pdf",
        ...     size_bytes=2450000,
        ...     created_at=datetime.now(),
        ...     modified_at=datetime.now(),
        ...     hash_sha256="a1b2c3...",
        ...     pages_count=320
        ... )
    """

    path: str = Field(..., description="Absolute filesystem path")
    name: str = Field(..., description="Filename with extension")
    extension: str = Field(..., description="File extension without dot (e.g., 'pdf')")
    size_bytes: int = Field(..., ge=0, description="File size in bytes")

    created_at: datetime = Field(
        ..., description="File creation timestamp (OS metadata)"
    )
    modified_at: datetime = Field(
        ..., description="Last modification timestamp (OS metadata)"
    )

    hash_sha256: str = Field(..., description="SHA256 hash for deduplication")

    pages_count: Optional[int] = Field(
        None, ge=0, description="Total page count (PDFs/images sequences)"
    )

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,  # Immutable after creation
        json_schema_extra={
            "example": {
                "path": "/data/books/freud.pdf",
                "name": "freud.pdf",
                "extension": "pdf",
                "size_bytes": 2450000,
                "created_at": "2024-01-15T10:30:00Z",
                "modified_at": "2024-01-15T10:30:00Z",
                "hash_sha256": "a1b2c3d4...",
                "pages_count": 320,
            }
        },
    )
