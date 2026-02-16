"""
File metadata extraction utilities.

Lightweight helpers to populate FileMetadata schema from filesystem.
No heavy processing — pure metadata extraction.
"""

from pathlib import Path
from datetime import datetime
from ..models.file import FileMetadata


def extract_file_metadata(path: Path, pages_count: int | None = None) -> FileMetadata:
    """
    Extract immutable metadata from filesystem path.

    Args:
        path: Absolute or relative filesystem path.
        pages_count: Optional page count (for PDFs/images sequences).

    Returns:
        Fully populated FileMetadata instance.

    Note:
        Hash computation omitted for performance — caller should compute
        SHA256 separately if deduplication required.
    """
    stat = path.stat()
    return FileMetadata(
        path=str(path.resolve()),
        name=path.name,
        extension=path.suffix.lstrip("."),
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        modified_at=datetime.fromtimestamp(stat.st_mtime),
        hash_sha256="",  # Empty — compute separately if needed
        pages_count=pages_count,
    )
