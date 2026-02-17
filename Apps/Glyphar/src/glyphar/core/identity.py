"""
Identity utilities for hash and canonical ID generation.

Provides SHA256 hashing and canonical ID creation for files and pages.
Useful for deduplication, audit, and tracking in OCR pipeline.
"""
import hashlib
from typing import Union

class Identity:
    """
    Identity utility class for hash and ID generation.
    """
    @staticmethod
    def sha256_hash(data: Union[str, bytes]) -> str:
        """Generate SHA256 hash for given data (string or bytes)."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def canonical_id(prefix: str, date: str, number: int) -> str:
        """Generate canonical id for files, e.g., pdf_A_20260216_001."""
        return f"{prefix}_{date}_{number:03d}"
