"""
Abstract base classes for file I/O operations in OCR pipeline.

Defines contracts for document ingestion (FileReader) and result export (FileWriter).
Enables pluggable storage backends without modifying core pipeline logic.

Design rationale:
    - ABC over Protocol: Enforces explicit implementation via inheritance
    - Path-based I/O: Works with filesystem paths (not raw bytes) for simplicity
    - List[Any] return type: Accommodates both single-page (images) and multi-page (PDFs)
    - No error handling in base: Delegated to concrete implementations
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class FileReader(ABC):
    """
    Abstract interface for document ingestion into OCR pipeline.

    Concrete implementations must handle format-specific conversion to
    OpenCV-compatible numpy arrays (BGR format).

    Contract:
        - Input: Filesystem path (Path object)
        - Output: List of page images (List[numpy.ndarray], BGR format)
        - Errors: Raise ValueError/FileNotFoundError on invalid input

    Example implementation:
        >>> class MockReader(FileReader):
        ...     def read(self, path: Path) -> List[np.ndarray]:
        ...         return [np.zeros((1000, 800, 3), dtype=np.uint8)]
    """

    @abstractmethod
    def read(self, path: Path) -> List[Any]:
        """
        Convert document file to list of page images.

        Args:
            path: Absolute or relative filesystem path to input document.

        Returns:
            List of page images as numpy arrays in BGR format (OpenCV standard).
            Single-page documents return list with one element.

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If file format unsupported or corrupted

        Note:
            Implementations should handle format detection internally.
            Output images must be compatible with OpenCV operations
            (dtype=uint8, channels=3 for color or 1 for grayscale).
        """


class FileWriter(ABC):
    """
    Abstract interface for exporting OCR results to persistent storage.

    Concrete implementations handle serialization to target format
    (JSON, plain text, structured databases).

    Contract:
        - Input: OCROutput instance (or compatible dict)
        - Output: Side effect (file written to filesystem)
        - Errors: Raise IOError on write failures

    Example implementation:
        >>> class JSONWriter(FileWriter):
        ...     def write(self, result: OCROutput, output_path: Path):
        ...         output_path.write_text(result.model_dump_json(indent=2))
    """

    @abstractmethod
    def write(self, result: Any, output_path: Path) -> None:
        """
        Serialize OCR result to filesystem.

        Args:
            result: OCROutput instance or compatible data structure.
            output_path: Target filesystem path for output file.

        Raises:
            IOError: If write operation fails (permissions, disk space, etc.)
            TypeError: If result type unsupported by this writer

        Note:
            Implementations should handle directory creation if needed.
            Atomic writes recommended to prevent partial/corrupted outputs.
        """
