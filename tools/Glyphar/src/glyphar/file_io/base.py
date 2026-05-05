"""
Abstract base class for file ingestion in the OCR pipeline.

Defines the contract for reading various document formats and converting them
into a standardized internal representation for the pipeline.

Design Rationale:
    - ABC over Protocol: Enforces explicit implementation via inheritance.
    - Path-based I/O: Works with filesystem paths for simplicity.
    - List[Any] return type: Accommodates both single-page (images) and
      multi-page (PDFs) documents seamlessly.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class FileReader(ABC):
    """
    Abstract interface for document ingestion into the OCR pipeline.

    Concrete implementations must handle format-specific conversion to
    OpenCV-compatible numpy arrays (BGR format).

    Contract:
        - Input: A filesystem path (Path object).
        - Output: A list of page images (List[numpy.ndarray], BGR format).
        - Errors: Raise ValueError/FileNotFoundError on invalid input.

    Example Implementation:
        >>> class MockReader(FileReader):
        ...     def read(self, path: Path) -> List[np.ndarray]:
        ...         return [np.zeros((1000, 800, 3), dtype=np.uint8)]
    """

    @abstractmethod
    def read(self, path: Path) -> List[Any]:
        """
        Convert a document file to a list of page images.

        Args:
            path: Absolute or relative filesystem path to the input document.

        Returns:
            A list of page images as numpy arrays in BGR format (OpenCV standard).
            Single-page documents return a list with one element.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the file format is unsupported or corrupted.
        """
