"""
Document I/O orchestration layer.

Abstracts file format detection and page extraction behind unified interface.
Enables format-agnostic pipeline consumption (PDFs, images treated identically).
"""

from pathlib import Path
from typing import List, Any
from glyphar.file_io.readers import PDFReader, ImageReader


def read_pages(file_reader, path: Path) -> List[Any]:
    """
    Extract page images from document file.

    Args:
        file_reader: FileReader instance (PDFReader/ImageReader).
        path: Filesystem path to input document.

    Returns:
        List of page images as numpy arrays (BGR format).
        Single-page images return list with one element.

    Note:
        Delegates actual reading to concrete FileReader implementation.
        This function provides uniform interface across formats.
    """
    return file_reader.read(path)


def make_default_reader(path: Path, dpi: int = 300):
    """
    Auto-detect document format and return appropriate reader.

    Args:
        path: Filesystem path to document.
        dpi: Rendering DPI for PDFs (ignored for raster images).

    Returns:
        FileReader instance configured for detected format.

    Format detection:
        - Raster images (.png, .jpg, .jpeg, .tiff): ImageReader
        - All others (assumed PDF): PDFReader with specified DPI

    Example:
        >>> reader = make_default_reader(Path("book.pdf"), dpi=200)
        >>> assert isinstance(reader, PDFReader)
    """
    suffix = path.suffix.lower()
    if suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
        return ImageReader()
    return PDFReader(dpi=dpi)
