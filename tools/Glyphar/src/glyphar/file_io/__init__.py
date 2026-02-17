"""
File I/O abstraction layer for OCR pipeline.

Provides format-agnostic document ingestion (PDF, images) and pluggable
result export. All readers normalize output to OpenCV BGR format.

Public API:
    Base classes:
        - FileReader: Abstract document ingestion interface
        - FileWriter: Abstract result export interface

    Concrete readers:
        - PDFReader: Multi-page PDF support via pdf2image
        - ImageReader: Single-page raster images (PNG/JPG/TIFF)

    Usage:
        >>> from file_io import PDFReader
        >>> reader = PDFReader(dpi=200)
        >>> pages = reader.read(Path("book.pdf"))
        >>> pipeline.process_pages(pages)
"""

from .base import FileReader, FileWriter
from .readers import PDFReader, ImageReader

__all__ = [
    "FileReader",
    "FileWriter",
    "PDFReader",
    "ImageReader",
]
