"""
File I/O abstraction layer for the OCR pipeline's ingestion stage.

Provides a format-agnostic way to read documents (PDFs, images) and
normalizes them into a consistent format (OpenCV BGR images) for the core
processing logic.

Public API:
    Base Class:
        - FileReader: The abstract interface for all document readers.

    Concrete Readers:
        - PDFReader: For multi-page PDFs, using the pdf2image library.
        - ImageReader: For single-page raster images (PNG/JPG/TIFF).
"""

from .base import FileReader
from .readers import PDFReader, ImageReader

__all__ = [
    "FileReader",
    "PDFReader",
    "ImageReader",
]
