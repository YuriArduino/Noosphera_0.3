"""
Core pipeline public interface.

This module exports the primary entry point for OCR processing.
All other core modules are implementation details — consumers should
import ONLY from this module or top-level glyphar package.
"""

from core.file_processor import FileProcessor
from core.page_processor import PageProcessor
from .pipeline import OCRPipeline


__all__ = [
    "OCRPipeline",
    "FileProcessor",
    "PageProcessor",
]
