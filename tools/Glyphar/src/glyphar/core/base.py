"""
Core pipeline public interface.

This module serves as the primary entry point for document-level OCR processing.
Consumers should prioritize importing from this module or the top-level
glyphar package to ensure stability across version updates.
"""

from .file_processor import FileProcessor
from .page_processor import PageProcessor
from .pipeline import OCRPipeline

__all__ = [
    "OCRPipeline",
    "FileProcessor",
    "PageProcessor",
]
