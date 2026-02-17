"""
Concrete file readers for common document formats.

Supports:
    - PDF (via pdf2image/poppler)
    - Raster images (PNG, JPG, TIFF via OpenCV)

All readers normalize output to OpenCV BGR format (3-channel uint8)
for consistent pipeline consumption.
"""

from pathlib import Path
from typing import List, Any
import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

from .base import FileReader


# pylint: disable=too-few-public-methods no-member
class PDFReader(FileReader):
    """
    PDF document reader using pdf2image (Poppler backend).

    Converts PDF pages to OpenCV-compatible BGR images at configurable DPI.
    Handles multi-page documents natively.

    Dependencies:
        - poppler-utils (system package)
        - pdf2image (Python wrapper)

    Performance characteristics:
        - 300 DPI: ~1.2s/page on Intel i7 (2000px width)
        - 200 DPI: ~0.7s/page (recommended for LLM post-processing)
        - Memory: ~5MB/page at 300 DPI

    Example:
        >>> reader = PDFReader(dpi=200)
        >>> pages = reader.read(Path("document.pdf"))
        >>> assert len(pages) == 10  # 10-page document
        >>> assert pages[0].shape[2] == 3  # BGR format
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize PDF reader with rendering resolution.

        Args:
            dpi: Dots per inch for PDF rasterization.
                Recommended values:
                    - 300: Production quality (slower)
                    - 200: Balanced (recommended for LLM pipelines)
                    - 150: Speed-optimized (acceptable for clean digital PDFs)

        Raises:
            ValueError: If dpi outside valid range (72-600)
        """
        if not 72 <= dpi <= 600:
            raise ValueError(f"DPI must be between 72 and 600, got {dpi}")
        self.dpi = dpi

    def read(self, path: Path) -> List[Any]:
        """
        Convert PDF file to list of OpenCV BGR images.

        Args:
            path: Path to PDF file.

        Returns:
            List of numpy arrays (BGR format, uint8) — one per page.

        Raises:
            FileNotFoundError: If PDF file does not exist
            ValueError: If PDF is encrypted, corrupted, or unreadable
            RuntimeError: If Poppler not installed or pdf2image misconfigured

        Error handling:
            - Encrypted PDFs: Raise ValueError with clear message
            - Missing Poppler: Raise RuntimeError with installation hint
            - Empty PDFs: Return empty list (valid edge case)
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        try:
            # Convert PDF pages to PIL images
            pil_images = convert_from_path(str(path), dpi=self.dpi)

            # Convert PIL (RGB) → OpenCV (BGR)
            return [
                cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_images
            ]

        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "Poppler not installed. Install with:\n"
                "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "  macOS: brew install poppler\n"
                "  Windows: https://github.com/oschwartz10612/poppler-windows"
            ) from e

        except PDFPageCountError as e:
            raise ValueError(f"Invalid or encrypted PDF: {path}") from e

        except Exception as e:
            raise ValueError(f"Failed to read PDF {path}: {e}") from e


class ImageReader(FileReader):
    """
    Raster image reader supporting common formats (PNG, JPG, TIFF, BMP).

    Uses OpenCV imread for format-agnostic loading. Normalizes all outputs
    to BGR format regardless of input color space.

    Supported formats:
        - PNG (with/without alpha)
        - JPG/JPEG
        - TIFF (single-page)
        - BMP
        - WebP (if OpenCV built with libwebp)

    Example:
        >>> reader = ImageReader()
        >>> pages = reader.read(Path("page.jpg"))
        >>> assert len(pages) == 1  # Single-page image
        >>> assert pages[0].dtype == np.uint8
    """

    def read(self, path: Path) -> List[Any]:
        """
        Load raster image file as OpenCV BGR array.

        Args:
            path: Path to image file.

        Returns:
            List containing single numpy array (BGR format, uint8).

        Raises:
            FileNotFoundError: If image file does not exist
            ValueError: If file unreadable or unsupported format

        Note:
            Alpha channels (PNG) are discarded — output always 3-channel BGR.
            Grayscale images are converted to 3-channel BGR (replicated channels).
        """
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # OpenCV imread returns None on failure (no exception)
        img = cv2.imread(str(path))

        if img is None:
            supported = ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]
            ext = path.suffix.lower().lstrip(".")
            hint = (
                f" (supported: {', '.join(supported)})" if ext not in supported else ""
            )
            raise ValueError(
                f"Cannot read image {path}{hint}. Check format/permissions."
            )

        # Ensure 3-channel BGR (handles grayscale input)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return [img]
