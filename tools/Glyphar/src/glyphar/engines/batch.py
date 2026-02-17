"""Batch processing strategies for Tesseract OCR engine."""

from typing import List, Any, Dict

from glyphar.engines.base import OCREngine


class TesseractBatchProcessor:
    """
    Batch processor implementing a heuristic PSM rotation strategy.

    Rotates Page Segmentation Modes (PSM) across pages to increase
    robustness for heterogeneous document layouts.

    Rotation pattern:
        - Index % 3 == 0 → PSM 3  (Fully automatic page segmentation)
        - Index % 3 == 1 → PSM 6  (Uniform block of text)
        - Index % 3 == 2 → PSM 11 (Sparse text)

    This heuristic increases layout diversity without performing
    expensive per-page structural analysis.

    Trade-offs:
        + May improve accuracy on mixed-layout documents
        - Slight performance overhead
        - Not optimal for strictly uniform documents
    """

    def __init__(self, engine: OCREngine):
        """
        Initialize the batch processor.

        Args:
            engine (OCREngine):
                Configured engine instance used for recognition.
        """
        self.engine = engine

    def recognize_batch(
        self,
        images: List[Any],
        config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images using PSM rotation.

        Args:
            images (List[Any]):
                List of input images (numpy arrays).
            config (Dict[str, Any]):
                Base OCR configuration applied to all pages.
                The "psm" parameter may be overridden per page.

        Returns:
            List[Dict[str, Any]]:
                OCR results in the same order as input images.
        """
        if not images:
            return []

        results: List[Dict[str, Any]] = []

        for index, image in enumerate(images):
            page_config = config.copy()

            # Default PSM (explicit)
            psm = page_config.get("psm", 3)

            rotation = index % 3
            if rotation == 1:
                psm = 6
            elif rotation == 2:
                psm = 11

            page_config["psm"] = psm

            result = self.engine.recognize(image, page_config)
            results.append(result)

        return results
