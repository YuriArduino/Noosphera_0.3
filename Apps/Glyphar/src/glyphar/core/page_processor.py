"""
Single-page OCR processing orchestrator.

Coordinates quality assessment, layout detection, and region-based OCR
execution for a single document page. Stateless design enables parallelization.
"""

import time
from typing import Any, List, Mapping
import numpy as np

from ..models.page import PageResult
from ..models.column import ColumnResult

from ..analysis.quality_assessor import QualityAssessor
from ..optimization.config_optimizer import ConfigOptimizer


class PageProcessor:
    """
    Orchestrates complete OCR processing for a single page.

    Pipeline stages:
        1. Quality assessment → adaptive preprocessing selection
        2. Layout detection → region segmentation
        3. Region processing → per-column OCR with optimal config
        4. Result aggregation → PageResult construction

    Design constraints:
        - Stateless: No persistent state between page invocations
        - Thread-safe: Safe for concurrent execution (parallel processing)
        - Fail-safe: Always returns valid PageResult (fallback on errors)

    Performance:
        - Typical page: 1.5-3.0s (depends on quality/preprocessing)
        - Memory: ~50MB peak per page (300 DPI, 2000px width)
    """

    def __init__(
        self,
        engine,
        layout_detector,
        preprocessing_strategies=None,  # pylint: disable=unused-argument
        min_confidence: float = 30.0,
    ) -> None:
        """
        Initialize page processor with required dependencies.

        Args:
            engine: OCREngine instance (TesseractEngine).
            layout_detector: LayoutDetector instance (ColumnLayoutDetector).
            preprocessing_strategies: Deprecated — ignored (kept for API compat).
            min_confidence: Minimum word confidence threshold (0.0-100.0).
        """
        self.layout_detector = layout_detector
        self.min_confidence = min_confidence
        self.quality_assessor = QualityAssessor()
        self.config_optimizer = ConfigOptimizer(engine)

    def process(self, image: Any, page_number: int) -> PageResult:
        """
        Process single page through complete OCR pipeline.

        Args:
            image: Page image as numpy array (BGR format).
            page_number: 1-based page number for metadata.

        Returns:
            PageResult with OCR text, confidence metrics, and layout metadata.

        Error handling:
            Individual region failures do NOT abort page processing —
            failed regions return empty ColumnResult with 0.0 confidence.
            Entire page failure triggers fallback_page (0.0 confidence).
        """
        t0 = time.perf_counter()

        # Stage 1: Quality assessment
        quality_metrics = self.quality_assessor.assess(image)

        # Stage 2: Layout detection
        layout = self.layout_detector.detect(image)
        layout_type = layout["layout_type"]
        regions = layout.get("regions", [])

        # Stage 3: Region processing
        columns: List[ColumnResult] = []
        confidences: List[float] = []

        for region in regions:
            try:
                column = self._process_region(
                    image=image,
                    region=region,
                    layout_type=layout_type,
                    quality_metrics=quality_metrics,
                )
                columns.append(column)
                confidences.append(column.confidence)
            except (RuntimeError, ValueError, TypeError, KeyError):
                # Isolated region failure — continue processing other regions
                columns.append(
                    ColumnResult(
                        col_index=region["col_index"],
                        text="",
                        confidence=0.0,
                        word_count=0,
                        char_count=0,
                        processing_time_s=0.0,
                        bbox=self._safe_bbox(region),
                        region_id=region.get("id"),
                        config_used=None,
                    )
                )
                confidences.append(0.0)

        # Stage 4: Result aggregation
        return PageResult(
            page_number=page_number,
            layout_type=layout_type,
            columns=columns,
            page_confidence_mean=float(np.mean(confidences)) if confidences else 0.0,
            processing_time_s=time.perf_counter() - t0,
            config_used=None,
        )

    def _process_region(
        self,
        image: Any,
        region: Mapping[str, Any],
        layout_type: str,
        quality_metrics: Mapping[str, Any],
    ) -> ColumnResult:
        """Process single region with optimal OCR configuration."""
        region_img = self._extract_region(image, region)

        ocr_result = self.config_optimizer.find_optimal_config(
            image=region_img,
            layout_type=layout_type,
            quality_metrics=quality_metrics,
        )

        text = str(ocr_result.get("text", ""))
        confidence = float(ocr_result.get("confidence", 0.0))
        words = ocr_result.get("words", [])
        if not isinstance(words, list):
            words = []

        config_used = ocr_result.get("config_used")
        if not isinstance(config_used, str):
            config_used = str(config_used) if config_used is not None else "unknown"

        word_count = int(ocr_result.get("word_count", len(words)))
        char_count = int(ocr_result.get("char_count", len(text)))
        processing_time_s = float(ocr_result.get("time_s", 0.0))

        return ColumnResult(
            col_index=region["col_index"],
            text=text,
            confidence=confidence,
            word_count=word_count,
            char_count=char_count,
            processing_time_s=processing_time_s,
            bbox=self._safe_bbox(region),
            region_id=region.get("id"),
            config_used=config_used,
        )

    @staticmethod
    def _extract_region(image: Any, region: Mapping[str, Any]) -> Any:
        """Extract region subimage using bounding box coordinates."""
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        return image[y : y + h, x : x + w]

    @staticmethod
    def _safe_bbox(region: Mapping[str, Any]):
        """Return valid bbox dict or None if dimensions invalid."""
        if region.get("w", 0) > 0 and region.get("h", 0) > 0:
            return {
                "x": region["x"],
                "y": region["y"],
                "w": region["w"],
                "h": region["h"],
            }
        return None
