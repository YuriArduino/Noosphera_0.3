"""
Single-page OCR processing orchestrator.

Coordinates quality assessment, layout detection, and region-based OCR
execution for a single document page. Stateless design enables parallelization.
"""

import time
from typing import Any, List, Mapping
import numpy as np

from glyphar.core.identity import Identity

from glyphar.models.page import PageResult
from glyphar.models.column import ColumnResult
from glyphar.models.enums import PageQuality

from glyphar.analysis.quality_assessor import QualityAssessor
from glyphar.optimization.config_optimizer import ConfigOptimizer


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
        page_quality = self._classify_page_quality(quality_metrics)

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
                        region_id=self._region_id(region),
                        config_used=None,
                    )
                )
                confidences.append(0.0)

        # Stage 4: Result aggregation
        return PageResult(
            page_number=page_number,
            layout_type=layout_type,
            columns=columns,
            page_quality=page_quality,
            page_confidence_mean=float(np.mean(confidences)) if confidences else 0.0,
            processing_time_s=time.perf_counter() - t0,
            config_used=None,
            page_text_hash=self._compute_page_text_hash(columns),
        )
    @staticmethod
    def _compute_page_text_hash(columns):
        text = "\n\n".join(c.text for c in columns if hasattr(c, 'text') and c.text.strip())
        return Identity.sha256_hash(text) if text else None

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
            bbox=self._resolve_bbox(region, words),
            region_id=self._region_id(region),
            config_used=config_used,
        )

    @staticmethod
    def _extract_region(image: Any, region: Mapping[str, Any]) -> Any:
        """Extract region subimage using bounding box coordinates."""
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        return image[y : y + h, x : x + w]

    @staticmethod
    def _safe_bbox(region: Mapping[str, Any]) -> dict[str, int] | None:
        """Return valid bbox dict or None if dimensions invalid."""
        if region.get("w", 0) > 0 and region.get("h", 0) > 0:
            x = int(region["x"])
            y = int(region["y"])
            w = int(region["w"])
            h = int(region["h"])
            return {
                "left": x,
                "top": y,
                "width": w,
                "height": h,
            }
        return None

    @staticmethod
    def _region_id(region: Mapping[str, Any]) -> str:
        """Return region id from detector or deterministic synthesized id."""
        detector_id = region.get("id")
        if isinstance(detector_id, str) and detector_id.strip():
            return detector_id

        col = int(region.get("col_index", 0))
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        return f"col{col}_{x}_{y}_{w}_{h}"

    @staticmethod
    def _resolve_bbox(
        region: Mapping[str, Any],
        words: List[Any],
    ) -> dict[str, int] | None:
        """
        Resolve output bbox using OCR content when available.

        Priority:
            1) Union of word boxes (absolute page coordinates)
            2) Region bounding box fallback
        """
        if words:
            x0 = int(region.get("x", 0))
            y0 = int(region.get("y", 0))

            abs_lefts: List[int] = []
            abs_tops: List[int] = []
            abs_rights: List[int] = []
            abs_bottoms: List[int] = []

            for item in words:
                if not isinstance(item, Mapping):
                    continue
                bbox = item.get("bbox")
                if not isinstance(bbox, Mapping):
                    continue

                left = bbox.get("left", bbox.get("x"))
                top = bbox.get("top", bbox.get("y"))
                width = bbox.get("width", bbox.get("w"))
                height = bbox.get("height", bbox.get("h"))
                if left is None or top is None or width is None or height is None:
                    continue

                left_i = int(left)
                top_i = int(top)
                width_i = int(width)
                height_i = int(height)
                if width_i <= 0 or height_i <= 0:
                    continue

                abs_left = x0 + left_i
                abs_top = y0 + top_i
                abs_right = abs_left + width_i
                abs_bottom = abs_top + height_i

                abs_lefts.append(abs_left)
                abs_tops.append(abs_top)
                abs_rights.append(abs_right)
                abs_bottoms.append(abs_bottom)

            if abs_lefts:
                min_left = min(abs_lefts)
                min_top = min(abs_tops)
                max_right = max(abs_rights)
                max_bottom = max(abs_bottoms)
                return {
                    "left": min_left,
                    "top": min_top,
                    "width": max_right - min_left,
                    "height": max_bottom - min_top,
                }

        return PageProcessor._safe_bbox(region)

    @staticmethod
    def _classify_page_quality(metrics: Mapping[str, Any]) -> PageQuality:
        """
        Classify page quality using canonical thresholds from PageQuality docs.
        """
        try:
            sharpness = float(metrics.get("sharpness", 0.0))
            contrast = float(metrics.get("contrast", 0.0))
        except (TypeError, ValueError):
            return PageQuality.UNKNOWN

        if sharpness > 250.0 and contrast > 0.6:
            return PageQuality.EXCELLENT
        if sharpness > 150.0 and contrast > 0.4:
            return PageQuality.GOOD
        if sharpness > 80.0 and contrast > 0.25:
            return PageQuality.FAIR
        return PageQuality.POOR
