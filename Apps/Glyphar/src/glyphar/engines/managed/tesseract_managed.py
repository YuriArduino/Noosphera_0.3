"""
Managed Tesseract engine: compõe o motor puro com otimizações, cache, stats e fallback.

Mantém a API compatível com a versão antiga (recognize(image, config) -> dict).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

from glyphar.engines.base import OCREngine
from glyphar.engines.config_builder import TesseractConfigBuilder
from glyphar.engines.core.tesseract_core import TesseractCoreEngine
from glyphar.engines.fallback import apply_fallback_strategy
from glyphar.engines.optimizer import optimize_ocr_config
from glyphar.engines.stats import OCRStats
from glyphar.engines.user_files import UserFilesManager
from glyphar.engines.validation import validate_tessdata
from glyphar.models.config import OCRConfig
from glyphar.models.enums import PageQuality

UInt8Image = npt.NDArray[np.uint8]


class TesseractManagedEngine(OCREngine):
    """
    Engine gerenciado: recebe a configuração, otimiza, usa cache, chama o motor puro,
    processa resultado e aplica fallback caso necessário.
    """

    def __init__(
        self,
        tessdata_dir: str,
        languages: str = "por+eng",
        model_type: str = "fast",
        config: Optional[OCRConfig] = None,
    ) -> None:
        self.tessdata_dir = Path(tessdata_dir)
        self.languages = validate_tessdata(self.tessdata_dir, languages)
        self.model_type = model_type
        self.config = config or OCRConfig()

        # Set TESSDATA_PREFIX for any lower-level call that needs it.
        os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

        self.builder = TesseractConfigBuilder(self.tessdata_dir, model_type)
        self.user_files = UserFilesManager(model_type)
        self.stats = OCRStats()
        self.cache: Dict[str, Dict[str, Any]] = {}

        self.core = TesseractCoreEngine(self.languages)

    def recognize(self, image: UInt8Image, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        config = config or {}

        psm = config.get("psm", 3)
        quality_hint = self._normalize_quality(config.get("quality_hint", PageQuality.UNKNOWN))
        min_confidence = float(config.get("min_confidence", 0.0))

        optimized_config = optimize_ocr_config(psm, quality_hint, self.model_type)
        self.user_files.prepare()

        cache_key = self._compute_cache_key(
            image=image,
            optimized_config=optimized_config,
            min_confidence=min_confidence,
        )
        if cache_key in self.cache:
            self.stats.record_cache_hit()
            return dict(self.cache[cache_key])

        self.stats.record_cache_miss()

        # Config string for metadata/traceability and fallback calls.
        config_used = self.builder.build(
            optimized_config["psm"],
            optimized_config["oem"],
            optimized_config.get("extra", ""),
        )

        try:
            core_result = self.core.recognize(
                image,
                config={
                    "psm": optimized_config["psm"],
                    "oem": optimized_config["oem"],
                    "timeout": optimized_config.get("timeout"),
                },
            )
            result = self._post_process_core_result(core_result, min_confidence)
            result["config_used"] = config_used

        except RuntimeError as error:
            result = apply_fallback_strategy(
                image=image,
                languages=self.languages,
                config_builder=self.builder,
                start_time=start_time,
                error=str(error),
                timeout=int(optimized_config.get("timeout", 10)),
            )
            result["config_used"] = config_used

        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        self.stats.update(float(result.get("confidence", 0.0)), result["processing_time_ms"])

        if len(self.cache) < 1000:
            self.cache[cache_key] = dict(result)

        return result

    @staticmethod
    def _normalize_quality(value: Any) -> PageQuality:
        if isinstance(value, PageQuality):
            return value
        if isinstance(value, str):
            try:
                return PageQuality(value.lower())
            except ValueError:
                return PageQuality.UNKNOWN
        return PageQuality.UNKNOWN

    @staticmethod
    def _post_process_core_result(
        core_result: Dict[str, Any], min_confidence: float
    ) -> Dict[str, Any]:
        words_in = core_result.get("words", []) or []
        kept_words = []
        confidences = []
        text_parts = []

        for item in words_in:
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            try:
                conf = float(item.get("conf", -1.0))
            except (TypeError, ValueError):
                conf = -1.0

            if conf < min_confidence:
                continue

            kept_words.append(
                {
                    "text": text,
                    "confidence": conf,
                    "bbox": item.get("bbox"),
                }
            )
            text_parts.append(text)
            if conf >= 0:
                confidences.append(conf)

        full_text = " ".join(text_parts).strip()
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "text": full_text,
            "confidence": avg_conf,
            "words": kept_words,
            "word_count": len(kept_words),
            "char_count": len(full_text),
            "avg_word_confidence": avg_conf,
            "min_word_confidence": min(confidences) if confidences else 0.0,
            "max_word_confidence": max(confidences) if confidences else 0.0,
        }

    def _compute_cache_key(
        self,
        image: UInt8Image,
        optimized_config: Dict[str, Any],
        min_confidence: float,
    ) -> str:
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        return "|".join(
            [
                image_hash,
                self.languages,
                self.model_type,
                f"psm={optimized_config['psm']}",
                f"oem={optimized_config['oem']}",
                f"to={optimized_config.get('timeout')}",
                f"mc={min_confidence:.2f}",
            ]
        )
