"""
Managed Tesseract engine: compõe o motor puro com otimizações, cache, stats e fallback.

Mantém a API compatível com a versão antiga (recognize(image, config) -> dict).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional
from pathlib import Path
import hashlib

import numpy as np
import numpy.typing as npt

from glyphar.engines.core.tesseract_core import TesseractCoreEngine
from glyphar.engines.base import OCREngine
from glyphar.engines.validation import validate_tessdata
from glyphar.engines.config_builder import TesseractConfigBuilder
from glyphar.engines.user_files import UserFilesManager
from glyphar.engines.optimizer import optimize_ocr_config
from glyphar.engines.processor import process_ocr_data
from glyphar.engines.fallback import apply_fallback_strategy
from glyphar.engines.stats import OCRStats
from glyphar.models.enums import PageQuality
from glyphar.models.config import OCRConfig

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

        # Set TESSDATA_PREFIX for any lower-level call that needs it
        import os

        os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

        # components
        self.builder = TesseractConfigBuilder(self.tessdata_dir, model_type)
        self.user_files = UserFilesManager(model_type)
        self.stats = OCRStats()
        self.cache: Dict[str, Dict[str, Any]] = {}

        # core motor puro
        self.core = TesseractCoreEngine(self.languages)

    def recognize(self, image: UInt8Image, config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        psm = config.get("psm", 3)
        quality_hint = config.get("quality_hint", PageQuality.UNKNOWN)
        min_confidence = float(config.get("min_confidence", 0.0))

        optimized_config = optimize_ocr_config(psm, quality_hint, self.model_type)
        self.user_files.prepare()

        cache_key = self._compute_cache_key(image, optimized_config)
        if cache_key in self.cache:
            self.stats.cache_hits += 1
            # ensure the cached entry is a fresh copy to avoid mutations
            return dict(self.cache[cache_key])

        self.stats.cache_misses += 1

        # build a builder-based config to pass to core (psm/oem/timeout)
        tconf = self.builder.build(
            optimized_config["psm"],
            optimized_config["oem"],
            optimized_config.get("extra", ""),
        )

        try:
            # core raises RuntimeError on Tesseract issues
            core_result = self.core.recognize(
                image,
                config={
                    "psm": optimized_config["psm"],
                    "oem": optimized_config["oem"],
                    "timeout": optimized_config.get("timeout"),
                },
            )
            # process_ocr_data handles domain-specific aggregation, filtering by min_confidence
            result = process_ocr_data(core_result, min_confidence)

        except RuntimeError as error:
            result = apply_fallback_strategy(
                image=image,
                languages=self.languages,
                config_builder=self.builder,
                start_time=start_time,
                error=str(error),
            )

        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        self.stats.update(result.get("confidence", 0.0), result["processing_time_ms"])

        if len(self.cache) < 1000:
            self.cache[cache_key] = dict(result)

        return result

    def _compute_cache_key(self, image: UInt8Image, config: Dict[str, Any]) -> str:
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        return f"{image_hash}_{config['psm']}_{config['oem']}"
