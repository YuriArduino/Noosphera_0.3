"""
OCR configuration optimization layer.

Adaptive strategy selection for Tesseract parameters based on document
characteristics (layout type, image quality). Enables speed/accuracy
trade-offs without manual tuning.

Public API:
    - ConfigOptimizer: Execution orchestrator (primary interface)
    - EngineConfig: Immutable configuration container (internal use)
    - ConfigStrategy: Pure strategy selector (advanced customization)
    - ImagePreprocessor: Preprocessing adapter (advanced customization)

Usage:
    >>> from optimization import ConfigOptimizer
    >>> optimizer = ConfigOptimizer(engine=tesseract_engine)
    >>> result = optimizer.find_optimal_config(image, "single", quality_metrics)
"""

from .config_optimizer import ConfigOptimizer
from .config_strategy import EngineConfig, ConfigStrategy
from .image_preprocessor import ImagePreprocessor

__all__ = [
    "ConfigOptimizer",
    "EngineConfig",
    "ConfigStrategy",
    "ImagePreprocessor",
]
