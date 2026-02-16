"""
Document image quality assessment for adaptive OCR pipelines.

Provides quantitative metrics to optimize preprocessing strategy selection:
    - Clean digital documents → minimal preprocessing (speed)
    - Scanned/noisy documents → aggressive preprocessing (accuracy)

Public API:
    - QualityAssessor: Stateless quality assessment engine
"""

from .quality_assessor import QualityAssessor

__all__ = ["QualityAssessor"]
