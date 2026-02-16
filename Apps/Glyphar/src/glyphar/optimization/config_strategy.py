"""
OCR engine configuration strategy selection.

Pure strategy pattern implementation — no state, no side effects.
Decides optimal Tesseract parameters (PSM, preprocessing, scaling) based on:
    - Layout type (single/double column)
    - Image quality metrics (sharpness, contrast)

Design constraints:
    - Frozen dataclass (immutable configuration)
    - No engine dependencies (pure decision logic)
    - Deterministic outputs (same inputs → same config)
    - Conservative bias (avoid over-processing unless clearly justified)

Architectural improvements (v2):
    - Introduces threshold bands (low / medium / high)
    - Blur and contrast compete instead of strict hierarchy
    - Progressive scaling based on degradation level
    - Clean digital requires strong evidence (sharpness + contrast)

This reduces over-processing (e.g., unnecessary adaptive thresholding)
while preserving deterministic behavior.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """
    Immutable Tesseract engine configuration parameters.

    Represents optimal engine settings for a specific image context.
    Used exclusively by ConfigOptimizer — not exposed to external API.

    Attributes:
        pre_type: Preprocessing strategy name ("gray", "otsu", "adaptive")
        psm: Page Segmentation Mode (1-13, Tesseract-specific)
        scale: Upscaling factor (>1.0 for low-quality scans)
        oem: OCR Engine Mode (1=fast, 3=best)

    Notes:
        - pre_type is intentionally restricted to lightweight strategies.
        - scale should remain <= 1.5 to avoid excessive memory overhead.
    """

    pre_type: str
    psm: int
    scale: float
    oem: int = 3


class ConfigStrategy:
    """
    Pure strategy selector for OCR engine configuration.

    Stateless decision engine — analyzes context and returns optimal EngineConfig.
    No side effects, no I/O, no engine interaction.

    Decision model (v2):

        1. Strong clean-digital override
        2. Blur vs contrast dominance comparison
        3. Conservative fallback (grayscale)

    Threshold philosophy:
        - Avoid binary cliffs
        - Avoid aggressive preprocessing unless clearly dominant
        - Bias toward grayscale unless degradation is measurable

    Empirical assumptions:
        - Sharpness is Laplacian variance
        - Contrast is Michelson ratio (0.0–1.0)
    """

    # ---- Sharpness thresholds ----
    SHARPNESS_LOW = 50
    SHARPNESS_MEDIUM = 90
    SHARPNESS_HIGH = 150

    # ---- Contrast thresholds ----
    CONTRAST_LOW = 0.30
    CONTRAST_MEDIUM = 0.45
    CONTRAST_HIGH = 0.55

    @staticmethod
    def decide(layout_type: str, quality: dict) -> EngineConfig:
        """
        Select optimal engine configuration based on document characteristics.

        Args:
            layout_type: LayoutType value ("single", "double", etc.)
            quality: Quality metrics dict from QualityAssessor with keys:
                - is_clean_digital: bool
                - sharpness: float
                - contrast: float

        Returns:
            EngineConfig with optimal Tesseract parameters.

        Deterministic:
            Same (layout_type, quality) → same EngineConfig
        """

        is_clean = quality.get("is_clean_digital", False)
        sharpness = float(quality.get("sharpness", 0.0))
        contrast = float(quality.get("contrast", 0.0))

        # ------------------------------------------------------------------
        # 1. STRONG CLEAN DIGITAL OVERRIDE
        # Require high sharpness + high contrast to avoid false positives.
        # ------------------------------------------------------------------
        if (
            is_clean
            and sharpness >= ConfigStrategy.SHARPNESS_HIGH
            and contrast >= ConfigStrategy.CONTRAST_HIGH
        ):
            return EngineConfig(
                pre_type="gray",
                psm=3 if layout_type == "single" else 4,
                scale=1.0,
                oem=1,  # Fast mode (clean PDFs)
            )

        # ------------------------------------------------------------------
        # 2. Compute degradation scores
        # ------------------------------------------------------------------
        blur_score = max(0.0, ConfigStrategy.SHARPNESS_LOW - sharpness)
        contrast_score = max(0.0, ConfigStrategy.CONTRAST_LOW - contrast)

        # ------------------------------------------------------------------
        # 3. Blur dominant → adaptive (only if clearly worse)
        # ------------------------------------------------------------------
        if blur_score > contrast_score and sharpness < ConfigStrategy.SHARPNESS_MEDIUM:

            # Progressive scaling
            if sharpness < 35:
                scale = 1.5
            elif sharpness < 50:
                scale = 1.3
            else:
                scale = 1.2

            return EngineConfig(
                pre_type="adaptive",
                psm=6,  # Block mode more tolerant to noise
                scale=scale,
                oem=3,  # Highest accuracy
            )

        # ------------------------------------------------------------------
        # 4. Low contrast dominant → Otsu
        # ------------------------------------------------------------------
        if contrast < ConfigStrategy.CONTRAST_MEDIUM:

            return EngineConfig(
                pre_type="otsu",
                psm=11 if layout_type != "single" else 3,
                scale=1.2,
                oem=2,  # Balanced
            )

        # ------------------------------------------------------------------
        # 5. Conservative default
        # ------------------------------------------------------------------
        return EngineConfig(
            pre_type="gray",
            psm=3 if layout_type == "single" else 4,
            scale=1.0,
            oem=2,
        )
