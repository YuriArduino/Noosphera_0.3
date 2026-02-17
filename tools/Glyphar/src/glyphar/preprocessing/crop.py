"""
Smart cropping strategy based on structural text density analysis.

Detects text-bearing regions using binarized density projection rather than
raw intensity averaging. Crops vertical margins while preserving document
structure and avoiding aggressive trimming.

Design rationale:
    - Uses grayscale normalization for deterministic behavior
    - Converts to structural binary mask for robust text detection
    - Measures per-row dark pixel density instead of mean brightness
    - Applies percentile-based adaptive thresholding
    - Crops only when meaningful area reduction is achieved

Trade-offs:
    + More robust to background noise and uneven illumination
    + Better alignment with downstream OCR expectations
    + Deterministic and O(n) complexity
    - Slightly higher computation than mean projection
    - Still heuristic for extremely degraded documents

Example:
    >>> strategy = SmartCropStrategy(padding=15)
    >>> cropped = strategy.apply(page_image)
"""

import numpy as np
import numpy.typing as npt

from .grayscale import GrayscaleStrategy
from .base import PreprocessingStrategy


# pylint: disable=too-few-public-methods,no-member
class SmartCropStrategy(PreprocessingStrategy):
    """
    Crops vertical margins based on structural text density.

    Instead of using mean intensity, this strategy measures the proportion
    of dark pixels per row after adaptive thresholding, improving robustness
    to uneven backgrounds and scanned artifacts.

    Attributes:
        padding: Extra pixels added to crop boundaries.
        percentile: Percentile used to determine global dark threshold.
        min_content_ratio: Minimum fraction of rows containing content.
        min_crop_gain: Minimum vertical reduction ratio required to crop.
    """

    def __init__(
        self,
        padding: int = 10,
        percentile: float = 20.0,
        min_content_ratio: float = 0.05,
        min_crop_gain: float = 0.03,
    ) -> None:
        """
        Initialize smart cropping strategy.

        Args:
            padding:
                Extra pixels added to top and bottom crop boundaries.
                Prevents clipping ascenders/descenders.
            percentile:
                Percentile used to determine dark pixel threshold globally.
                Lower values = stricter dark detection.
            min_content_ratio:
                Minimum fraction of rows classified as containing content.
                Prevents cropping sparse layouts.
            min_crop_gain:
                Minimum proportion of vertical reduction required to perform
                cropping (e.g., 0.03 = must remove at least 3% of height).
        """
        self.padding = padding
        self.percentile = percentile
        self.min_content_ratio = min_content_ratio
        self.min_crop_gain = min_crop_gain
        self.grayscale = GrayscaleStrategy()

    def apply(
        self,
        image: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """
        Crop image to content region using structural density analysis.

        Algorithm:
            1. Convert to grayscale (deterministic normalization)
            2. Compute global dark threshold via percentile
            3. Generate binary mask of dark pixels
            4. Compute per-row dark pixel density
            5. Identify content rows
            6. Validate density ratio and minimum crop gain
            7. Crop with padding

        Returns:
            Cropped image (uint8) or original if conditions not met.

        Safety rules:
            - Never crop sparse pages
            - Never crop if vertical reduction is negligible
            - Always preserve uint8 contract
        """
        gray = self.grayscale.apply(image)
        h, _ = gray.shape

        # Step 1: Determine adaptive dark threshold
        dark_threshold = np.percentile(gray, self.percentile)

        # Step 2: Structural binary mask (True = likely text pixel)
        binary_mask = gray < dark_threshold

        # Step 3: Per-row density of dark pixels
        row_density = np.mean(binary_mask, axis=1)

        # Step 4: Identify rows containing meaningful content
        content_rows = row_density > 0.01  # at least 1% dark pixels

        # Sparse page safeguard
        if content_rows.mean() < self.min_content_ratio:
            return image

        indices = np.where(content_rows)[0]
        if indices.size == 0:
            return image

        top = max(0, int(indices[0]) - self.padding)
        bottom = min(h, int(indices[-1]) + self.padding)

        crop_height = bottom - top
        crop_ratio = 1.0 - (crop_height / h)

        # Only crop if meaningful gain
        if crop_ratio < self.min_crop_gain:
            return image

        cropped = image[top:bottom, :]

        return cropped.astype(np.uint8)
