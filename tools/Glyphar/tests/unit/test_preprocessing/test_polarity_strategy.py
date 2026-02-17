# test_polarity_strategy.py

import numpy as np
import cv2

from glyphar.preprocessing.polarity import PolarityCorrectionStrategy


def test_polarity_inversion_detection():
    """Test polarity inversion detection"""
    strategy = PolarityCorrectionStrategy()

    # Create synthetic inverted image (white text on black background)
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.putText(
        image,
        "TEST",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255),
        2,
        cv2.LINE_AA,
    )

    corrected = strategy.apply(image)

    assert corrected.dtype == np.uint8
    assert corrected.shape == image.shape

    # After correction background should become light
    assert np.mean(corrected) > np.mean(image)
