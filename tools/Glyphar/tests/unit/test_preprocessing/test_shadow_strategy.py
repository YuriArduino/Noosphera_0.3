import numpy as np
import cv2

from glyphar.preprocessing.shadow import ShadowRemovalStrategy


def test_shadow_strategy_returns_uint8_grayscale():
    strategy = ShadowRemovalStrategy()

    # Imagem sintética com gradiente simulando sombra
    h, w = 300, 400
    gradient = np.tile(np.linspace(50, 200, w, dtype=np.uint8), (h, 1))

    # Simula texto
    cv2.putText(
        gradient,
        "Test",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,),
        3,
        cv2.LINE_AA,
    )

    result = strategy.apply(gradient)

    assert result.dtype == np.uint8
    assert len(result.shape) == 2
    assert result.shape == gradient.shape
