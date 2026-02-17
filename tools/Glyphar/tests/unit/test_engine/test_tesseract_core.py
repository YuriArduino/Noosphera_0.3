import numpy as np
import pytest
from pytesseract import TesseractError

from glyphar.engines.core.tesseract_core import TesseractCoreEngine


def fake_output_dict() -> dict:
    return {
        "level": [1, 2, 3],
        "page_num": [1, 1, 1],
        "block_num": [1, 1, 1],
        "par_num": [1, 1, 1],
        "line_num": [1, 1, 1],
        "word_num": [1, 2, 3],
        "left": [0, 10, 30],
        "top": [0, 5, 5],
        "width": [10, 8, 12],
        "height": [10, 8, 12],
        "conf": ["95", "85", "-1"],
        "text": ["Hello", "world", ""],
    }


def test_core_successful_recognition(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractCoreEngine("eng")
    img = np.zeros((50, 200, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "pytesseract.image_to_data",
        lambda *args, **kwargs: fake_output_dict(),
    )

    result = engine.recognize(img, {"psm": 3})

    assert isinstance(result, dict)
    assert result["text"] == "Hello world"
    assert pytest.approx(result["confidence"], rel=1e-2) == (95.0 + 85.0) / 2
    assert isinstance(result["words"], list)
    assert result["words"][0]["text"] == "Hello"
    assert result["words"][0]["conf"] == 95.0


def test_core_raises_on_tesseract_error(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = TesseractCoreEngine("eng")
    img = np.zeros((50, 200), dtype=np.uint8)

    def raise_error(*args, **kwargs):
        raise TesseractError(1, "boom")

    monkeypatch.setattr("pytesseract.image_to_data", raise_error)

    with pytest.raises(RuntimeError):
        engine.recognize(img, {"psm": 3})
