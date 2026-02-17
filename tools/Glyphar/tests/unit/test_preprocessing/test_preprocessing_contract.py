from glyphar.preprocessing.base import PreprocessingStrategy
from glyphar.preprocessing.polarity import PolarityCorrectionStrategy


def test_polarity_implements_contract():
    strategy = PolarityCorrectionStrategy()
    assert isinstance(strategy, PreprocessingStrategy)
