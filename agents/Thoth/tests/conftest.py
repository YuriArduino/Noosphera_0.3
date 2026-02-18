"""
Pytest configuration and fixtures for Thoth tests.

Automatically loaded by pytest for all test modules.
"""

import sys
from pathlib import Path
import pytest


# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )
    config.addinivalue_line("markers", "llmstudio: marks tests that require LLMStudio running")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent.parent / "Test" / "Data"
