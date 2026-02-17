from pathlib import Path

import pytest

from glyphar.engines.user_files import UserFilesManager


def test_rejects_invalid_model_type() -> None:
    with pytest.raises(ValueError):
        UserFilesManager("ultra")


def test_prepare_creates_words_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(UserFilesManager, "DEFAULT_WORDS_PATH", tmp_path / "words.txt")
    monkeypatch.setattr(
        UserFilesManager,
        "DEFAULT_PATTERNS_PATH",
        tmp_path / "patterns.txt",
    )

    manager = UserFilesManager("fast")
    manager.prepare()

    words_path = Path(manager.words_file or "")
    assert words_path.exists()
    content = words_path.read_text(encoding="utf-8")
    assert "psicanálise" in content
    assert manager.patterns_file is None

    manager.cleanup()


def test_best_model_creates_patterns_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(UserFilesManager, "DEFAULT_WORDS_PATH", tmp_path / "words.txt")
    monkeypatch.setattr(
        UserFilesManager,
        "DEFAULT_PATTERNS_PATH",
        tmp_path / "patterns.txt",
    )

    manager = UserFilesManager("best")
    manager.prepare()

    assert Path(manager.words_file or "").exists()
    patterns_path = Path(manager.patterns_file or "")
    assert patterns_path.exists()
    assert r"\d{4}[a-z]?" in patterns_path.read_text(encoding="utf-8")

    manager.cleanup()


def test_context_manager_cleans_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(UserFilesManager, "DEFAULT_WORDS_PATH", tmp_path / "words.txt")
    monkeypatch.setattr(
        UserFilesManager,
        "DEFAULT_PATTERNS_PATH",
        tmp_path / "patterns.txt",
    )

    with UserFilesManager("best") as manager:
        words = Path(manager.words_file or "")
        patterns = Path(manager.patterns_file or "")
        assert words.exists()
        assert patterns.exists()

    assert not (tmp_path / "words.txt").exists()
    assert not (tmp_path / "patterns.txt").exists()
