"""Utilities for managing Tesseract user files used by managed engines."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import List, Optional


class UserFilesManager:
    """
    Manages Tesseract user dictionaries for domain-specific vocabulary.

    Specialized for psychoanalytic terminology to improve recognition of:
        - Proper names (Freud, Lacan, Winnicott)
        - Technical terms (inconsciente, recalque, transferência)
        - French/German loanwords common in psychoanalytic texts

    Design trade-offs:
        - Temporary files (auto-deleted on cleanup) vs persistent dictionaries
        - Small curated list (14 terms) vs large generic dictionaries
        - Portuguese-focused vs multi-language support

    Example:
        >>> manager = UserFilesManager("fast")
        >>> manager.prepare()  # Creates /tmp/tesseract_user_words.txt
        >>> # Tesseract automatically loads this when --user-words is set
    """

    PSYCHOANALYTIC_TERMS: List[str] = [
        "psicanálise",
        "Freud",
        "Lacan",
        "inconsciente",
        "transferência",
        "recalque",
        "sintoma",
        "gozo",
        "objeto a",
        "Édipo",
        "Jacques",
        "Sigmund",
        "Winnicott",
        "Bion",
    ]

    CITATION_PATTERNS: List[str] = [
        r"\d{4}[a-z]?",  # Years with optional letter (1998a)
        r"[A-Z]\.[A-Z]\.",  # Initials (S.F.)
        r"\d+-\d+",  # Page ranges (45-67)
        r"p\. \d+",  # Page references (p. 123)
    ]

    SUPPORTED_MODEL_TYPES = {"fast", "standard", "best"}

    DEFAULT_WORDS_PATH = Path("/tmp/tesseract_user_words.txt")
    DEFAULT_PATTERNS_PATH = Path("/tmp/tesseract_user_patterns.txt")

    def __init__(self, model_type: str):
        """
        Initialize manager for given model type.

        Args:
            model_type: Engine profile ("fast", "standard", "best").
                Only "best" uses pattern dictionaries.
        """
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_MODEL_TYPES))
            raise ValueError(
                f"Invalid model_type '{model_type}'. Supported values: {supported}."
            )

        self.model_type = model_type
        self.words_file: Optional[str] = None
        self.patterns_file: Optional[str] = None
        self._cleanup_registered = False
        self._register_cleanup()

    def prepare(self) -> None:
        """
        Create temporary dictionary files for Tesseract consumption.

        Files created:
            - /tmp/tesseract_user_words.txt: Psychoanalytic terms
            - /tmp/tesseract_user_patterns.txt: Citation patterns (best model only)

        Note:
            Files persist until explicit cleanup() or process termination.
            Tesseract must be configured with --user-words/--user-patterns to use them.
        """
        if not self.words_file:
            self.words_file = self._write_to_file(
                self.DEFAULT_WORDS_PATH,
                self.PSYCHOANALYTIC_TERMS,
            )

        if self.model_type == "best" and not self.patterns_file:
            self.patterns_file = self._write_to_file(
                self.DEFAULT_PATTERNS_PATH,
                self.CITATION_PATTERNS,
            )

    def cleanup(self) -> None:
        """Remove temporary dictionary files."""
        for filepath in (self.words_file, self.patterns_file):
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
        self.words_file = None
        self.patterns_file = None
        self._unregister_cleanup()

    def __enter__(self) -> "UserFilesManager":
        """Context manager support to guarantee temp-file lifecycle safety."""
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False

    def _register_cleanup(self) -> None:
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True

    def _unregister_cleanup(self) -> None:
        if self._cleanup_registered:
            try:
                atexit.unregister(self.cleanup)
            except Exception:
                # Best-effort unregister. Non-critical across Python versions.
                pass
            self._cleanup_registered = False

    @staticmethod
    def _write_to_file(path: Path, lines: List[str]) -> str:
        """Write lines to a path and return the resulting file path as string."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)
