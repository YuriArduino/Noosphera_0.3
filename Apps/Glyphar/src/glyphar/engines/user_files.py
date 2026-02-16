"""
Dynamic Tesseract configuration builder with caching
for optimized OCR performance.
"""

import tempfile
import os
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

    def __init__(self, model_type: str):
        """
        Initialize manager for given model type.

        Args:
            model_type: Engine profile ("fast", "standard", "best").
                Only "best" uses pattern dictionaries.
        """
        self.model_type = model_type
        self.words_file: Optional[str] = None
        self.patterns_file: Optional[str] = None

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
            self.words_file = self._write_to_tempfile(self.PSYCHOANALYTIC_TERMS)

        if self.model_type == "best" and not self.patterns_file:
            self.patterns_file = self._write_to_tempfile(self.CITATION_PATTERNS)

    def cleanup(self) -> None:
        """Remove temporary dictionary files."""
        for filepath in (self.words_file, self.patterns_file):
            if filepath and os.path.exists(filepath):
                os.unlink(filepath)
        self.words_file = None
        self.patterns_file = None

    @staticmethod
    def _write_to_tempfile(lines: List[str]) -> str:
        """Write lines to temporary file and return path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(lines))
            return f.name
