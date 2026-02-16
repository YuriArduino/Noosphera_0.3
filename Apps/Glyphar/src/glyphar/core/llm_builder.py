"""
LLM-optimized text formatting for post-OCR correction pipelines.

Generates structured text with explicit page boundaries and confidence metadata.
Enables LLM to:
    - Identify low-confidence regions requiring correction
    - Preserve document structure during correction
    - Contextualize corrections using neighboring pages

Design rationale:
    - Explicit page markers prevent LLM from merging/splitting pages incorrectly
    - Confidence scores guide LLM attention (low-confidence = higher correction priority)
    - Minimal formatting overhead (<2% size increase vs raw text)
"""

from typing import List
from models.page import PageResult


def build_llm_ready_text(
    pages_results: List[PageResult], separator: str = "\n\n"
) -> str:
    """
    Format OCR results for LLM ingestion with structural metadata.

    Output structure:
        === OCR RESULTS - N PAGES ===

        === PAGE 1 | Confidence: 92.3% ===
        [page text]

        === PAGE 2 | Confidence: 78.1% ===
        [page text]

        === END OF DOCUMENT ===

    Args:
        pages_results: List of PageResult instances from OCR pipeline.
        separator: String inserted between columns within a page.

    Returns:
        Single string with LLM-optimized formatting.

    LLM correction benefits:
        - Page boundaries preserved during correction
        - Low-confidence pages receive priority attention
        - Context maintained across page boundaries
        - Easy post-correction parsing (split by "=== PAGE X ===")

    Example usage:
        >>> text = build_llm_ready_text(pages_results)
        >>> corrected = llm.correct(text, system_prompt="Fix OCR errors while preserving structure")
    """
    parts = []
    parts.append(f"=== OCR RESULTS - {len(pages_results)} PAGES ===")

    for page in pages_results:
        header = (
            f"\n\n=== PAGE {page.page_number} | Confidence: "
            f"{page.page_confidence_mean:.1f}% ===\n"
        )
        parts.append(header)

        # Safely extract text from columns (handles empty/fallback pages)
        page_text = separator.join(col.text for col in page.columns if col.text.strip())
        parts.append(page_text)

    parts.append("\n=== END OF DOCUMENT ===")
    return "\n".join(parts)
