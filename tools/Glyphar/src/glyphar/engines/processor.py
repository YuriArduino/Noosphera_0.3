"""
Data processing module for OCR results from Tesseract.
"""

from typing import Dict, List, Any
import numpy as np


def process_ocr_data(
    data: Dict[str, List[Any]],
    min_confidence: float,
) -> Dict[str, Any]:
    """
    Transform raw Tesseract output into structured OCR result.

    Processing steps:
        1. Confidence filtering
        2. Bounding box normalization
        3. Spatial line reconstruction
        4. Statistical aggregation
    """
    words: List[Dict[str, Any]] = []
    confidences: List[float] = []
    total_chars = 0

    texts = data.get("text", [])
    confs = data.get("conf", [])

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        try:
            confidence = float(confs[i])
        except (ValueError, TypeError, IndexError):
            continue

        if confidence <= min_confidence:
            continue

        confidences.append(confidence)
        total_chars += len(text)

        words.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": {
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                },
            }
        )

    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": _reconstruct_text_lines(words),
        "confidence": avg_confidence,
        "words": words,
        "word_count": len(words),
        "char_count": total_chars,
        "avg_word_confidence": avg_confidence,
        "min_word_confidence": min(confidences) if confidences else 0.0,
        "max_word_confidence": max(confidences) if confidences else 0.0,
    }


def _reconstruct_text_lines(words: List[Dict[str, Any]]) -> str:
    """
    Reconstruct multi-line text using spatial clustering.
    """
    if not words:
        return ""

    # Dynamic bucket based on median height (more robust)
    heights = [w["bbox"]["height"] for w in words]
    median_height = np.median(heights)
    bucket_size = max(5, int(median_height * 0.7))

    lines: Dict[int, List[Dict[str, Any]]] = {}

    for word in words:
        y_bucket = int(word["bbox"]["top"] // bucket_size)
        lines.setdefault(y_bucket, []).append(word)

    output_lines = []

    for _, line_words in sorted(lines.items()):
        line_words.sort(key=lambda w: w["bbox"]["left"])
        output_lines.append(" ".join(w["text"] for w in line_words))

    return "\n".join(output_lines)
