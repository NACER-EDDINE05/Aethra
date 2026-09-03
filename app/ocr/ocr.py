"""Screenshot OCR via Tesseract (optional dependency, degrades gracefully)."""

from __future__ import annotations

import io


class OCRError(RuntimeError):
    pass


def run_ocr(image_bytes: bytes, lang: str = "eng") -> str:
    """Extract text from a screenshot. Raises OCRError with setup guidance on failure."""
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise OCRError(
            "OCR dependencies are not installed. Install them with: "
            "pip install Pillow pytesseract  (and the Tesseract binary for your OS)"
        ) from exc

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        text = pytesseract.image_to_string(image, lang=lang)
    except Exception as exc:  # TesseractNotFoundError and image decode errors
        raise OCRError(
            f"OCR failed: {exc}. Make sure the Tesseract binary is installed and on PATH "
            f"(language pack '{lang}' must be available)."
        ) from exc

    return text.strip()
