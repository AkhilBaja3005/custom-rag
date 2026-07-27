import os
import json
import base64
import fitz  # PyMuPDF for page image rendering
from typing import List, Dict, Any, Optional
from google import genai
import config

class VisionNativeColPaliParser:
    """
    SOTA Vision-Native Document Processor:
    Renders PDF pages directly as high-resolution visual images (ColPali / Visual Patch Style)
    and passes page images to Gemini 3.1 Flash-Lite VLM to extract structured, layout-aware
    Markdown preserving multi-column tables, charts, infographics, and visual hierarchy natively.
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def render_page_image_base64(self, page: fitz.Page, dpi: int = 150) -> str:
        """Renders a PDF page to a high-resolution PNG image and converts to base64."""
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def parse_page_vision_native(self, doc: fitz.Document, page_num: int) -> str:
        """Visual VLM parsing of a PDF page image to preserve tables, charts, and layout."""
        if not self.client:
            # Fallback to standard text extraction if API key missing
            return doc[page_num].get_text("text")

        try:
            page = doc[page_num]
            img_b64 = self.render_page_image_base64(page)

            prompt = (
                "You are an expert Vision-Language Document Parser (ColPali / Layout-Aware VLM).\n"
                "Parse this PDF page image directly into high-fidelity structured Markdown.\n"
                "• Preserve multi-column table alignment using Markdown tables.\n"
                "• Describe charts, diagrams, and visual infographics in detail.\n"
                "• Do NOT drop any text, footnotes, or mathematical equations."
            )

            # Pass raw visual page image to Gemini VLM
            response = self.client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    prompt
                ],
                config={"temperature": 0.0}
            )
            return response.text.strip()
        except Exception:
            return doc[page_num].get_text("text")
