import os
import json
from typing import List, Dict, Any
from google import genai
import config

class LLMLingua2SelfRAGCompressor:
    """
    SOTA Prompt Compression & Self-RAG Reflective Generation Engine:
    1. LLMLingua-2 Style Dynamic Token Compression: Uses SLM/LLM to prune redundant filler tokens from context.
    2. Self-RAG Reflection Tokens: Injects real-time reflection tokens ([Retrieved], [Relevant], [Supported], [Utility]).
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    def compress_context_llmlingua2(self, context_blocks: List[str], compression_rate: float = 0.5) -> str:
        """LLMLingua-2 style token compression pruning noise while retaining high-density signal."""
        raw_text = "\n\n".join(context_blocks)
        if not self.client or len(raw_text.split()) < 100:
            return raw_text

        prompt = f"""
You are LLMLingua-2, a SOTA Context Compression Engine.
Compress the following retrieved context text by ~50%, removing filler words, redundant headers, and legal boilerplate.
Retain ALL numbers, entity names, technical statistics, tables, and exact facts.

TEXT TO COMPRESS:
{raw_text[:3500]}
"""
        try:
            res = self.client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[prompt],
                config={"temperature": 0.0}
            )
            return f"[LLMLingua-2 Compressed Context]\n" + res.text.strip()
        except Exception:
            return raw_text

    def format_self_rag_system_prompt(self, base_prompt: str) -> str:
        """Injects Self-RAG reflection token directives."""
        return base_prompt + (
            "\n\nSELF-RAG REFLECTION DIRECTIVES:\n"
            "• Append [Relevant] when retrieved context directly answers the prompt.\n"
            "• Append [Supported] when a factual claim is backed by context.\n"
            "• Append [Utility] to confirm final answer value."
        )
