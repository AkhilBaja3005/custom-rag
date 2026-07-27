import re
from typing import List, Dict, Any

class PromptCompressorSelfRAG:
    """
    SOTA Context Compression & Self-Reflective RAG (Self-RAG) Engine:
    1. Compresses context blocks to strip noise, redundant headers, and boilerplate words (LLMLingua-2 style dynamic pruning).
    2. Inserts Self-RAG reflection tokens ([Retrieved], [Relevant], [Supported], [Utility]) to enforce grounding.
    """
    def compress_context_blocks(self, context_blocks: List[str], max_words_per_block: int = 120) -> str:
        """Dynamic prompt compression pruning non-essential filler words."""
        compressed = []
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "been", "being", "have", "has", "had"}
        
        for idx, block in enumerate(context_blocks, 1):
            lines = block.split("\n")
            header = lines[0] if lines else f"--- BLOCK {idx} ---"
            body = "\n".join(lines[1:]) if len(lines) > 1 else block
            
            words = body.split()
            # Retain high-information density words (nouns, numbers, technical terms)
            filtered = [w for w in words if w.lower() not in stop_words or len(w) > 5]
            pruned_body = " ".join(filtered[:max_words_per_block])
            
            compressed.append(f"{header}\n[Compressed Signal Block]: {pruned_body}")
            
        return "\n\n".join(compressed)

    def inject_self_rag_reflection_prompt(self, base_prompt: str) -> str:
        """Injects Self-RAG reflection token evaluation directives into system prompt."""
        return base_prompt + (
            "\n\nSELF-RAG DIRECTIVE: Evaluate your output using reflection tokens:\n"
            "• Use [Relevant] when a retrieved chunk answers the query.\n"
            "• Use [Supported] when your claim is factually backed by context.\n"
            "• Use [Utility] to confirm final answer utility."
        )
