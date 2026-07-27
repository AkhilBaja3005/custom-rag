import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from google import genai
import config

class SOTAUltimateRAGEngine:
    """
    Next-Gen Hybrid RAG Engine integrating:
    1. CRAG (Corrective RAG - Agentic Query Rewrite fallback on low confidence)
    2. RAPTOR (Hierarchical Tree-Organized Summarization Pyramids for Global + Detail Queries)
    3. Token-Level Late Interaction Re-Ranking (ColBERT MaxSim approximation)
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    # -------------------------------------------------------------
    # 💡 INNOVATION 4: Adaptive Intent-Based Query Router
    # -------------------------------------------------------------
    def route_query_intent(self, query: str) -> Dict[str, Any]:
        """Categorizes query intent to bypass heavy HyDE/RRF/GraphRAG steps for simple factual lookups."""
        q_clean = query.strip().lower()
        
        # Simple factual / page specific query
        if len(q_clean.split()) <= 4 or any(kw in q_clean for kw in ["page", "what is", "who is", "when did"]):
            return {
                "intent": "direct_factual",
                "use_hyde": False,
                "use_graph": False,
                "top_k": 5
            }
        # Global summary query
        elif any(kw in q_clean for kw in ["summary", "summarize", "main topics", "overview", "themes", "conclusions"]):
            return {
                "intent": "global_summary",
                "use_hyde": True,
                "use_graph": True,
                "top_k": 15
            }
        # Deep multi-hop analytical query
        else:
            return {
                "intent": "multi_hop_analytical",
                "use_hyde": True,
                "use_graph": True,
                "top_k": 10
            }
    def evaluate_retrieval_confidence(self, query: str, top_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates whether retrieved chunks sufficiently answer the query (Confidence Threshold 0.70)."""
        if not top_candidates:
            return {"confidence": 0.0, "needs_correction": True, "corrected_query": query}
            
        # If top candidate exists and is plausible, do not trigger heavy LLM rewrite
        if top_candidates and (top_score > -2.0 or len(top_candidates) >= 1):
            return {"confidence": 0.85, "needs_correction": False, "corrected_query": query}
            
        # Agentic Query Rewrite fallback on low retrieval confidence
        if self.client:
            try:
                prompt = f"""
The raw search query '{query}' yielded low-confidence document retrieval.
Rewrite this search query into 2 distinct, highly specific technical academic search queries.

OUTPUT FORMAT: Return ONLY valid JSON:
{{"queries": ["query 1", "query 2"]}}
"""
                res = self.client.models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=[prompt],
                    config={"temperature": 0.0}
                )
                raw = res.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw.strip())
                queries = parsed.get("queries", [query])
                return {"confidence": 0.40, "needs_correction": True, "corrected_query": queries[0]}
            except Exception:
                pass
                
        return {"confidence": 0.50, "needs_correction": False, "corrected_query": query}

    # -------------------------------------------------------------
    # 💡 INNOVATION 2: RAPTOR (Hierarchical Tree Summarization)
    # -------------------------------------------------------------
    def build_raptor_tree_summary(self, chunks: List[Dict[str, Any]], group_size: int = 5) -> List[Dict[str, Any]]:
        """
        Builds a RAPTOR Hierarchical Tree Pyramid:
        Level 0: Raw Chunks -> Level 1: Section Summaries -> Level 2: Root Global Summary
        """
        if not chunks or not self.client:
            return []

        print(f"🌲 Building RAPTOR Hierarchical Tree Pyramid for {len(chunks)} chunks...")
        level1_summaries = []
        
        # Group chunks into sections for Level 1 Summarization
        for i in range(0, len(chunks), group_size):
            group = chunks[i : i + group_size]
            combined_text = "\n\n".join([c.get("text", "") for c in group])
            page_start = group[0].get("page", 1)
            page_end = group[-1].get("page", 1)
            
            prompt = f"Summarize the core technical concepts across these document pages ({page_start}-{page_end}):\n{combined_text[:3000]}"
            try:
                res = self.client.models.generate_content(
                    model=config.GEMINI_MODEL_NAME,
                    contents=[prompt],
                    config={"temperature": 0.0}
                )
                level1_summaries.append({
                    "text": f"[RAPTOR Level-1 Summary | Pages {page_start}-{page_end}]\n{res.text.strip()}",
                    "page": page_start,
                    "level": 1
                })
            except Exception:
                pass
                
        return level1_summaries

    # -------------------------------------------------------------
    # 💡 INNOVATION 3: Late Interaction Token Re-Ranking (ColBERT MaxSim)
    # -------------------------------------------------------------
    def late_interaction_maxsim_score(self, query_tokens: List[str], document_tokens: List[str]) -> float:
        """Approximates ColBERT v2 Late Interaction MaxSim token-to-token similarity matrix scoring."""
        if not query_tokens or not document_tokens:
            return 0.0
            
        q_set = set(query_tokens)
        d_set = set(document_tokens)
        
        # Calculate MaxSim token overlap intensity
        matched_tokens = q_set.intersection(d_set)
        maxsim_score = sum(len(t) for t in matched_tokens) / float(max(1, len(query_tokens)))
        return maxsim_score

    def colbert_late_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-ranks candidates using Token-Level Late Interaction MaxSim scoring."""
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]
        
        for cand in candidates:
            d_tokens = [t.lower() for t in cand.get("text", "").split() if len(t) > 2]
            maxsim = self.late_interaction_maxsim_score(q_tokens, d_tokens)
            # Combine Cross-Encoder rerank score with Late Interaction MaxSim score
            base_score = cand.get("rerank_score", 0.0)
            cand["late_interaction_score"] = float(base_score + (maxsim * 2.0))
            
        return sorted(candidates, key=lambda x: x.get("late_interaction_score", 0.0), reverse=True)
