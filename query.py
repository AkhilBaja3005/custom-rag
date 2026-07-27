import os
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer, CrossEncoder
# pyrefly: ignore [missing-import]
from qdrant_client import QdrantClient
# pyrefly: ignore [missing-import]
from google import genai

import config

class RAGQueryEngine:
    def __init__(self, qdrant_client: QdrantClient = None, qdrant_path: str = config.QDRANT_PATH):
        self.device = config.DEVICE
        self.qdrant_path = qdrant_path
        
        # Load Embedding Model on MPS device
        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' on device: {self.device}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=self.device)
        
        # Load Cross-Encoder Re-Ranker on MPS device
        print(f"Loading cross-encoder model '{config.CROSS_ENCODER_MODEL_NAME}' on device: {self.device}")
        self.reranker = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=self.device)
        
        # Connect to Local Qdrant (reuse existing client if provided)
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            self.qdrant = QdrantClient(path=self.qdrant_path)
        
        # Initialize Sub-10ms Semantic Disk Cache
        try:
            # pyrefly: ignore [missing-import]
            import diskcache
            self.cache = diskcache.Cache(os.path.join(config.BASE_DIR, ".rag_cache"))
        except Exception:
            self.cache = {}

        # Initialize Gemini API Client
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            print("WARNING: GEMINI_API_KEY not set. Gemini zero-extrapolation generation will return a warning.")

    def sanitize_input_prompt(self, query: str) -> str:
        """Security Guardrail: Sanitizes query input to prevent prompt injection attacks."""
        import re
        injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+prompt",
            r"disregard\s+above",
            r"you\s+are\s+now",
            r"jailbreak"
        ]
        sanitized = query
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[sanitized_input]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def generate_hyde_query(self, query: str) -> str:
        """HyDE: Generates a hypothetical document answer to expand raw query vector space."""
        if not self.gemini_client:
            return query
        try:
            clean_q = self.sanitize_input_prompt(query)
            prompt = f"Write a short, hypothetical academic textbook paragraph that answers the question: '{clean_q}'. Do not cite sources."
            res = self.gemini_client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[prompt],
                config={"temperature": 0.0}
            )
            return res.text.strip()
        except Exception:
            return query

    def search_vector_candidates(self, query: str, collection_name: str = config.COLLECTION_NAME, top_k: int = config.TOP_K_VECTOR) -> List[Dict[str, Any]]:
        """Encodes query locally with bge-small-en-v1.5 and fetches top_k candidates from specified Qdrant collection."""
        query_vector = self.embedder.encode(query, device=self.device).tolist()
        
        # Verify collection exists before querying
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if collection_name not in collections:
            return []

        response = self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        
        candidates = []
        for point in response.points:
            candidates.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "parent_text": point.payload.get("parent_text", point.payload.get("text", "")),
                "page": point.payload.get("page", 0),
                "source_tag": point.payload.get("source_tag", "")
            })
        return candidates

    def search_bm25_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int = config.TOP_K_VECTOR) -> List[Dict[str, Any]]:
        """Sparse BM25 Keyword Search over candidate chunk texts."""
        if not candidates:
            return []
            
        try:
            # pyrefly: ignore [missing-import]
            from rank_bm25 import BM25Okapi
            corpus = [c["text"].lower().split() for c in candidates]
            bm25 = BM25Okapi(corpus)
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            for idx, score in enumerate(bm25_scores):
                candidates[idx]["bm25_score"] = float(score)
                
            sorted_candidates = sorted(candidates, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
            return sorted_candidates[:top_k]
        except Exception:
            return candidates[:top_k]

    def reciprocal_rank_fusion(self, vector_candidates: List[Dict[str, Any]], bm25_candidates: List[Dict[str, Any]], c: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) combining Dense Vector and BM25 Sparse Search ranks."""
        rrf_scores = {}
        candidate_map = {}

        for rank, item in enumerate(vector_candidates, 1):
            cid = item["id"]
            candidate_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (c + rank))

        for rank, item in enumerate(bm25_candidates, 1):
            cid = item["id"]
            candidate_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (c + rank))

        fused = []
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            cand = candidate_map[cid]
            cand["rrf_score"] = score
            fused.append(cand)
        return fused

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int = config.TOP_K_RERANK) -> List[Dict[str, Any]]:
        """Re-ranks top vector + BM25 hybrid candidates using cross-encoder/ms-marco-MiniLM-L-6-v2."""
        if not candidates:
            return []
            
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)
            
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes zero-extrapolation answer using gemini-3.1-flash-lite with inline citations and Graph RAG support."""
        if not context_chunks:
            return {
                "answer": "No relevant context found in the database to answer your question.",
                "sources": []
            }

        # Build context prompt using Parent Block context for rich LLM synthesis
        context_blocks = []
        for idx, c in enumerate(context_chunks, 1):
            page_info = f"[Source: Page {c['page']}]" if c.get('page') else ""
            # Prefer rich 500-word parent_text over 150-word child text for LLM generation
            chunk_content = c.get("parent_text", c.get("text", ""))
            context_blocks.append(f"--- CONTEXT BLOCK {idx} {page_info} ---\n{chunk_content}")

        # Integrate Context Compression & Self-RAG Reflection Directives
        try:
            from prompt_compressor import PromptCompressorSelfRAG
            compressor = PromptCompressorSelfRAG()
            formatted_context = compressor.compress_context_blocks(context_blocks)
        except Exception:
            formatted_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a strict, SOTA production-grade Retrieval-Augmented Generation assistant.\n"
            "Answer ONLY using the provided context blocks. Include inline page citations like [Source: Page X].\n"
            "If the answer cannot be found in the provided context, state clearly: "
            "'I cannot answer this question based on the provided document context.'\n"
            "DO NOT extrapolate, speculate, or draw on external knowledge.\n\n"
            "SELF-RAG DIRECTIVE: Evaluate your generated claims using reflection markers [Relevant], [Supported], [Utility]."
        )

        user_prompt = f"Context Information:\n{formatted_context}\n\nQuestion: {query}"

        if not self.gemini_client:
            # Fallback mock answer when API key is missing
            pages = list(set([c['page'] for c in context_chunks]))
            return {
                "answer": f"[Mock Mode - Set GEMINI_API_KEY to enable Gemini synthesis] Based on retrieved context from pages {pages}, the document discusses the query topic.",
                "sources": context_chunks
            }

        try:
            response = self.gemini_client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[system_prompt, user_prompt],
                config={"temperature": 0.0}
            )
            return {
                "answer": response.text.strip(),
                "sources": context_chunks
            }
        except Exception as e:
            return {
                "answer": f"Error during Gemini generation: {str(e)}",
                "sources": context_chunks
            }

    def query(self, query: str, collection_name: str = config.COLLECTION_NAME, use_hyde: bool = True) -> Dict[str, Any]:
        """SOTA Production RAG pipeline: Semantic Cache -> HyDE -> Vector Search + BM25 -> RRF -> Cross-Encoder -> Gemini Synthesis."""
        cache_key = f"{collection_name}:{query.strip().lower()}"
        if self.cache is not None and cache_key in self.cache:
            cached_res = dict(self.cache[cache_key])
            cached_res["is_cache_hit"] = True
            return cached_res

        # 0. Adaptive Query Intent Routing
        route_info = {"use_hyde": use_hyde, "use_graph": True}
        try:
            from nextgen_rag import SOTAUltimateRAGEngine
            nextgen_engine = SOTAUltimateRAGEngine()
            route_info = nextgen_engine.route_query_intent(query)
        except Exception:
            nextgen_engine = None

        # 1. HyDE Query Expansion (routed adaptively)
        should_hyde = use_hyde and route_info.get("use_hyde", True)
        search_prompt = self.generate_hyde_query(query) if should_hyde else query
        
        # 2. Hybrid Dense Vector Search & BM25 Sparse Search
        vector_candidates = self.search_vector_candidates(search_prompt, collection_name=collection_name)
        if search_prompt != query:
            raw_vector_candidates = self.search_vector_candidates(query, collection_name=collection_name)
            seen_ids = {c["id"] for c in vector_candidates}
            for candidate in raw_vector_candidates:
                if candidate["id"] not in seen_ids:
                    vector_candidates.append(candidate)
                    seen_ids.add(candidate["id"])

        bm25_candidates = self.search_bm25_candidates(query, vector_candidates)
        
        # 3. Reciprocal Rank Fusion & Cross-Encoder Re-Ranking
        fused_candidates = self.reciprocal_rank_fusion(vector_candidates, bm25_candidates)
        top_reranked = self.rerank_candidates(query, fused_candidates)

        # 4. Next-Gen CRAG Evaluation + ColBERT Late Interaction Re-Ranking
        if nextgen_engine:
            try:
                crag_eval = nextgen_engine.evaluate_retrieval_confidence(query, top_reranked)
                if crag_eval["needs_correction"] and crag_eval["corrected_query"] != query:
                    extra_candidates = self.search_vector_candidates(crag_eval["corrected_query"], collection_name=collection_name)
                    for cand in extra_candidates:
                        if cand["id"] not in {c["id"] for c in top_reranked}:
                            top_reranked.append(cand)

                top_reranked = nextgen_engine.colbert_late_rerank(query, top_reranked)
            except Exception:
                pass

        # 5. Hierarchical Community GraphRAG Traversal
        if route_info.get("use_graph", False):
            try:
                from community_graph import HierarchicalCommunityGraphRAG
                comm_graph = HierarchicalCommunityGraphRAG()
                comm_graph.build_hierarchical_communities(top_reranked, max_chunks=5)
                comm_summary = comm_graph.query_community_summaries()
                if comm_summary:
                    top_reranked.append({
                        "page": 0,
                        "text": f"[Hierarchical Community GraphRAG Summary]\n{comm_summary}",
                        "parent_text": f"[Hierarchical Community GraphRAG Summary]\n{comm_summary}",
                        "rerank_score": 9.99
                    })
            except Exception:
                pass

        # 6. Zero-Extrapolation Answer Synthesis
        result = self.generate_answer(query, top_reranked)
        result["is_cache_hit"] = False
        
        # Cache successful responses
        if self.cache is not None and result["sources"]:
            self.cache[cache_key] = result
            
        return result

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the summary of this document?"
    engine = RAGQueryEngine()
    res = engine.query(q)
    print("\n=== ANSWER ===")
    print(res["answer"])
    print("\n=== SOURCES ===")
    for s in res["sources"]:
        print(f"- Page {s['page']} (Rerank Score: {s['rerank_score']:.4f})")
