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
        
        # Initialize Gemini API Client
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            print("WARNING: GEMINI_API_KEY not set. Gemini zero-extrapolation generation will return a warning.")

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
                "page": point.payload.get("page", 0),
                "source_tag": point.payload.get("source_tag", "")
            })
        return candidates

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int = config.TOP_K_RERANK) -> List[Dict[str, Any]]:
        """Re-ranks top vector candidates using cross-encoder/ms-marco-MiniLM-L-6-v2."""
        if not candidates:
            return []
            
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)
            
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesizes zero-extrapolation answer using gemini-3.1-flash-lite with inline citations."""
        if not context_chunks:
            return {
                "answer": "No relevant context found in the database to answer your question.",
                "sources": []
            }

        # Build context prompt with page citations
        formatted_context = ""
        for idx, chunk in enumerate(context_chunks, 1):
            formatted_context += f"--- CONTEXT BLOCK {idx} [Source: Page {chunk['page']}] ---\n{chunk['text']}\n\n"

        system_prompt = (
            "You are a strict, production-grade Retrieval-Augmented Generation assistant.\n"
            "Answer ONLY using the provided context blocks. Include inline page citations like [Source: Page X].\n"
            "If the answer cannot be found in the provided context, state clearly: "
            "'I cannot answer this question based on the provided document context.'\n"
            "DO NOT extrapolate, speculate, or draw on external knowledge."
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

    def query(self, query: str, collection_name: str = config.COLLECTION_NAME) -> Dict[str, Any]:
        """Full end-to-end RAG pipeline: vector search -> reranking -> LLM synthesis."""
        vector_candidates = self.search_vector_candidates(query, collection_name=collection_name)
        top_reranked = self.rerank_candidates(query, vector_candidates)
        result = self.generate_answer(query, top_reranked)
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
