import os
import json
import networkx as nx
from typing import List, Dict, Any
from google import genai
import config

class GraphRAGEngine:
    """Lightweight, memory-efficient Graph RAG implementation for entity and relationship extraction."""
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        self.graph = nx.Graph()

    def extract_entities_and_relations(self, text_chunk: str, page_num: int) -> List[Dict[str, Any]]:
        """Uses Gemini Flash-Lite to extract key entities and relationships from text."""
        if not self.client:
            return []

        prompt = f"""
Extract key entities (Concepts, Methods, Algorithms, Authors, Systems) and their relationships from the text below.

Text (Page {page_num}):
{text_chunk[:1500]}

OUTPUT FORMAT: Return ONLY a valid JSON array of objects with keys "source", "target", and "relation":
[
  {{"source": "EntityA", "target": "EntityB", "relation": "uses / refutes / evaluates"}}, ...
]
"""
        try:
            response = self.client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[prompt],
                config={"temperature": 0.0}
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            return data
        except Exception:
            return []

    def build_graph_from_chunks(self, chunks: List[Dict[str, Any]], max_chunks: int = 20):
        """Builds a NetworkX knowledge graph from extracted text chunks."""
        print(f"🕸️ Building Graph RAG knowledge network from top {min(len(chunks), max_chunks)} chunks...")
        for chunk in chunks[:max_chunks]:
            page = chunk.get("page", 1)
            text = chunk.get("text", "")
            triplets = self.extract_entities_and_relations(text, page)
            for t in triplets:
                src = t.get("source")
                tgt = t.get("target")
                rel = t.get("relation", "connected_to")
                if src and tgt:
                    self.graph.add_node(src, type="Entity", page=page)
                    self.graph.add_node(tgt, type="Entity", page=page)
                    self.graph.add_edge(src, tgt, relation=rel, page=page)

    def query_graph_context(self, entity: str) -> str:
        """Retrieves sub-graph context around a specific entity for multi-hop graph reasoning."""
        if entity not in self.graph:
            return ""
        
        neighbors = list(self.graph.neighbors(entity))
        lines = []
        for n in neighbors:
            edge_data = self.graph.get_edge_data(entity, n)
            rel = edge_data.get("relation", "related to")
            lines.append(f"- ({entity}) --[{rel}]--> ({n}) [Source: Page {edge_data.get('page')}]")
        return "\n".join(lines)

if __name__ == "__main__":
    engine = GraphRAGEngine()
    sample_text = "Fuzzy Commitments offer insufficient protection to biometric templates produced by deep learning. Danny Keller and Margarita Osadchy evaluated reconstruction attacks on facial recognition systems."
    triplets = engine.extract_entities_and_relations(sample_text, page_num=1)
    print("Extracted Graph Triplets:", json.dumps(triplets, indent=2))
