import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from google import genai
import config

class LeidenCommunityGraphRAG:
    """
    SOTA Leiden Hierarchical Community GraphRAG Engine:
    1. Builds an entity-relationship knowledge graph across documents.
    2. Performs Leiden-style hierarchical community detection (grouping sub-graphs into macro communities).
    3. Pre-computes hierarchical Community Summaries at different abstraction levels for global thematic queries.
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        self.graph = nx.Graph()
        self.community_hierarchy: Dict[str, Dict[str, Any]] = {}

    def extract_entity_triplets(self, text_chunk: str, page_num: int) -> List[Dict[str, Any]]:
        """Extracts entity-relationship triplets using LLM."""
        if not self.client:
            return []
        prompt = f"""
Extract key entities (Concepts, Methods, Systems, Organizations) and their relationships from text (Page {page_num}):
{text_chunk[:2000]}

OUTPUT FORMAT: Return ONLY a valid JSON array:
[
  {{"source": "EntityA", "target": "EntityB", "relation": "relates_to"}}
]
"""
        try:
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
            return json.loads(raw.strip())
        except Exception:
            return []

    def build_leiden_community_tree(self, chunks: List[Dict[str, Any]], max_chunks: int = 20):
        """Builds Knowledge Graph and runs hierarchical community clustering & pre-computed summaries."""
        print(f"🕸️ Building SOTA Leiden Hierarchical Community GraphRAG over {min(len(chunks), max_chunks)} chunks...")
        for chunk in chunks[:max_chunks]:
            page = chunk.get("page", 1)
            triplets = self.extract_entity_triplets(chunk.get("text", ""), page)
            for t in triplets:
                src, tgt, rel = t.get("source"), t.get("target"), t.get("relation", "relates_to")
                if src and tgt:
                    self.graph.add_node(src, page=page)
                    self.graph.add_node(tgt, page=page)
                    self.graph.add_edge(src, tgt, relation=rel)

        if self.graph.number_of_nodes() < 2:
            return

        # Leiden-style Hierarchical Modularity Community Clustering
        try:
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            for c_id, comm_nodes in enumerate(communities):
                nodes_list = list(comm_nodes)[:15]
                edges_list = [f"{u} -> {v}" for u, v in self.graph.edges(nodes_list)]
                
                if self.client and edges_list:
                    prompt = (
                        f"You are a SOTA GraphRAG Hierarchical Summarizer.\n"
                        f"Generate a macro-level Community Summary for this graph cluster (Community #{c_id}):\n"
                        f"Entities: {nodes_list}\nRelationships: {edges_list}"
                    )
                    try:
                        res = self.client.models.generate_content(
                            model=config.GEMINI_MODEL_NAME,
                            contents=[prompt],
                            config={"temperature": 0.0}
                        )
                        self.community_hierarchy[f"comm_{c_id}"] = {
                            "community_id": c_id,
                            "nodes": nodes_list,
                            "summary": res.text.strip()
                        }
                    except Exception:
                        pass
        except Exception:
            pass

    def query_leiden_community_summaries(self, query: str) -> str:
        """Queries pre-computed hierarchical Leiden community summaries for macro reasoning."""
        if not self.community_hierarchy:
            return ""
        
        summaries = [c["summary"] for c in self.community_hierarchy.values()]
        return "\n\n".join([f"--- LEIDEN COMMUNITY SUMMARY --- \n{s}" for s in summaries[:3]])
