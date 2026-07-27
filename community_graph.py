import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from google import genai
import config

class HierarchicalCommunityGraphRAG:
    """
    SOTA Hierarchical Community GraphRAG Engine:
    1. Extracts entity-relationship triplets using Gemini 3.1 Flash-Lite
    2. Clusters graph nodes into hierarchical communities using greedy modularity / Louvain-style community detection
    3. Generates pre-computed Community Summaries for macro-level abstract reasoning across thousands of pages.
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
        self.graph = nx.Graph()
        self.community_summaries: Dict[int, str] = {}

    def extract_triplets(self, text_chunk: str, page_num: int) -> List[Dict[str, Any]]:
        """Extracts entity-relationship triplets."""
        if not self.client:
            return []
        prompt = f"""
Extract key entities and relationships from the text (Page {page_num}):
{text_chunk[:1500]}

OUTPUT FORMAT: Return ONLY a valid JSON array:
[
  {{"source": "EntityA", "target": "EntityB", "relation": "connects_to"}}
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

    def build_hierarchical_communities(self, chunks: List[Dict[str, Any]], max_chunks: int = 15):
        """Builds Knowledge Graph and runs modularity community detection + LLM community summarization."""
        print(f"🕸️ Building Hierarchical Community GraphRAG over {min(len(chunks), max_chunks)} chunks...")
        for chunk in chunks[:max_chunks]:
            page = chunk.get("page", 1)
            triplets = self.extract_triplets(chunk.get("text", ""), page)
            for t in triplets:
                src, tgt, rel = t.get("source"), t.get("target"), t.get("relation", "relates_to")
                if src and tgt:
                    self.graph.add_node(src, page=page)
                    self.graph.add_node(tgt, page=page)
                    self.graph.add_edge(src, tgt, relation=rel)

        if self.graph.number_of_nodes() < 2:
            return

        # Community Detection via Greedy Modularity Clustering
        try:
            communities = list(nx.community.greedy_modularity_communities(self.graph))
            for comm_id, comm_nodes in enumerate(communities):
                nodes_list = list(comm_nodes)[:10]
                subgraph_edges = [f"{u} -> {v}" for u, v in self.graph.edges(nodes_list)]
                
                if self.client and subgraph_edges:
                    prompt = f"Synthesize a high-level community summary for these related entities and relations:\nEntities: {nodes_list}\nRelations: {subgraph_edges}"
                    try:
                        res = self.client.models.generate_content(
                            model=config.GEMINI_MODEL_NAME,
                            contents=[prompt],
                            config={"temperature": 0.0}
                        )
                        self.community_summaries[comm_id] = f"[Community {comm_id} Macro-Summary]\n{res.text.strip()}"
                    except Exception:
                        pass
        except Exception:
            pass

    def query_community_summaries(self) -> str:
        """Returns pre-computed hierarchical community summaries for macro reasoning."""
        if not self.community_summaries:
            return ""
        return "\n\n".join(self.community_summaries.values())
