import os
import io
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Redirect C-level stderr (file descriptor 2) to devnull to completely block MuPDF C-library syntax prints
try:
    c_stderr_fd = 2
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, c_stderr_fd)
    os.close(devnull_fd)
except Exception:
    pass

import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False)
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from google import genai
from google.genai import types

import config

class MultiParserIngestionEngine:
    def __init__(self, pdf_path: str, collection_name: str = config.COLLECTION_NAME, qdrant_client: QdrantClient = None, qdrant_path: str = config.QDRANT_PATH):
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        self.device = config.DEVICE
        
        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' on device: {self.device}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=self.device)
        self.vector_dim = self.embedder.get_embedding_dimension()
        
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            self.qdrant = QdrantClient(path=self.qdrant_path)
            
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            from qdrant_client.models import HnswConfigDiff
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100)  # SOTA HNSW Graph Indexing
            )

    def _extract_and_chunk_page(self, doc: fitz.Document, page_num: int):
        """In-memory extraction with Contextual Chunk Prepending."""
        try:
            page = doc[page_num]
            text = page.get_text("text")
            if not text or not text.strip():
                return []
                
            content = text.strip()
            # Contextual Header Prepending (Doc Name + Page Meta)
            doc_name = os.path.basename(self.pdf_path)
            context_header = f"[Doc: {doc_name} | Page {page_num + 1}]\n"
            
            widgets = page.widgets()
            annots = page.annots()
            if widgets or annots:
                meta = []
                if widgets:
                    for w in widgets:
                        if w.field_value:
                            meta.append(f"FormField '{w.field_name}': {w.field_value}")
                if annots:
                    for a in annots:
                        c = a.info.get("content")
                        if c:
                            meta.append(f"Annotation '{a.info.get('title', 'Note')}': {c}")
                if meta:
                    content += "\n[Metadata & Form Fields]\n" + "\n".join(meta)

            # Semantic Sentence Boundary Chunking:
            # Splits text at natural sentence boundaries (.!?) rather than hard token cuts
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            chunks = []
            curr_words = []
            curr_len = 0
            
            for sentence in sentences:
                s_words = sentence.split()
                if not s_words:
                    continue
                if curr_len + len(s_words) > 450:
                    parent_text = context_header + " ".join(curr_words)
                    # Create 150-word semantic child window
                    child_words = curr_words[:150]
                    child_text = context_header + " ".join(child_words)
                    chunks.append({
                        "text": child_text,
                        "parent_text": parent_text,
                        "page": page_num + 1
                    })
                    # Keep overlap for continuity
                    curr_words = curr_words[300:] + s_words
                    curr_len = len(curr_words)
                else:
                    curr_words.extend(s_words)
                    curr_len += len(s_words)
                    
            if curr_words and len(curr_words) > 20:
                parent_text = context_header + " ".join(curr_words)
                child_text = context_header + " ".join(curr_words[:150])
                chunks.append({
                    "text": child_text,
                    "parent_text": parent_text,
                    "page": page_num + 1
                })
            return chunks
        except Exception:
            return []

    def process_pdf_streaming(self, progress_callback=None):
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        
        print(f"🚀 Starting SAFE HIGH-SPEED ingestion for '{self.pdf_path}' ({total_pages} pages)...")
        batch_size = 50  # Safe streaming batch size (50 pages) to strictly prevent Unified RAM spikes
        point_id_counter = int(time.time() * 1000)
        
        total_chunks_indexed = 0
        start_time = time.time()

        for start_idx in range(0, total_pages, batch_size):
            end_idx = min(start_idx + batch_size, total_pages)
            if progress_callback:
                progress_callback(start_idx + 1, end_idx, total_pages)
                
            batch_chunks = []
            for p in range(start_idx, end_idx):
                chunks = self._extract_and_chunk_page(doc, p)
                if chunks:
                    batch_chunks.extend(chunks)
                
            if batch_chunks:
                texts = [c["text"] for c in batch_chunks]
                # Safe MPS GPU matrix encoding batch size (64)
                embeddings = self.embedder.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    device=self.device
                )
                
                points = [
                    PointStruct(
                        id=point_id_counter + idx,
                        vector=emb.tolist(),
                        payload={
                            "text": chunk["text"],
                            "parent_text": chunk.get("parent_text", chunk["text"]),
                            "page": chunk["page"],
                            "source_tag": f"[Source: Page {chunk['page']}]"
                        }
                    )
                    for idx, (chunk, emb) in enumerate(zip(batch_chunks, embeddings))
                ]
                point_id_counter += len(points)
                    
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                total_chunks_indexed += len(points)

            # Explicit Garbage Collection & GPU Memory Flush to prevent RAM/VRAM overflow
            import gc
            gc.collect()
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            elif self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        doc.close()
        duration = time.time() - start_time
        pages_per_sec = total_pages / duration if duration > 0 else 0
        print(f"\n⚡ INGESTION COMPLETED: {total_pages} pages ({total_chunks_indexed} chunks) in {duration:.2f}s ({pages_per_sec:.2f} pages/sec).")
        return {
            "total_pages": total_pages,
            "total_chunks": total_chunks_indexed,
            "duration": duration,
            "speed": pages_per_sec
        }

if __name__ == "__main__":
    import sys
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    if os.path.exists(pdf_file):
        engine = MultiParserIngestionEngine(pdf_file)
        engine.process_pdf_streaming()
    else:
        print(f"Usage: python ingest.py <pdf_path>")
