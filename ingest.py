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
    def __init__(self, pdf_path: str, qdrant_path: str = config.QDRANT_PATH):
        self.pdf_path = pdf_path
        self.qdrant_path = qdrant_path
        self.device = config.DEVICE
        
        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' on device: {self.device}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=self.device)
        self.vector_dim = self.embedder.get_embedding_dimension()
        
        self.qdrant = QdrantClient(path=self.qdrant_path)
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if config.COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )

    def _extract_and_chunk_page(self, doc: fitz.Document, page_num: int):
        """In-memory extraction and fast chunking for a single page."""
        try:
            page = doc[page_num]
            text = page.get_text("text")
            if not text or not text.strip():
                return []
                
            content = text.strip()
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

            words = content.split()
            chunks = []
            step = 450
            for i in range(0, len(words), step):
                chunk_words = words[i : i + 500]
                if len(chunk_words) > 30:
                    chunks.append({
                        "text": " ".join(chunk_words),
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
                            "page": chunk["page"],
                            "source_tag": f"[Source: Page {chunk['page']}]"
                        }
                    )
                    for idx, (chunk, emb) in enumerate(zip(batch_chunks, embeddings))
                ]
                point_id_counter += len(points)
                    
                self.qdrant.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points
                )
                total_chunks_indexed += len(points)

            # Explicit Garbage Collection & MPS Memory Flush to prevent macOS Kernel Panics
            import gc
            gc.collect()
            if self.device == "mps":
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
