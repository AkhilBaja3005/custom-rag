import os
import io
import time
import fitz  # PyMuPDF
import pymupdf4llm
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
        
        # Initialize Embedding Model on MPS device
        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' on device: {self.device}")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=self.device)
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize Qdrant Client (On-Disk Storage)
        self.qdrant = QdrantClient(path=self.qdrant_path)
        self._init_qdrant_collection()
        
        # Initialize Gemini Client if API Key is set
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.gemini_client = genai.Client(api_key=api_key)
        else:
            self.gemini_client = None
            print("WARNING: GEMINI_API_KEY not set. Diagram vision pass will return fallback text.")

    def _init_qdrant_collection(self):
        """Creates or resets the Qdrant collection."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if config.COLLECTION_NAME not in collections:
            self.qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            print(f"Created Qdrant collection '{config.COLLECTION_NAME}'.")

    def describe_image_with_gemini(self, image_bytes: bytes) -> str:
        """Offloads diagram/visual asset descriptions to gemini-3.1-flash-lite via google-genai SDK."""
        if not self.gemini_client:
            return "[Diagram Description: Visual asset detected. Set GEMINI_API_KEY to enable AI vision analysis.]"
        
        try:
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            )
            prompt = "Provide a detailed textual summary of this diagram or image including key labels, variables, and structural relationships."
            response = self.gemini_client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[image_part, prompt]
            )
            return f"[Diagram Summary: {response.text.strip()}]"
        except Exception as e:
            return f"[Diagram Processing Error: {str(e)}]"

    def extract_page_components(self, doc: fitz.Document, page_num: int) -> str:
        """Extracts digital text, Markdown tables, widgets/annots, and visual assets for a single page."""
        page = doc[page_num]
        content_parts = []
        
        # 1. Digital Text & Markdown Tables via pymupdf4llm (page index is 0-based)
        try:
            md_text = pymupdf4llm.to_markdown(doc, pages=[page_num])
            if md_text:
                content_parts.append(md_text.strip())
        except Exception:
            text = page.get_text()
            if text:
                content_parts.append(text.strip())
                
        # 2. Extract AcroForms (widgets) and Sticky Notes/Annotations (annots)
        meta_parts = []
        widgets = page.widgets()
        if widgets:
            for w in widgets:
                val = w.field_value
                name = w.field_name
                if val:
                    meta_parts.append(f"FormField '{name}': {val}")
                    
        annots = page.annots()
        if annots:
            for a in annots:
                info = a.info
                content = info.get("content")
                if content:
                    meta_parts.append(f"Annotation '{info.get('title', 'Note')}': {content}")
                    
        if meta_parts:
            content_parts.append("\n[Metadata & Form Fields]\n" + "\n".join(meta_parts))

        # 3. Detect and offload embedded diagrams/images (> 200x200 px)
        images = page.get_images(full=True)
        for img_info in images:
            xref = img_info[0]
            try:
                base_img = doc.extract_image(xref)
                width = base_img["width"]
                height = base_img["height"]
                
                if width >= config.MIN_IMAGE_DIMENSION and height >= config.MIN_IMAGE_DIMENSION:
                    img_bytes = base_img["image"]
                    diagram_summary = self.describe_image_with_gemini(img_bytes)
                    content_parts.append(f"\n[Embedded Visual Asset ({width}x{height}px)]\n{diagram_summary}")
            except Exception as e:
                continue

        return "\n\n".join(content_parts)

    def chunk_text_sliding_window(self, text: str, page_num: int):
        """Splits page content into 400-500 word chunks with page number source tags."""
        words = text.split()
        if not words:
            return []
            
        chunks = []
        step = config.CHUNK_MAX_WORDS - config.CHUNK_OVERLAP_WORDS
        for i in range(0, len(words), step):
            chunk_words = words[i : i + config.CHUNK_MAX_WORDS]
            if len(chunk_words) < 50 and chunks:
                # Append tiny trailing text to previous chunk
                chunks[-1]["text"] += " " + " ".join(chunk_words)
            else:
                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num + 1  # 1-based page index
                })
        return chunks

    def process_pdf_streaming(self, progress_callback=None):
        """Processes the PDF page-by-page in streaming batches of 50 to maintain <1.5GB RAM."""
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        print(f"Starting streaming ingestion for '{self.pdf_path}' ({total_pages} total pages)...")
        
        batch_size = config.STREAMING_BATCH_SIZE
        point_id_counter = int(time.time() * 1000)
        
        total_chunks_indexed = 0
        start_time = time.time()
        
        for start_idx in range(0, total_pages, batch_size):
            end_idx = min(start_idx + batch_size, total_pages)
            if progress_callback:
                progress_callback(start_idx + 1, end_idx, total_pages)
                
            print(f"Processing streaming batch: Pages {start_idx + 1} to {end_idx} of {total_pages}...")
            
            batch_chunks = []
            for page_num in range(start_idx, end_idx):
                page_text = self.extract_page_components(doc, page_num)
                page_chunks = self.chunk_text_sliding_window(page_text, page_num)
                batch_chunks.extend(page_chunks)
                
            if batch_chunks:
                texts = [c["text"] for c in batch_chunks]
                embeddings = self.embedder.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    device=self.device
                )
                
                points = []
                for idx, (chunk, emb) in enumerate(zip(batch_chunks, embeddings)):
                    points.append(
                        PointStruct(
                            id=point_id_counter,
                            vector=emb.tolist(),
                            payload={
                                "text": chunk["text"],
                                "page": chunk["page"],
                                "source_tag": f"[Source: Page {chunk['page']}]"
                            }
                        )
                    )
                    point_id_counter += 1
                    
                self.qdrant.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points
                )
                total_chunks_indexed += len(points)

        doc.close()
        duration = time.time() - start_time
        pages_per_sec = total_pages / duration if duration > 0 else 0
        print(f"Completed Ingestion: {total_pages} pages ({total_chunks_indexed} chunks) in {duration:.2f}s ({pages_per_sec:.2f} pages/sec).")
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
