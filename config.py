import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Device Configuration (MPS for Apple Silicon acceleration with CPU fallback)
if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Directory & Database Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "pdf_rag_collection"

# Ingestion Parameters
STREAMING_BATCH_SIZE = 50       # Pages per ingestion batch to strictly control RAM (< 1.5 GB)
CHUNK_MIN_WORDS = 400            # Sliding window minimum word length
CHUNK_MAX_WORDS = 500            # Sliding window maximum word length
CHUNK_OVERLAP_WORDS = 50         # Overlap between consecutive chunks
MIN_IMAGE_DIMENSION = 200        # Minimum width and height for diagram offloading to LLM

# Model Settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

# Search Parameters
TOP_K_VECTOR = 15                # Number of candidate chunks from Qdrant vector search
TOP_K_RERANK = 5                 # Final top chunks selected after cross-encoder re-ranking
