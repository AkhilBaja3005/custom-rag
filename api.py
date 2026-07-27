import os
import re
import json
import asyncio
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

import config
from query import RAGQueryEngine
from ingest import MultiParserIngestionEngine
from qdrant_client import QdrantClient

app = FastAPI(
    title="SOTA RAG Engine Microservice API",
    description="High-performance async FastAPI backend with SSE streaming and background ingestion jobs",
    version="2.0.0"
)

# CORS middleware for Next.js 15 frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons to prevent startup locks
_qdrant_client: Optional[QdrantClient] = None
_query_engine: Optional[RAGQueryEngine] = None

def get_db_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=config.QDRANT_PATH)
    return _qdrant_client

def get_engine() -> RAGQueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = RAGQueryEngine(qdrant_client=get_db_client())
    return _query_engine

# Ingestion Jobs State Tracker
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = config.COLLECTION_NAME
    use_hyde: Optional[bool] = True

def sanitize_collection_name(name: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    return clean if clean else config.COLLECTION_NAME

def run_background_ingestion(job_id: str, file_path: str, collection_name: str):
    try:
        INGESTION_JOBS[job_id] = {"status": "processing", "progress": 0.0, "details": "Starting ingestion..."}
        
        def progress_cb(start_p, end_p, total_p):
            pct = round((end_p / total_p) * 100, 1)
            INGESTION_JOBS[job_id] = {
                "status": "processing",
                "progress": pct,
                "details": f"Processed pages {start_p} to {end_p} of {total_p}"
            }
            
        engine = MultiParserIngestionEngine(file_path, collection_name=collection_name, qdrant_client=get_db_client())
        stats = engine.process_pdf_streaming(progress_callback=progress_cb)
        
        INGESTION_JOBS[job_id] = {
            "status": "completed",
            "progress": 100.0,
            "details": f"Successfully indexed {stats['total_pages']} pages ({stats['total_chunks']} chunks) at {stats['speed']:.2f} pages/sec",
            "stats": stats
        }
    except Exception as e:
        INGESTION_JOBS[job_id] = {"status": "failed", "progress": 0.0, "error": str(e)}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "device": config.DEVICE, "qdrant_path": config.QDRANT_PATH}

@app.get("/api/collections")
async def list_collections():
    """Lists all active indexed document collections in Qdrant."""
    try:
        client = get_db_client()
        cols = client.get_collections().collections
        result = []
        for c in cols:
            info = client.get_collection(c.name)
            result.append({"name": c.name, "points_count": info.points_count})
        return {"collections": result}
    except Exception as e:
        return {"collections": []}

@app.delete("/api/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Deletes an indexed collection from Qdrant."""
    try:
        client = get_db_client()
        client.delete_collection(collection_name=collection_name)
        engine = get_engine()
        if engine.cache:
            engine.cache.clear()
        return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None)
):
    """Starts asynchronous background streaming PDF ingestion."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    target_col = sanitize_collection_name(collection_name or file.filename.replace(".pdf", ""))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = f"job_{int(asyncio.get_event_loop().time() * 1000)}"
    INGESTION_JOBS[job_id] = {"status": "pending", "progress": 0.0, "details": "Queued"}
    
    background_tasks.add_task(run_background_ingestion, job_id, tmp_path, target_col)
    
    return {"job_id": job_id, "collection_name": target_col, "status": "processing"}

@app.get("/api/ingest/status/{job_id}")
async def get_job_status(job_id: str):
    """Polls ingestion job status."""
    if job_id not in INGESTION_JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return INGESTION_JOBS[job_id]

@app.post("/api/query")
async def sync_query(req: QueryRequest):
    """Executes full SOTA RAG query synchronously."""
    try:
        engine = get_engine()
        res = engine.query(req.query, collection_name=req.collection_name, use_hyde=req.use_hyde)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/query/stream")
async def stream_query(query: str, collection_name: str = config.COLLECTION_NAME):
    """Server-Sent Events (SSE) stream endpoint for real-time word token streaming to Next.js frontend."""
    async def event_generator():
        try:
            engine = get_engine()
            res = engine.query(query, collection_name=collection_name)
            answer = res["answer"]
            sources = res["sources"]
            is_cache = res.get("is_cache_hit", False)

            # Send metadata header chunk
            meta_event = {
                "type": "metadata",
                "is_cache_hit": is_cache,
                "sources": sources
            }
            yield {"event": "metadata", "data": json.dumps(meta_event)}

            # Stream words
            for word in answer.split(" "):
                yield {"event": "token", "data": json.dumps({"type": "token", "content": word + " "})}
                await asyncio.sleep(0.01)

            yield {"event": "done", "data": json.dumps({"type": "done"})}
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"type": "error", "message": str(e)})}

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
