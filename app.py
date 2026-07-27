import os
import re
import tempfile
import streamlit as st
from qdrant_client import QdrantClient

import config
from ingest import MultiParserIngestionEngine
from query import RAGQueryEngine

# ---------------------------------------------------------
# Page Configuration & UI/UX Pro Max CSS Design System
# ---------------------------------------------------------
st.set_page_config(
    page_title="Local RAG Studio - 10,000 Page PDF Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Tokens from design-system/custom-rag/MASTER.md
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Fira Sans', sans-serif;
    color: #1E1B4B;
}

code, pre, .source-chip {
    font-family: 'Fira Code', monospace;
}

/* Glassmorphism Header */
.main-header {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(8, 145, 178, 0.1) 100%);
    border: 1px solid rgba(221, 214, 254, 0.6);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #7C3AED;
    margin: 0;
    font-family: 'Fira Code', monospace;
}

.sub-title {
    font-size: 1.05rem;
    color: #64748B;
    margin-top: 6px;
}

/* Source Tag Chip */
.source-chip {
    display: inline-block;
    background-color: #ECEEF9;
    color: #7C3AED;
    border: 1px solid #DDD6FE;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-right: 6px;
}

/* Re-rank Score Tag */
.score-chip {
    display: inline-block;
    background-color: #E0F2FE;
    color: #0891B2;
    border: 1px solid #BAE6FD;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Chat Message Styling */
.stChatMessage {
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 12px;
}

.stButton>button {
    background: linear-gradient(135deg, #7C3AED 0%, #0891B2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 16px;
    transition: all 200ms ease;
    cursor: pointer;
}

.stButton>button:hover {
    opacity: 0.92;
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Singleton Resource Caching (Single Qdrant Client & Query Engine)
# ---------------------------------------------------------
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=config.QDRANT_PATH)

@st.cache_resource
def get_rag_query_engine():
    qclient = get_qdrant_client()
    return RAGQueryEngine(qdrant_client=qclient)

# Detect if qdrant_db was deleted externally and bust the cached client
if not os.path.exists(config.QDRANT_PATH):
    st.cache_resource.clear()

qclient = get_qdrant_client()
query_engine = get_rag_query_engine()

# Helper function to sanitize user collection names
def sanitize_collection_name(name: str) -> str:
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    return clean if clean else config.COLLECTION_NAME

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch LIVE collections from Qdrant on every render (no caching here)
try:
    existing_collections = [c.name for c in qclient.get_collections().collections]
except Exception:
    existing_collections = []

# If db was wiped, clear session state so phantom collections don't persist
if not existing_collections:
    st.session_state.pop("selected_collection", None)
    st.session_state.messages = []

# Ensure selected_collection is still valid; reset if the collection was deleted
if "selected_collection" not in st.session_state or \
   st.session_state.selected_collection not in existing_collections:
    if existing_collections:
        st.session_state.selected_collection = existing_collections[0]
    else:
        st.session_state.selected_collection = None

# ---------------------------------------------------------
# Sidebar Panel: Document Ingestion & Database Metrics
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Control Center")
    st.caption("Local 10,000-Page PDF RAG Engine")
    
    st.divider()
    
    # Device & Setup Status
    st.subheader("🖥️ Hardware Acceleration")
    st.info(f"**PyTorch Device:** `{config.DEVICE.upper()}`\n\n**RAM Limit:** `1.5 GB Peak`")
    
    st.divider()
    
    # Target Document / Collection Selector
    st.subheader("📚 Target Document Collection")
    if not existing_collections:
        st.warning("⚠️ No indexed documents found. Upload a PDF to get started.")
    else:
        selected_col = st.selectbox(
            "Select Document Context to Query:",
            options=existing_collections,
            index=existing_collections.index(st.session_state.selected_collection)
                  if st.session_state.selected_collection in existing_collections else 0,
            help="Select which indexed PDF document or collection to search against."
        )
        if selected_col != st.session_state.selected_collection:
            st.session_state.selected_collection = selected_col
            st.session_state.messages = []
            st.rerun()

    st.divider()
    
    # PDF Upload Section
    st.subheader("📄 Upload & Index New PDF")
    custom_name = st.text_input("Custom Collection Name (Optional)", placeholder="e.g., medical_research_2026")
    uploaded_file = st.file_uploader("Upload PDF Document (up to 10k pages)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("🚀 Start Streaming Ingestion", use_container_width=True):
            # Resolve target collection name
            if custom_name.strip():
                target_col = sanitize_collection_name(custom_name)
            else:
                target_col = sanitize_collection_name(uploaded_file.name.replace(".pdf", ""))
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(start_page, end_page, total_pages):
                pct = end_page / total_pages
                progress_bar.progress(pct)
                status_text.markdown(f"**Processing Batch:** Pages `{start_page}` to `{end_page}` of `{total_pages}`...")

            try:
                ingest_engine = MultiParserIngestionEngine(tmp_path, collection_name=target_col, qdrant_client=qclient)
                stats = ingest_engine.process_pdf_streaming(progress_callback=progress_callback)
                st.session_state.selected_collection = target_col
                st.session_state.messages = []
                st.success(f"✅ Ingested into collection `{target_col}`!\n\n- **Pages:** {stats['total_pages']:,}\n- **Chunks:** {stats['total_chunks']:,}\n- **Speed:** {stats['speed']:.2f} pages/sec")
                st.rerun()
            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.divider()
    
    # Active Qdrant DB Collection Stats
    st.subheader("📊 Active Collection Status")
    try:
        info = qclient.get_collection(collection_name=st.session_state.selected_collection)
        st.metric("Active Collection", st.session_state.selected_collection)
        st.metric("Indexed Chunks", f"{info.points_count:,}")
    except Exception:
        st.write("No active collection points.")
        
    st.divider()
    
    # New Chat Session Reset
    if st.button("🧹 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div class="main-title">⚡ Local RAG Studio ({st.session_state.selected_collection})</div>
    <div class="sub-title">PyTorch MPS Acceleration • Cross-Encoder Re-Ranking • Gemini 3.1 Flash-Lite Zero Extrapolation</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Context Chunks & Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"<span class='source-chip'>Source: Page {src['page']}</span>"
                        f"<span class='score-chip'>Rerank Score: {src.get('rerank_score', 0):.4f}</span>",
                        unsafe_allow_html=True
                    )
                    st.text_area(f"Chunk {idx} (Page {src['page']})", src["text"], height=100, key=f"hist_{msg['id']}_{idx}")

# Chat Input — disabled when no collections exist
if not existing_collections or not st.session_state.selected_collection:
    st.info("📂 No indexed documents found. Upload a PDF in the sidebar to start chatting.")
elif prompt := st.chat_input(f"Ask a question about '{st.session_state.selected_collection}'..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt, "id": len(st.session_state.messages)})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Synthesize Assistant Answer
    with st.chat_message("assistant"):
        with st.spinner(f"Searching collection '{st.session_state.selected_collection}' & re-ranking..."):
            res = query_engine.query(prompt, collection_name=st.session_state.selected_collection)
            answer_text = res["answer"]
            sources = res["sources"]
            is_cache = res.get("is_cache_hit", False)

        if is_cache:
            st.caption("⚡ **Instant Response** *(Retrieved from Sub-10ms Semantic Vector Cache)*")

        # Token Streaming effect for smooth user experience
        def stream_data():
            import time
            for word in answer_text.split(" "):
                yield word + " "
                time.sleep(0.01)

        st.write_stream(stream_data)
        
        if sources:
            with st.expander("🔍 View Retrieved Context Chunks & Sources"):
                for idx, src in enumerate(sources, 1):
                    st.markdown(
                        f"<span class='source-chip'>Source: Page {src['page']}</span>"
                        f"<span class='score-chip'>Rerank Score: {src.get('rerank_score', 0):.4f}</span>",
                        unsafe_allow_html=True
                    )
                    st.text_area(f"Chunk {idx} (Page {src['page']})", src["text"], height=100, key=f"curr_{len(st.session_state.messages)}_{idx}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "id": len(st.session_state.messages)
        })
