"""
Clinical RAG API - Streamlit UI
A clean, interactive interface for querying patient medical records.
"""

import streamlit as st
import requests

# --- Configuration ---
API_BASE_URL = "http://localhost:8000/api"

# --- Page Config ---
st.set_page_config(
    page_title="Clinical RAG",
    page_icon="🏥",
    layout="centered"
)

# --- Custom CSS for clean UI ---
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 14px;
    }
    .score-badge {
        background-color: #e1e5eb;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.ConnectionError:
        return False, None


def query_rag(question: str, top_k: int = 3):
    """Send query to RAG API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return False, str(e)


def upload_pdf(file):
    """Upload PDF to ingestion endpoint."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/ingest/pdf", files=files, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Upload failed")
    except Exception as e:
        return False, str(e)


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Health Check
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("✅ API Connected")
        st.caption(f"Documents: {health_data.get('collection_entities', 0)} chunks")
    else:
        st.error("❌ API Offline")
        st.caption("Start the FastAPI server first")
    
    st.divider()
    
    # Search Settings
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=3)
    
    st.divider()
    
    # PDF Upload
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file:
        if st.button("Ingest PDF", use_container_width=True):
            with st.spinner("Processing..."):
                success, result = upload_pdf(uploaded_file)
                if success:
                    st.success(f"✅ Ingested {result['chunks_processed']} chunks")
                else:
                    st.error(f"❌ {result}")


# --- Main Content ---
st.title("🏥 Clinical RAG")
st.caption("Query patient medical records using AI")

# Query Input
question = st.text_input(
    "Ask a question",
    placeholder="e.g., What is the patient's diagnosis?",
    label_visibility="collapsed"
)

# Example Questions
with st.expander("💡 Example questions"):
    examples = [
        "What is the patient's diagnosis?",
        "What medications is the patient taking?",
        "What symptoms did the patient present with?",
        "What is the patient's medical history?",
        "What were the lab results?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            question = ex
            st.session_state["question"] = ex

# Handle example button clicks
if "question" in st.session_state:
    question = st.session_state.pop("question")

# Search Button & Results
if question:
    with st.spinner("Searching..."):
        success, result = query_rag(question, top_k)
    
    if success:
        # Answer Section
        st.subheader("Answer")
        st.markdown(result["answer"])
        
        # Sources Section
        with st.expander(f"📚 Sources ({result['chunks_retrieved']} chunks)", expanded=False):
            for i, source in enumerate(result["sources"], 1):
                st.markdown(f"""
                <div class="source-box">
                    <strong>Chunk {i}</strong> 
                    <span class="score-badge">Score: {source['score']:.3f}</span>
                    <br><br>
                    {source['text'][:500]}{'...' if len(source['text']) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error(f"Error: {result}")

# Footer
st.divider()
st.caption("Built with FastAPI + Milvus + Gemini")