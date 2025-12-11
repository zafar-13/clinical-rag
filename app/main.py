"""
Clinical RAG API - FastAPI Application

A production-ready RAG service for querying patient medical records.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import router
from app.services import init_rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    Initializes the RAG service on startup.
    """
    # Startup
    print("Starting Clinical RAG API...")
    settings = get_settings()
    
    try:
        service = init_rag_service(settings)
        entity_count = service.collection.num_entities if service.collection else 0
        print(f"Connected to Milvus. Collection entities: {entity_count}")
        print(f"Gemini configured with model: {settings.generation_model_id}")
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("   API will start but some features may be unavailable.")
    
    yield
    
    # Shutdown
    print("Shutting down Clinical RAG API...")


# Create FastAPI app
app = FastAPI(
    title="Clinical RAG API",
    description="""
## Patient Medical Records RAG Service

Query patient medical records using Retrieval-Augmented Generation.

### Features
- **Query**: Ask questions about patient records
- **Ingest**: Upload new PDF documents or raw text
- **Health**: Check system status

### Architecture
- **Vector Store**: Milvus (Zilliz Cloud)
- **Embeddings**: Google Gemini text-embedding-004
- **Generation**: Google Gemini 2.5 Flash
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (configure for your needs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clinical RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }
