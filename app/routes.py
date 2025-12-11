"""
API routes for the RAG service.
"""

import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from app.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    HealthResponse
)
from app.services import get_rag_service, RAGService


router = APIRouter()


def get_service() -> RAGService:
    """Dependency to get the RAG service."""
    return get_rag_service()


# ============ Health Check ============

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(service: RAGService = Depends(get_service)):
    """Check system health and connectivity."""
    entity_count = None
    if service.collection:
        try:
            entity_count = service.collection.num_entities
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if service.is_healthy else "degraded",
        milvus_connected=service.collection is not None,
        gemini_configured=True,
        collection_entities=entity_count
    )


# ============ Query Endpoint ============

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(
    request: QueryRequest,
    service: RAGService = Depends(get_service)
):
    """
    Query the knowledge base using RAG.
    
    - Embeds the question
    - Retrieves relevant chunks from Milvus
    - Generates an answer using Gemini
    """
    if not service.is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Collection may not be initialized."
        )
    
    answer, chunks = service.query(request.question, request.top_k)
    
    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=chunks,
        chunks_retrieved=len(chunks)
    )


# ============ Ingestion Endpoints ============

@router.post("/ingest/pdf", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    service: RAGService = Depends(get_service)
):
    """
    Ingest a PDF document into the knowledge base.
    
    - Extracts text from the PDF
    - Chunks the text
    - Generates embeddings
    - Stores in Milvus
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        chunks_processed = service.ingest_pdf(tmp_path)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {file.filename}",
            chunks_processed=chunks_processed,
            filename=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@router.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(
    request: IngestRequest,
    service: RAGService = Depends(get_service)
):
    """
    Ingest raw text into the knowledge base.
    """
    try:
        chunks_processed = service.ingest_text(request.text)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested text from {request.source}",
            chunks_processed=chunks_processed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
