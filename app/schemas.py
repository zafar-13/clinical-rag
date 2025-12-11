"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field


# ============ Request Models ============

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the patient's diagnosis?",
                    "top_k": 3
                }
            ]
        }
    }


class IngestRequest(BaseModel):
    """Request model for manual text ingestion (optional)."""
    text: str = Field(..., min_length=1, description="Text content to ingest")
    source: str = Field(default="manual", description="Source identifier")


# ============ Response Models ============

class RetrievedChunk(BaseModel):
    """A single retrieved chunk with metadata."""
    text: str
    score: float = Field(description="Similarity score (lower is better for L2)")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    chunks_retrieved: int
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is the patient's diagnosis?",
                    "answer": "The patient was diagnosed with NSTEMI and CAD.",
                    "sources": [
                        {"text": "Diagnosis: NSTEMI, CAD...", "score": 0.23}
                    ],
                    "chunks_retrieved": 3
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    message: str
    chunks_processed: int
    filename: str | None = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    milvus_connected: bool
    gemini_configured: bool
    collection_entities: int | None = None
