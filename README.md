# Clinical RAG API

A production-ready FastAPI service for querying patient medical records using Retrieval-Augmented Generation (RAG).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
├─────────────────────────────────────────────────────────────┤
│  POST /api/v1/query         →  RAG Pipeline                 │
│  POST /api/v1/ingest/pdf    →  PDF Ingestion               │
│  POST /api/v1/ingest/text   →  Text Ingestion              │
│  GET  /api/v1/health        →  Health Check                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                         ┌─────────────────┐
│  Milvus       │                         │  Gemini API     │
│  (Zilliz)     │                         │                 │
│               │                         │  - Embeddings   │
│  Vector Store │                         │  - Generation   │
└───────────────┘                         └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd rag-api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

**Option A: Using GCP Secret Manager** (recommended for production)
```bash
# Set your GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

**Option B: Using Environment Variables** (for local development)
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run the Server

```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Query Documents

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the patient diagnosis?", "top_k": 3}'
```

**Response:**
```json
{
  "question": "What is the patient diagnosis?",
  "answer": "The patient was diagnosed with NSTEMI and CAD...",
  "sources": [
    {"text": "Diagnosis: NSTEMI...", "score": 0.23}
  ],
  "chunks_retrieved": 3
}
```

### Ingest PDF

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/pdf" \
  -F "file=@patient_record.pdf"
```

### Ingest Text

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presented with...", "source": "manual_entry"}'
```

## Project Structure

```
rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings & secrets management
│   ├── schemas.py        # Pydantic models
│   ├── routes.py         # API endpoints
│   └── services/
│       ├── __init__.py
│       └── rag_service.py  # Core RAG logic
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_PROJECT_ID` | `llmops-rag-project` | GCP project for Secret Manager |
| `MILVUS_COLLECTION_NAME` | `pdf_documents_gemini` | Milvus collection name |
| `EMBEDDING_MODEL_ID` | `models/text-embedding-004` | Gemini embedding model |
| `GENERATION_MODEL_ID` | `models/gemini-2.5-flash` | Gemini generation model |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `DEFAULT_TOP_K` | `3` | Default number of chunks to retrieve |

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Google Cloud Run

```bash
gcloud run deploy clinical-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Security Notes

- In production, restrict CORS origins in `app/main.py`
- Use GCP Secret Manager for credentials (already implemented)
- Consider adding authentication (API keys, OAuth, etc.)
- Enable HTTPS in production deployments

## CI/CD Status
![CI](https://github.com/zafar-13/clinical-rag/actions/workflows/ci-cd.yml/badge.svg)
