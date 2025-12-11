"""
Core RAG service: embedding, retrieval, and generation logic.
"""

import time
import fitz  # PyMuPDF
import google.generativeai as genai
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

from app.config import Settings
from app.schemas import RetrievedChunk


class RAGService:
    """
    Handles all RAG operations: ingestion, embedding, retrieval, generation.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._collection: Collection | None = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize Gemini and Milvus connections."""
        if self._initialized:
            return
        
        # Configure Gemini
        if self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
        else:
            raise ValueError("GEMINI_API_KEY not configured")
        
        # Connect to Milvus
        if not self.settings.milvus_uri or not self.settings.milvus_token:
            raise ValueError("Milvus credentials not configured")
        
        connections.connect(
            alias="default",
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token,
            secure=True
        )
        
        # Load collection if it exists
        if utility.has_collection(self.settings.milvus_collection_name):
            self._collection = Collection(self.settings.milvus_collection_name)
            self._collection.load()
        
        self._initialized = True
    
    @property
    def collection(self) -> Collection | None:
        return self._collection
    
    @property
    def is_healthy(self) -> bool:
        return self._initialized and self._collection is not None
    
    # ============ Embedding Methods ============
    
    def embed_query(self, text: str) -> list[float] | None:
        """Embed a query for retrieval."""
        try:
            result = genai.embed_content(
                model=self.settings.embedding_model_id,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        """Embed documents for storage."""
        embeddings = []
        try:
            for text in texts:
                result = genai.embed_content(
                    model=self.settings.embedding_model_id,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            print(f"Batch embedding error: {e}")
            return None
    
    # ============ Retrieval Methods ============
    
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[RetrievedChunk]:
        """Search Milvus for relevant chunks."""
        if not self._collection or not query_embedding:
            return []
        
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        try:
            results = self._collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text_chunk"]
            )
            
            chunks = []
            for hit in results[0]:
                chunks.append(RetrievedChunk(
                    text=hit.entity.get("text_chunk"),
                    score=hit.distance
                ))
            return chunks
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    # ============ Generation Methods ============
    
    def generate_answer(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        """Generate an answer using retrieved context."""
        context_block = "\n\n".join([c.text for c in context_chunks])
        
        prompt = f"""You are an intelligent clinical assistant. Use the following context to answer the user's question.

--- CONTEXT ---
{context_block}
--- END CONTEXT ---

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer strictly based on the provided context.
2. If the answer is not in the context, state "I cannot find the answer in the provided documents."
3. Keep the answer professional and concise.
"""
        
        try:
            model = genai.GenerativeModel(self.settings.generation_model_id)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Generation error: {e}"
    
    # ============ Full RAG Pipeline ============
    
    def query(self, question: str, top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
        """
        Execute the full RAG pipeline: embed → retrieve → generate.
        Returns (answer, retrieved_chunks).
        """
        # 1. Embed the question
        query_vector = self.embed_query(question)
        if not query_vector:
            return "Failed to process your question.", []
        
        # 2. Retrieve relevant chunks
        chunks = self.search(query_vector, top_k)
        if not chunks:
            return "No relevant information found in the knowledge base.", []
        
        # 3. Generate answer
        answer = self.generate_answer(question, chunks)
        
        return answer, chunks
    
    # ============ Ingestion Methods ============
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _ensure_collection_exists(self) -> Collection:
        """Create collection if it doesn't exist."""
        if utility.has_collection(self.settings.milvus_collection_name):
            collection = Collection(self.settings.milvus_collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.settings.embedding_dimension)
            ]
            schema = CollectionSchema(fields, description="PDF Text Embeddings (Gemini)")
            collection = Collection(name=self.settings.milvus_collection_name, schema=schema)
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="vector", index_params=index_params)
        
        collection.load()
        self._collection = collection
        return collection
    
    def ingest_pdf(self, pdf_path: str) -> int:
        """
        Ingest a PDF file into the vector store.
        Returns number of chunks processed.
        """
        # 1. Extract text from PDF
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += page.get_text("text")
        
        # 2. Chunk the text
        chunks = self._chunk_text(full_text)
        if not chunks:
            return 0
        
        # 3. Ensure collection exists
        collection = self._ensure_collection_exists()
        
        # 4. Embed and insert in batches
        batch_size = 10
        total_processed = 0
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = self.embed_documents(batch_chunks)
            
            if embeddings and len(embeddings) == len(batch_chunks):
                data_to_insert = [batch_chunks, embeddings]
                collection.insert(data_to_insert)
                total_processed += len(batch_chunks)
                time.sleep(1)  # Rate limiting
        
        collection.flush()
        return total_processed
    
    def ingest_text(self, text: str) -> int:
        """
        Ingest raw text into the vector store.
        Returns number of chunks processed.
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        
        collection = self._ensure_collection_exists()
        embeddings = self.embed_documents(chunks)
        
        if embeddings and len(embeddings) == len(chunks):
            collection.insert([chunks, embeddings])
            collection.flush()
            return len(chunks)
        
        return 0


# Singleton instance (created on startup)
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Get the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        raise RuntimeError("RAG service not initialized. Call init_rag_service() first.")
    return _rag_service


def init_rag_service(settings: Settings) -> RAGService:
    """Initialize the RAG service singleton."""
    global _rag_service
    _rag_service = RAGService(settings)
    _rag_service.initialize()
    return _rag_service
