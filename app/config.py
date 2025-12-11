"""
Configuration management using Pydantic Settings.
Supports both GCP Secret Manager and environment variables.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from google.cloud import secretmanager


class Settings(BaseSettings):
    """Application settings with GCP Secret Manager integration."""
    
    # GCP Configuration
    google_project_id: str = "llmops-rag-project"
    
    # Milvus Configuration
    milvus_collection_name: str = "pdf_documents_gemini"
    milvus_uri: str | None = None
    milvus_token: str | None = None
    
    # Gemini Configuration
    gemini_api_key: str | None = None
    embedding_model_id: str = "models/text-embedding-004"
    generation_model_id: str = "models/gemini-2.5-flash"
    embedding_dimension: int = 768
    
    # Chunking Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    # Search Configuration
    default_top_k: int = 3
    
    class Config:
        env_file = ".env"
        extra = "ignore"


def _get_secret_from_gcp(secret_id: str, project_id: str) -> str | None:
    """Fetch a secret from GCP Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Warning: Could not fetch secret '{secret_id}': {e}")
        return None


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings. 
    Tries GCP Secret Manager first, falls back to environment variables.
    """
    settings = Settings()
    
    # Try to fetch secrets from GCP if not already set
    if not settings.gemini_api_key:
        settings.gemini_api_key = _get_secret_from_gcp(
            "GEMINI_API_KEY", settings.google_project_id
        ) or os.getenv("GEMINI_API_KEY")
    
    if not settings.milvus_uri:
        settings.milvus_uri = _get_secret_from_gcp(
            "MILVUS_URI", settings.google_project_id
        ) or os.getenv("MILVUS_URI")
    
    if not settings.milvus_token:
        settings.milvus_token = _get_secret_from_gcp(
            "MILVUS_TOKEN", settings.google_project_id
        ) or os.getenv("MILVUS_TOKEN")
    
    return settings
