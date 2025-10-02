"""
Configuration settings for the Crenovent AI Service
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Database Configuration
    database_url: str
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_deployment_name: str = "gpt-4o-mini"  # For LangChain SQL Agent
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    
    # Service Configuration
    service_port: int = 8000
    service_host: str = "localhost"
    log_level: str = "INFO"
    
    # Integration Configuration
   nodejs_backend_url: str = "https://revai-api-mainv2.azurewebsites.net"
    
    # Strategic Planning RAG Configuration (pgvector only)
    max_search_results: int = 10  # Strategic plans and conversations
    similarity_threshold: float = 0.7  # Higher threshold for quality results
    max_tokens: int = 4000
    temperature: float = 0.1
    
    # Planning Data Configuration
    plan_embedding_dimensions: int = 1536  # OpenAI text-embedding-3-small
    conversation_memory_limit: int = 20  # Recent conversations to remember
    plan_search_limit: int = 5  # Similar plans to retrieve
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
