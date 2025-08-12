"""
Configuration constants for the LLM-powered query-retrieval system.
"""

import os
from typing import List

# Document Processing Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Number of characters to overlap between chunks
MAX_CHUNKS_PER_DOCUMENT = 100  # Maximum number of chunks per document

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformers model
EMBEDDING_DIMENSION = 384  # Dimension of embeddings
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation

# Vector Database Configuration
VECTOR_DB_TYPE = "pinecone"  # Options: "pinecone", "faiss", "chroma"
PINECONE_INDEX_NAME = "claimsure-index"
PINECONE_METRIC = "cosine"  # Distance metric for similarity search

# OpenAI Configuration
OPENAI_MODEL = "gpt-3.5-turbo"  # Default model for text generation
OPENAI_MAX_TOKENS = 1000  # Maximum tokens for response
OPENAI_TEMPERATURE = 0.7  # Temperature for response generation

# File Processing Configuration
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".eml"]
MAX_FILE_SIZE_MB = 10  # Maximum file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Search Configuration
TOP_K_RESULTS = 5  # Number of top results to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for results

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache Configuration
CACHE_TTL = 3600  # Cache time-to-live in seconds
CACHE_MAX_SIZE = 1000  # Maximum number of cached items

# Environment Variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Validation
def validate_config():
    """Validate that required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return True
