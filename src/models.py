"""
API models for Claimsure.

Defines Pydantic models for request and response schemas.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    file_paths: List[str] = Field(..., description="List of file paths to upload")
    chunk_size: Optional[int] = Field(1000, description="Size of text chunks")
    chunk_overlap: Optional[int] = Field(200, description="Overlap between chunks")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Response message")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")


class QueryRequest(BaseModel):
    """Request model for query processing."""
    
    query: str = Field(..., description="User query string", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of top results to retrieve", ge=1, le=20)
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)


class QueryResult(BaseModel):
    """Model for individual query result."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    file_path: str = Field(..., description="Source file path")
    file_type: str = Field(..., description="File type")
    similarity_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    chunk_index: int = Field(..., description="Chunk index in document")
    total_chunks: int = Field(..., description="Total chunks in document")
    chunk_size: int = Field(..., description="Size of chunk in characters")


class QueryResponse(BaseModel):
    """Response model for query processing."""
    
    success: bool = Field(..., description="Whether the query was successful")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    results: List[QueryResult] = Field(default_factory=list, description="Retrieved chunks")
    num_results: int = Field(..., description="Number of results returned")
    similarity_threshold: float = Field(..., description="Similarity threshold used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchQueryRequest(BaseModel):
    """Request model for batch query processing."""
    
    queries: List[str] = Field(..., description="List of queries to process", min_items=1, max_items=10)
    top_k: Optional[int] = Field(5, description="Number of top results to retrieve", ge=1, le=20)
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score", ge=0.0, le=1.0)


class BatchQueryResponse(BaseModel):
    """Response model for batch query processing."""
    
    success: bool = Field(..., description="Whether the batch processing was successful")
    results: List[QueryResponse] = Field(..., description="Results for each query")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")


class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    
    system_status: str = Field(..., description="Overall system status")
    embedding_model: str = Field(..., description="Embedding model name")
    llm_model: str = Field(..., description="LLM model name")
    vector_db_type: str = Field(..., description="Vector database type")
    vector_db_status: str = Field(..., description="Vector database status")
    total_documents: int = Field(..., description="Total documents in system")
    total_chunks: int = Field(..., description="Total chunks in system")
    api_version: str = Field(..., description="API version")


class StatisticsResponse(BaseModel):
    """Response model for system statistics."""
    
    query_statistics: Dict[str, Any] = Field(..., description="Query processing statistics")
    embedding_statistics: Dict[str, Any] = Field(..., description="Embedding generation statistics")
    vector_db_statistics: Dict[str, Any] = Field(..., description="Vector database statistics")
    chunk_statistics: Dict[str, Any] = Field(..., description="Chunk processing statistics")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="System uptime in seconds")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
