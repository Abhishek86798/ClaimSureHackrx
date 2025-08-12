"""
FastAPI application for Claimsure.

Provides REST API endpoints for document processing and query retrieval.
"""

import time
import logging
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.models import (
    DocumentUploadRequest, DocumentUploadResponse,
    QueryRequest, QueryResponse, BatchQueryRequest, BatchQueryResponse,
    SystemInfoResponse, StatisticsResponse, HealthCheckResponse, ErrorResponse
)
from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm import LLMInterface
from src.query_processor import QueryProcessor
from config import API_HOST, API_PORT, API_RELOAD

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Claimsure API",
    description="LLM-powered document query and retrieval system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
document_loader = None
text_chunker = None
embedding_generator = None
vector_store = None
llm_interface = None
query_processor = None
start_time = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global document_loader, text_chunker, embedding_generator, vector_store, llm_interface, query_processor, start_time
    
    try:
        logger.info("Initializing Claimsure API components...")
        
        # Initialize components
        document_loader = DocumentLoader()
        text_chunker = TextChunker()
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()
        llm_interface = LLMInterface()
        query_processor = QueryProcessor(embedding_generator, vector_store, llm_interface)
        
        start_time = time.time()
        logger.info("Claimsure API components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API components: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Claimsure API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time if start_time else None
    
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=uptime
    )


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(request: DocumentUploadRequest):
    """Upload and process documents."""
    try:
        logger.info(f"Processing {len(request.file_paths)} documents")
        
        # Load documents
        documents = document_loader.load_multiple_documents(request.file_paths)
        
        if not documents:
            return DocumentUploadResponse(
                success=False,
                message="No documents were successfully loaded",
                documents_processed=0,
                chunks_created=0,
                errors=["No valid documents found"]
            )
        
        # Chunk documents
        chunks = text_chunker.chunk_multiple_documents(documents)
        
        if not chunks:
            return DocumentUploadResponse(
                success=False,
                message="No chunks were created from documents",
                documents_processed=len(documents),
                chunks_created=0,
                errors=["Failed to create chunks from documents"]
            )
        
        # Generate embeddings
        enriched_chunks = embedding_generator.generate_embeddings_for_chunks(chunks)
        
        if not enriched_chunks:
            return DocumentUploadResponse(
                success=False,
                message="Failed to generate embeddings for chunks",
                documents_processed=len(documents),
                chunks_created=len(chunks),
                errors=["Failed to generate embeddings"]
            )
        
        # Store in vector database
        success = vector_store.add_chunks(enriched_chunks)
        
        if not success:
            return DocumentUploadResponse(
                success=False,
                message="Failed to store chunks in vector database",
                documents_processed=len(documents),
                chunks_created=len(enriched_chunks),
                errors=["Failed to store in vector database"]
            )
        
        return DocumentUploadResponse(
            success=True,
            message=f"Successfully processed {len(documents)} documents",
            documents_processed=len(documents),
            chunks_created=len(enriched_chunks),
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}")
        return DocumentUploadResponse(
            success=False,
            message=f"Error processing documents: {str(e)}",
            documents_processed=0,
            chunks_created=0,
            errors=[str(e)]
        )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a single query."""
    start_time_query = time.time()
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process query
        result = query_processor.process_query(
            request.query,
            request.top_k,
            request.similarity_threshold
        )
        
        processing_time = time.time() - start_time_query
        
        # Convert to response model
        query_results = []
        for chunk in result.get("results", []):
            query_result = QueryResult(
                chunk_id=chunk.get("chunk_id", ""),
                content=chunk.get("content", ""),
                file_path=chunk.get("file_path", ""),
                file_type=chunk.get("file_type", ""),
                similarity_score=chunk.get("similarity_score", 0.0),
                chunk_index=chunk.get("chunk_index", 0),
                total_chunks=chunk.get("total_chunks", 0),
                chunk_size=chunk.get("chunk_size", 0)
            )
            query_results.append(query_result)
        
        return QueryResponse(
            success=result.get("success", False),
            query=result.get("query", request.query),
            response=result.get("response", ""),
            results=query_results,
            num_results=result.get("num_results", 0),
            similarity_threshold=result.get("similarity_threshold", request.similarity_threshold),
            processing_time=processing_time,
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            success=False,
            query=request.query,
            response="",
            results=[],
            num_results=0,
            similarity_threshold=request.similarity_threshold,
            processing_time=time.time() - start_time_query,
            error=str(e)
        )


@app.post("/query/batch", response_model=BatchQueryResponse)
async def process_batch_queries(request: BatchQueryRequest):
    """Process multiple queries in batch."""
    start_time_batch = time.time()
    
    try:
        logger.info(f"Processing {len(request.queries)} queries in batch")
        
        # Process queries
        results = query_processor.batch_process_queries(
            request.queries,
            request.top_k,
            request.similarity_threshold
        )
        
        # Convert to response models
        query_responses = []
        for result in results:
            query_results = []
            for chunk in result.get("results", []):
                query_result = QueryResult(
                    chunk_id=chunk.get("chunk_id", ""),
                    content=chunk.get("content", ""),
                    file_path=chunk.get("file_path", ""),
                    file_type=chunk.get("file_type", ""),
                    similarity_score=chunk.get("similarity_score", 0.0),
                    chunk_index=chunk.get("chunk_index", 0),
                    total_chunks=chunk.get("total_chunks", 0),
                    chunk_size=chunk.get("chunk_size", 0)
                )
                query_results.append(query_result)
            
            query_response = QueryResponse(
                success=result.get("success", False),
                query=result.get("query", ""),
                response=result.get("response", ""),
                results=query_results,
                num_results=result.get("num_results", 0),
                similarity_threshold=result.get("similarity_threshold", request.similarity_threshold),
                error=result.get("error")
            )
            query_responses.append(query_response)
        
        processing_time = time.time() - start_time_batch
        successful_queries = sum(1 for r in query_responses if r.success)
        failed_queries = len(query_responses) - successful_queries
        
        return BatchQueryResponse(
            success=successful_queries > 0,
            results=query_responses,
            total_queries=len(query_responses),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {str(e)}")
        return BatchQueryResponse(
            success=False,
            results=[],
            total_queries=len(request.queries),
            successful_queries=0,
            failed_queries=len(request.queries),
            processing_time=time.time() - start_time_batch
        )


@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information."""
    try:
        # Get vector database stats
        vector_db_stats = vector_store.get_index_stats()
        
        return SystemInfoResponse(
            system_status="operational",
            embedding_model=embedding_generator.model_name,
            llm_model=llm_interface.model,
            vector_db_type=vector_store.vector_db_type,
            vector_db_status="connected" if vector_store.is_initialized() else "disconnected",
            total_documents=vector_db_stats.get("total_vector_count", 0),
            total_chunks=vector_db_stats.get("total_vector_count", 0),
            api_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/stats", response_model=StatisticsResponse)
async def get_system_statistics():
    """Get system statistics."""
    try:
        # Get various statistics
        embedding_info = embedding_generator.get_embedding_info()
        llm_info = llm_interface.get_model_info()
        vector_db_stats = vector_store.get_index_stats()
        
        return StatisticsResponse(
            query_statistics={},  # Would need to track query stats
            embedding_statistics=embedding_info,
            vector_db_statistics=vector_db_stats,
            chunk_statistics={}  # Would need to track chunk stats
        )
        
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
