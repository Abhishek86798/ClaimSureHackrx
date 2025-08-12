#!/usr/bin/env python3
"""
Claimsure - Main FastAPI Application
Insurance document query system with LLM-powered retrieval
"""

import os
import logging
import gc
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our core components
from src.core.hybrid_processor import HybridProcessor
from src.core.embeddings import EmbeddingSystem
from src.document_loader import DocumentLoader
from src.text_chunker import TextChunker
from src.utils.response_formatter import format_json_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Claimsure API",
    description="LLM-powered insurance document query system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances (initialized lazily)
embedding_system = None
document_loader = None
text_chunker = None
hybrid_processor = None

def initialize_services():
    """Initialize services with memory optimization"""
    global embedding_system, document_loader, text_chunker, hybrid_processor
    
    try:
        logger.info("Initializing Claimsure services...")
        
        # Initialize with memory optimization
        embedding_system = EmbeddingSystem(
            model_name="all-MiniLM-L6-v2",
            cache_dir="/tmp/embeddings_cache"
        )
        
        document_loader = DocumentLoader()
        text_chunker = TextChunker()
        hybrid_processor = HybridProcessor(embedding_system)
        
        # Force garbage collection after initialization
        gc.collect()
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()

# Request/Response models
class QueryRequest(BaseModel):
    documents: str  # URL or file path
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
    metadata: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claimsure API - Insurance Document Query System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "embedding_system": embedding_system is not None,
            "document_loader": document_loader is not None,
            "text_chunker": text_chunker is not None,
            "hybrid_processor": hybrid_processor is not None
        }
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if embedding_system:
        return {
            "total_documents": len(embedding_system.documents),
            "total_chunks": len(embedding_system.chunks),
            "model_name": embedding_system.model_name
        }
    return {"message": "No documents loaded"}

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(request: QueryRequest):
    """
    Main endpoint for processing insurance document queries
    """
    try:
        logger.info(f"Processing {len(request.questions)} questions")
        
        # Load and process document
        logger.info("Loading document...")
        doc_content = document_loader.load_document(request.documents)
        
        if not doc_content or not doc_content.get('content'):
            raise HTTPException(status_code=400, detail="Failed to load document content")
        
        # Chunk the document
        logger.info("Chunking document...")
        chunks = text_chunker.chunk_document(doc_content['content'])
        
        # Add chunks to embedding system
        logger.info("Adding chunks to embedding system...")
        embedding_system.add_chunks(chunks)
        
        # Process each question
        all_answers = []
        all_metadata = []
        
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Process the question through the hybrid pipeline
            result = hybrid_processor.process_query(question)
            
            if result and hasattr(result, 'answer'):
                all_answers.append(result.answer)
                all_metadata.append({
                    "confidence": getattr(result, 'confidence', 0.8),
                    "source": f"Document: {request.documents}",
                    "question_index": i
                })
            else:
                all_answers.append("Unable to process this question.")
                all_metadata.append({
                    "confidence": 0.0,
                    "source": "Error in processing",
                    "question_index": i
                })
        
        # Format the response
        formatted_response = format_json_output(all_answers, all_metadata)
        
        logger.info("Query processing completed successfully")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error processing queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the uploaded file
        doc_content = document_loader.load_document(temp_path)
        
        if not doc_content or not doc_content.get('content'):
            raise HTTPException(status_code=400, detail="Failed to process uploaded file")
        
        # Chunk and add to embedding system
        chunks = text_chunker.chunk_document(doc_content['content'])
        embedding_system.add_chunks(chunks)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable (for Render)
    port = int(os.getenv("PORT", 8000))
    
    # Configure uvicorn for production
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker to save memory
        log_level="info"
    )
