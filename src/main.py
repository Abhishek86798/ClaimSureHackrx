"""
FastAPI application for Claimsure - Insurance Document Query System.

This module provides the main API endpoints for processing insurance documents
and answering questions using the complete hybrid LLM pipeline.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import logging
import asyncio
import time
import json
import os
import sys

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our core modules
from document_loader import DocumentLoader
from text_chunker import TextChunker
from core.embeddings import EmbeddingSystem
from core.query_processing import QueryProcessor
from core.retrieval import ClauseRetrieval
from core.logic_evaluator import LogicEvaluator
from utils.response_formatter import format_json_output, validate_json_output

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Claimsure API",
    description="Insurance Document Query System with Hybrid LLM Processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class QuestionRequest(BaseModel):
    documents: str = Field(..., description="URL or path to the document")
    questions: List[str] = Field(..., description="List of questions to ask about the document")
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v:
            raise ValueError('At least one question is required')
        if len(v) > 10:  # Limit to prevent abuse
            raise ValueError('Maximum 10 questions allowed per request')
        return v

class QuestionResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers to the questions")
    metadata: List[Dict[str, Any]] = Field(..., description="Metadata for each answer")
    processing_time: float = Field(..., description="Total processing time in seconds")
    documents_processed: int = Field(..., description="Number of documents processed")
    questions_processed: int = Field(..., description="Number of questions processed")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    llm_services: List[str] = Field(..., description="Available LLM services")

# Global instances (initialize once)
document_loader = None
text_chunker = None
embedding_system = None
query_processor = None
clause_retrieval = None
logic_evaluator = None

def initialize_services():
    """Initialize all service components"""
    global document_loader, text_chunker, embedding_system, query_processor, clause_retrieval, logic_evaluator
    
    try:
        logger.info("Initializing Claimsure services...")
        
        # Initialize document processing services
        document_loader = DocumentLoader()
        text_chunker = TextChunker()
        
        # Initialize embedding system
        embedding_system = EmbeddingSystem()
        
        # Initialize query processing
        query_processor = QueryProcessor()
        
        # Initialize clause retrieval
        clause_retrieval = ClauseRetrieval(embedding_system)
        
        # Initialize logic evaluator
        logic_evaluator = LogicEvaluator()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    initialize_services()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Claimsure API - Insurance Document Query System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check service availability
        llm_services = []
        if query_processor:
            stats = query_processor.get_statistics()
            llm_services = stats.get("available_llm_services", [])
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            llm_services=llm_services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(request: QuestionRequest, background_tasks: BackgroundTasks):
    """
    Process questions about insurance documents.
    
    This endpoint:
    1. Loads the document from the provided URL
    2. Chunks and embeds the document
    3. For each question:
       - Parses the query
       - Retrieves relevant clauses
       - Evaluates the decision
       - Formats the output
    4. Returns structured JSON for all questions
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing {len(request.questions)} questions for document: {request.documents}")
        
        # Step 1: Load document
        logger.info("Step 1: Loading document...")
        document_text = await load_document(request.documents)
        
        # Step 2: Chunk and embed document
        logger.info("Step 2: Chunking and embedding document...")
        chunks = await chunk_and_embed_document(document_text)
        
        # Step 3: Process each question
        logger.info("Step 3: Processing questions...")
        all_answers = []
        
        for i, question in enumerate(request.questions):
            try:
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                
                # Step 3a: Parse query
                query_analysis = await parse_query(question)
                
                # Step 3b: Retrieve relevant clauses
                retrieved_chunks = await retrieve_clauses(question, chunks)
                
                # Step 3c: Evaluate decision
                decision_result = await evaluate_decision(question, retrieved_chunks)
                
                # Add question context to the result
                decision_result["question"] = question
                decision_result["query_analysis"] = query_analysis
                
                all_answers.append(decision_result)
                
                logger.info(f"✅ Question {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                # Add error result
                all_answers.append({
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": 0.0,
                    "source_clauses": [],
                    "model": "error",
                    "error": str(e),
                    "question": question
                })
        
        # Step 4: Format output
        logger.info("Step 4: Formatting output...")
        formatted_output = format_json_output(all_answers)
        
        # Validate the output
        if not validate_json_output(formatted_output):
            logger.warning("Output validation failed, but continuing...")
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Successfully processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        return QuestionResponse(
            answers=formatted_output["answers"],
            metadata=formatted_output["metadata"],
            processing_time=processing_time,
            documents_processed=1,
            questions_processed=len(request.questions)
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error processing questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing questions: {str(e)}"
        )

async def load_document(document_url: str) -> str:
    """Load document from URL or path"""
    try:
        if document_loader is None:
            raise Exception("Document loader not initialized")
        
        # Load the document
        document_text = document_loader.load_document(document_url)
        
        if not document_text:
            raise Exception("Failed to load document or document is empty")
        
        logger.info(f"✅ Document loaded successfully ({len(document_text)} characters)")
        return document_text
        
    except Exception as e:
        logger.error(f"❌ Error loading document: {e}")
        raise Exception(f"Failed to load document: {str(e)}")

async def chunk_and_embed_document(document_text: str) -> List[Dict[str, Any]]:
    """Chunk and embed the document"""
    try:
        if text_chunker is None or embedding_system is None:
            raise Exception("Text chunker or embedding system not initialized")
        
        # Chunk the document
        document_dict = {
            "content": document_text,
            "file_path": "uploaded_document",
            "file_type": "text",
            "metadata": {}
        }
        chunks = text_chunker.chunk_document(document_dict)
        
        if not chunks:
            raise Exception("No chunks created from document")
        
        # Embed the chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Create chunk metadata
                chunk_data = {
                    "id": chunk.get("chunk_id", f"chunk_{i+1}"),
                    "text": chunk.get("content", ""),
                    "source": chunk.get("file_path", "uploaded_document"),
                    "chunk_index": i
                }
                
                # Add to embedded chunks
                embedded_chunks.append(chunk_data)
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                continue
        
        # Add chunks to the embedding system
        embedding_system.add_chunks(embedded_chunks)
        
        logger.info(f"✅ Document chunked and embedded successfully ({len(embedded_chunks)} chunks)")
        return embedded_chunks
        
    except Exception as e:
        logger.error(f"❌ Error chunking and embedding document: {e}")
        raise Exception(f"Failed to process document: {str(e)}")

async def parse_query(question: str) -> Dict[str, Any]:
    """Parse and classify the query"""
    try:
        if query_processor is None:
            raise Exception("Query processor not initialized")
        
        # Parse the query
        query_analysis = query_processor.parse_query(question)
        
        logger.info(f"✅ Query parsed: {query_analysis['intent']} - {query_analysis['entities']}")
        return query_analysis
        
    except Exception as e:
        logger.error(f"❌ Error parsing query: {e}")
        # Return fallback analysis
        return {
            "intent": "general_info",
            "entities": [],
            "raw_query": question,
            "model": "fallback"
        }

async def retrieve_clauses(question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Retrieve relevant clauses for the question"""
    try:
        if clause_retrieval is None:
            raise Exception("Clause retrieval not initialized")
        
        # Retrieve relevant chunks
        retrieved_chunks = clause_retrieval.retrieve_clauses(question, top_k=5)
        
        logger.info(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks
        
    except Exception as e:
        logger.error(f"❌ Error retrieving clauses: {e}")
        # Return empty list as fallback
        return []

async def evaluate_decision(question: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate decision based on retrieved chunks"""
    try:
        if logic_evaluator is None:
            raise Exception("Logic evaluator not initialized")
        
        # Evaluate the decision
        decision_result = logic_evaluator.evaluate_decision(question, retrieved_chunks)
        
        logger.info(f"✅ Decision evaluated with confidence: {decision_result.get('confidence', 0.0)}")
        return decision_result
        
    except Exception as e:
        logger.error(f"❌ Error evaluating decision: {e}")
        # Return fallback result
        return {
            "answer": f"Unable to evaluate question: {str(e)}",
            "confidence": 0.0,
            "source_clauses": [],
            "model": "fallback",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "embedding_system": {
                "total_documents": embedding_system.get_total_documents() if embedding_system else 0,
                "vector_store_type": "faiss" if embedding_system else "none"
            },
            "query_processor": query_processor.get_statistics() if query_processor else {},
            "logic_evaluator": logic_evaluator.get_evaluation_statistics() if logic_evaluator else {},
            "available_services": {
                "document_loader": document_loader is not None,
                "text_chunker": text_chunker is not None,
                "embedding_system": embedding_system is not None,
                "query_processor": query_processor is not None,
                "clause_retrieval": clause_retrieval is not None,
                "logic_evaluator": logic_evaluator is not None
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Initialize services
    initialize_services()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
