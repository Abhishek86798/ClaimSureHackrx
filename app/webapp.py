"""
Claimsure Web Interface - Streamlit App

A modern web interface for the Claimsure Insurance Document Query System.
Allows users to upload documents, preview extracted text, ask queries, and view results.
"""

import streamlit as st
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import Claimsure modules
from config_manager import ConfigManager
from document_loader import DocumentLoader
from text_chunker import TextChunker
from core.embeddings import EmbeddingSystem
from core.query_processing import QueryProcessor
from core.retrieval import ClauseRetrieval
from core.logic_evaluator import LogicEvaluator
from utils.response_formatter import format_json_output

# Configure page
st.set_page_config(
    page_title="Claimsure - Insurance Document Query System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'document_text' not in st.session_state:
    st.session_state.document_text = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = None
if 'embedding_system' not in st.session_state:
    st.session_state.embedding_system = None
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = None
if 'clause_retrieval' not in st.session_state:
    st.session_state.clause_retrieval = None
if 'logic_evaluator' not in st.session_state:
    st.session_state.logic_evaluator = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []

def initialize_services():
    """Initialize Claimsure services."""
    try:
        if st.session_state.embedding_system is None:
            with st.spinner("Initializing embedding system..."):
                st.session_state.embedding_system = EmbeddingSystem()
        
        if st.session_state.query_processor is None:
            with st.spinner("Initializing query processor..."):
                st.session_state.query_processor = QueryProcessor()
        
        if st.session_state.clause_retrieval is None:
            with st.spinner("Initializing clause retrieval..."):
                st.session_state.clause_retrieval = ClauseRetrieval(st.session_state.embedding_system)
        
        if st.session_state.logic_evaluator is None:
            with st.spinner("Initializing logic evaluator..."):
                st.session_state.logic_evaluator = LogicEvaluator()
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return False

def process_document(uploaded_file) -> Optional[str]:
    """Process uploaded document and extract text."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            tmp_file_path = tmp_file.name
        
        # Load document
        document_loader = DocumentLoader()
        document_data = document_loader.load_document(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        # Extract text content from the document data
        if isinstance(document_data, dict) and "content" in document_data:
            return document_data["content"]
        elif isinstance(document_data, str):
            return document_data
        else:
            st.error(f"Unexpected document format: {type(document_data)}")
            return None
            
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

def chunk_document(document_text: str) -> List[Dict[str, Any]]:
    """Chunk the document text."""
    try:
        text_chunker = TextChunker()
        document_dict = {
            "content": document_text,
            "file_path": "uploaded_document",
            "file_type": "text",
            "metadata": {}
        }
        chunks = text_chunker.chunk_document(document_dict)
        
        # Convert chunks to format expected by embedding system
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": chunk.get("chunk_id", f"chunk_{i+1}"),
                "text": chunk.get("content", ""),
                "source": chunk.get("file_path", "uploaded_document"),
                "chunk_index": i
            }
            embedded_chunks.append(chunk_data)
        
        return embedded_chunks
    except Exception as e:
        st.error(f"Error chunking document: {e}")
        return []

def process_query(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process a query and return results."""
    try:
        # Parse query
        query_analysis = st.session_state.query_processor.parse_query(query)
        
        # Retrieve relevant clauses
        retrieved_chunks = st.session_state.clause_retrieval.retrieve_clauses(query, top_k=5)
        
        # Evaluate decision
        decision_result = st.session_state.logic_evaluator.evaluate_decision(query, retrieved_chunks)
        
        # Add metadata
        decision_result["question"] = query
        decision_result["query_analysis"] = query_analysis
        decision_result["retrieved_chunks"] = retrieved_chunks
        
        return decision_result
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "confidence": 0.0,
            "source_clauses": [],
            "model": "error",
            "error": str(e)
        }

def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🏥 Claimsure</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Insurance Document Query System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h3 class="sub-header">📋 System Status</h3>', unsafe_allow_html=True)
        
        # Initialize services
        if initialize_services():
            st.markdown('<div class="success-box">✅ All services initialized</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ Service initialization failed</div>', unsafe_allow_html=True)
            return
        
        # Configuration info
        try:
            config = ConfigManager()
            st.markdown('<h4>🔧 Configuration</h4>', unsafe_allow_html=True)
            st.write(f"**Environment:** {config.get('environment')}")
            st.write(f"**API Host:** {config.get('api_host')}")
            st.write(f"**API Port:** {config.get('api_port')}")
            
            # Available services
            available_services = config.get_llm_services()
            st.markdown('<h4>🤖 Available LLM Services</h4>', unsafe_allow_html=True)
            for service in available_services:
                st.write(f"✅ {service.title()}")
            
            # System metrics
            if st.session_state.document_text and isinstance(st.session_state.document_text, str):
                st.markdown('<h4>📊 Document Stats</h4>', unsafe_allow_html=True)
                st.write(f"**Characters:** {len(st.session_state.document_text):,}")
                st.write(f"**Chunks:** {len(st.session_state.document_chunks) if st.session_state.document_chunks else 0}")
                st.write(f"**Queries Processed:** {len(st.session_state.processing_results)}")
        
        except Exception as e:
            st.error(f"Configuration error: {e}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📄 Document Upload</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'docx', 'txt'],
            help="Upload a PDF, DOCX, or TXT file to analyze"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process document button
            if st.button("🔍 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Extract text
                    document_text = process_document(uploaded_file)
                    if document_text:
                        st.session_state.document_text = document_text
                        
                        # Chunk document
                        chunks = chunk_document(document_text)
                        if chunks:
                            st.session_state.document_chunks = chunks
                            
                            # Add chunks to embedding system
                            st.session_state.embedding_system.add_chunks(chunks)
                            
                            st.success(f"✅ Document processed successfully! Created {len(chunks)} chunks.")
                        else:
                            st.error("❌ Failed to chunk document.")
                    else:
                        st.error("❌ Failed to extract text from document.")
    
    with col2:
        st.markdown('<h2 class="sub-header">❓ Query Interface</h2>', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your question about the document:",
            height=100,
            placeholder="e.g., What are the coverage limits for medical expenses?",
            help="Ask specific questions about the insurance document"
        )
        
        # Query processing
        if query and st.session_state.document_chunks:
            if st.button("🔍 Process Query", type="primary"):
                with st.spinner("Processing query..."):
                    start_time = time.time()
                    
                    # Process query
                    result = process_query(query, st.session_state.document_chunks)
                    
                    processing_time = time.time() - start_time
                    result["processing_time"] = processing_time
                    
                    # Add to results
                    st.session_state.processing_results.append(result)
                    
                    st.success(f"✅ Query processed in {processing_time:.2f}s")
    
    # Document preview
    if st.session_state.document_text and isinstance(st.session_state.document_text, str):
        st.markdown('<h2 class="sub-header">📖 Document Preview</h2>', unsafe_allow_html=True)
        
        # Show document stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{len(st.session_state.document_text):,}")
        with col2:
            st.metric("Words", f"{len(st.session_state.document_text.split()):,}")
        with col3:
            st.metric("Chunks", len(st.session_state.document_chunks) if st.session_state.document_chunks else 0)
        with col4:
            st.metric("Queries", len(st.session_state.processing_results))
        
        # Document text preview
        with st.expander("📄 View Extracted Text", expanded=False):
            preview_length = 1000
            preview_text = st.session_state.document_text[:preview_length]
            if len(st.session_state.document_text) > preview_length:
                preview_text += "..."
            
            st.markdown(f'<div class="code-block">{preview_text}</div>', unsafe_allow_html=True)
    
    # Results section
    if st.session_state.processing_results:
        st.markdown('<h2 class="sub-header">🎯 Query Results</h2>', unsafe_allow_html=True)
        
        # Show latest result
        latest_result = st.session_state.processing_results[-1]
        
        # Result cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{latest_result.get('confidence', 0.0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Processing Time", f"{latest_result.get('processing_time', 0.0):.2f}s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Used", latest_result.get('model', 'Unknown'))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Answer
        st.markdown('<h3>💡 Answer</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{latest_result.get("answer", "No answer generated")}</div>', unsafe_allow_html=True)
        
        # Query analysis
        if "query_analysis" in latest_result:
            st.markdown('<h3>🔍 Query Analysis</h3>', unsafe_allow_html=True)
            analysis = latest_result["query_analysis"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Intent:** {analysis.get('intent', 'Unknown')}")
            with col2:
                st.write(f"**Entities:** {', '.join(analysis.get('entities', []))}")
        
        # Retrieved clauses
        if "retrieved_chunks" in latest_result and latest_result["retrieved_chunks"]:
            st.markdown('<h3>📋 Retrieved Clauses</h3>', unsafe_allow_html=True)
            
            for i, chunk in enumerate(latest_result["retrieved_chunks"][:3]):  # Show top 3
                with st.expander(f"Clause {i+1} (Score: {chunk.get('score', 0.0):.3f})", expanded=False):
                    st.write(f"**Source:** {chunk.get('source', 'Unknown')}")
                    st.write(f"**Text:** {chunk.get('text', 'No text')[:200]}...")
        
        # Raw JSON result
        with st.expander("🔧 View Raw JSON Result", expanded=False):
            st.json(latest_result)
        
        # History
        if len(st.session_state.processing_results) > 1:
            st.markdown('<h3>📚 Query History</h3>', unsafe_allow_html=True)
            
            for i, result in enumerate(reversed(st.session_state.processing_results[:-1])):  # Exclude latest
                with st.expander(f"Query {len(st.session_state.processing_results) - i}: {result.get('question', 'Unknown')[:50]}...", expanded=False):
                    st.write(f"**Answer:** {result.get('answer', 'No answer')[:100]}...")
                    st.write(f"**Confidence:** {result.get('confidence', 0.0):.2f}")
                    st.write(f"**Model:** {result.get('model', 'Unknown')}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Claimsure - Insurance Document Query System | Powered by Hybrid LLM Processing</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
