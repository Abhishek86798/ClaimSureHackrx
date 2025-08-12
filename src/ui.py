"""
Streamlit UI for Claimsure.

Provides a web interface for document upload and query processing.
"""

import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Claimsure - Document Query System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_system_info() -> Optional[Dict[str, Any]]:
    """Get system information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/system/info", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error getting system info: {str(e)}")
    return None


def upload_documents(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    """Upload documents to the system."""
    try:
        payload = {
            "file_paths": file_paths,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        response = requests.post(f"{API_BASE_URL}/upload", json=payload, timeout=60)
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "message": f"Error uploading documents: {str(e)}",
            "documents_processed": 0,
            "chunks_created": 0,
            "errors": [str(e)]
        }


def process_query(query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> Dict[str, Any]:
    """Process a query through the API."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=30)
        return response.json()
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "response": f"Error processing query: {str(e)}",
            "results": [],
            "num_results": 0,
            "error": str(e)
        }


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("📄 Claimsure - Document Query System")
    st.markdown("LLM-powered document processing and query retrieval system")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check API health
        if check_api_health():
            st.success("✅ API Connected")
            
            # Get system info
            system_info = get_system_info()
            if system_info:
                st.subheader("System Information")
                st.write(f"**Status:** {system_info.get('system_status', 'Unknown')}")
                st.write(f"**Embedding Model:** {system_info.get('embedding_model', 'Unknown')}")
                st.write(f"**LLM Model:** {system_info.get('llm_model', 'Unknown')}")
                st.write(f"**Vector DB:** {system_info.get('vector_db_type', 'Unknown')}")
                st.write(f"**Total Documents:** {system_info.get('total_documents', 0)}")
                st.write(f"**Total Chunks:** {system_info.get('total_chunks', 0)}")
        else:
            st.error("❌ API Not Connected")
            st.warning("Please ensure the API server is running on localhost:8000")
            st.stop()
        
        st.divider()
        
        # Configuration
        st.subheader("⚙️ Configuration")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        top_k = st.slider("Top K Results", 1, 10, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.1)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🔍 Query Documents", "📊 System Stats"])
    
    # Upload Documents Tab
    with tab1:
        st.header("📤 Upload Documents")
        st.markdown("Upload and process documents for querying.")
        
        # File upload section
        st.subheader("Select Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files:")
            
            # Display file information
            file_info = []
            for file in uploaded_files:
                file_info.append({
                    "name": file.name,
                    "size": f"{file.size / 1024:.1f} KB",
                    "type": file.type
                })
            
            # Create a temporary directory for uploaded files
            temp_dir = Path("data/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save uploaded files
            saved_paths = []
            for file in uploaded_files:
                file_path = temp_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                saved_paths.append(str(file_path))
            
            # Display file table
            st.dataframe(file_info, use_container_width=True)
            
            # Upload button
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    # Upload documents
                    result = upload_documents(saved_paths, chunk_size, chunk_overlap)
                    
                    if result.get("success"):
                        st.success(f"✅ {result['message']}")
                        st.write(f"**Documents processed:** {result['documents_processed']}")
                        st.write(f"**Chunks created:** {result['chunks_created']}")
                        
                        # Show processing details
                        with st.expander("📋 Processing Details"):
                            st.json(result)
                    else:
                        st.error(f"❌ {result['message']}")
                        if result.get("errors"):
                            st.write("**Errors:**")
                            for error in result["errors"]:
                                st.write(f"- {error}")
    
    # Query Documents Tab
    with tab2:
        st.header("🔍 Query Documents")
        st.markdown("Ask questions about your uploaded documents.")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="What is this document about? What are the main topics?",
            height=100
        )
        
        # Query options
        col1, col2 = st.columns(2)
        with col1:
            top_k_query = st.number_input("Top K Results", 1, 10, top_k)
        with col2:
            similarity_threshold_query = st.slider("Similarity Threshold", 0.0, 1.0, similarity_threshold, 0.1)
        
        # Process query button
        if st.button("🔍 Process Query", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("Processing query..."):
                    result = process_query(query, top_k_query, similarity_threshold_query)
                    
                    if result.get("success"):
                        st.success("✅ Query processed successfully!")
                        
                        # Display response
                        st.subheader("🤖 AI Response")
                        st.write(result.get("response", "No response generated."))
                        
                        # Display results
                        if result.get("results"):
                            st.subheader(f"📄 Retrieved Chunks ({len(result['results'])})")
                            
                            for i, chunk in enumerate(result["results"], 1):
                                with st.expander(f"Chunk {i} - {chunk.get('file_path', 'Unknown')} (Score: {chunk.get('similarity_score', 0):.3f})"):
                                    st.write(f"**File:** {chunk.get('file_path', 'Unknown')}")
                                    st.write(f"**Type:** {chunk.get('file_type', 'Unknown')}")
                                    st.write(f"**Similarity Score:** {chunk.get('similarity_score', 0):.3f}")
                                    st.write(f"**Chunk Size:** {chunk.get('chunk_size', 0)} characters")
                                    st.write("**Content:**")
                                    st.text(chunk.get("content", ""))
                        
                        # Show processing time
                        if result.get("processing_time"):
                            st.info(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
                        
                        # Show query details
                        with st.expander("📋 Query Details"):
                            st.json(result)
                    else:
                        st.error("❌ Query processing failed!")
                        st.write(f"**Error:** {result.get('error', 'Unknown error')}")
                        
                        # Show error details
                        with st.expander("📋 Error Details"):
                            st.json(result)
            else:
                st.warning("Please enter a query.")
    
    # System Stats Tab
    with tab3:
        st.header("📊 System Statistics")
        st.markdown("View system performance and statistics.")
        
        # Refresh button
        if st.button("🔄 Refresh Statistics"):
            st.rerun()
        
        # Get system info
        system_info = get_system_info()
        if system_info:
            # System overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("System Status", system_info.get("system_status", "Unknown"))
            
            with col2:
                st.metric("Total Documents", system_info.get("total_documents", 0))
            
            with col3:
                st.metric("Total Chunks", system_info.get("total_chunks", 0))
            
            with col4:
                st.metric("API Version", system_info.get("api_version", "Unknown"))
            
            # Detailed information
            st.subheader("🔧 System Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Embedding Model:** {system_info.get('embedding_model', 'Unknown')}")
                st.write(f"**LLM Model:** {system_info.get('llm_model', 'Unknown')}")
            
            with col2:
                st.write(f"**Vector DB Type:** {system_info.get('vector_db_type', 'Unknown')}")
                st.write(f"**Vector DB Status:** {system_info.get('vector_db_status', 'Unknown')}")
            
            # Raw system info
            with st.expander("📋 Raw System Information"):
                st.json(system_info)
        else:
            st.error("Unable to retrieve system information.")
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Claimsure - LLM-powered document query system</p>
            <p>Built with FastAPI, Streamlit, and OpenAI</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
