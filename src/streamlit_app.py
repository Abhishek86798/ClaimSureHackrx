"""
Streamlit frontend for the LLM-powered query-retrieval system.
"""

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Claimsure - LLM Query-Retrieval System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Claimsure - LLM Query-Retrieval System")
    st.markdown("Upload documents and query them using natural language with AI-powered search.")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page:",
            ["Upload Documents", "Query Documents", "View Documents", "Settings"]
        )
    
    # Main content
    if page == "Upload Documents":
        upload_page()
    elif page == "Query Documents":
        query_page()
    elif page == "View Documents":
        view_documents_page()
    elif page == "Settings":
        settings_page()

def upload_page():
    """Document upload page."""
    st.header("📤 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files for processing"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        # Display file information
        file_info = []
        for file in uploaded_files:
            file_info.append({
                "Filename": file.name,
                "Size (KB)": round(file.size / 1024, 2),
                "Type": file.type or "Unknown"
            })
        
        st.dataframe(pd.DataFrame(file_info))
        
        # Upload button
        if st.button("Upload and Process", type="primary"):
            with st.spinner("Uploading and processing documents..."):
                upload_results = []
                
                for file in uploaded_files:
                    try:
                        # Prepare file for upload
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        
                        # Upload to API
                        response = requests.post(
                            f"{API_BASE_URL}/upload",
                            files=files
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            upload_results.append({
                                "file": file.name,
                                "status": "✅ Success",
                                "message": result.get("message", "Uploaded successfully")
                            })
                        else:
                            upload_results.append({
                                "file": file.name,
                                "status": "❌ Error",
                                "message": f"HTTP {response.status_code}: {response.text}"
                            })
                    
                    except Exception as e:
                        upload_results.append({
                            "file": file.name,
                            "status": "❌ Error",
                            "message": str(e)
                        })
                
                # Display results
                st.subheader("Upload Results")
                results_df = pd.DataFrame(upload_results)
                st.dataframe(results_df, use_container_width=True)

def query_page():
    """Document query page."""
    st.header("🔍 Query Documents")
    
    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="Ask a question about your documents...",
        height=100,
        help="Use natural language to query your uploaded documents"
    )
    
    # Query options
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    with col2:
        similarity_threshold = st.slider(
            "Similarity threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            step=0.05
        )
    
    # Query button
    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching documents..."):
            try:
                # Prepare query request
                query_data = {
                    "query": query.strip(),
                    "top_k": top_k
                }
                
                # Send query to API
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=query_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    display_query_results(result)
                else:
                    st.error(f"Error: HTTP {response.status_code}")
                    st.code(response.text)
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

def display_query_results(result: Dict[str, Any]):
    """Display query results in a formatted way."""
    st.subheader("Search Results")
    
    query = result.get("query", "")
    results = result.get("results", [])
    total_results = result.get("total_results", 0)
    
    st.info(f"Found {total_results} results for: '{query}'")
    
    if not results:
        st.warning("No results found. Try adjusting your query or similarity threshold.")
        return
    
    # Display results
    for i, result_item in enumerate(results, 1):
        with st.expander(f"Result {i} - Score: {result_item.get('similarity_score', 0):.3f}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Content:**")
                st.write(result_item.get("content", "No content available"))
            
            with col2:
                st.write("**Metadata:**")
                metadata = result_item.get("metadata", {})
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
            
            # Similarity score
            score = result_item.get("similarity_score", 0)
            st.progress(score)
            st.caption(f"Similarity: {score:.3f}")

def view_documents_page():
    """View uploaded documents page."""
    st.header("📋 View Documents")
    
    if st.button("Refresh Documents", type="primary"):
        with st.spinner("Loading documents..."):
            try:
                # Get documents from API
                response = requests.get(f"{API_BASE_URL}/documents")
                
                if response.status_code == 200:
                    documents = response.json()
                    display_documents(documents)
                else:
                    st.error(f"Error: HTTP {response.status_code}")
                    st.code(response.text)
            
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")

def display_documents(documents: List[Dict[str, Any]]):
    """Display documents in a formatted table."""
    if not documents:
        st.info("No documents uploaded yet. Go to the Upload page to add documents.")
        return
    
    # Convert to DataFrame for better display
    df_data = []
    for doc in documents:
        df_data.append({
            "Document ID": doc.get("doc_id", ""),
            "Filename": doc.get("filename", ""),
            "Type": doc.get("file_type", ""),
            "Upload Date": doc.get("upload_date", ""),
            "Chunks": doc.get("chunk_count", 0)
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Document actions
    st.subheader("Document Actions")
    selected_doc = st.selectbox(
        "Select a document to delete:",
        options=[doc.get("doc_id", "") for doc in documents],
        format_func=lambda x: next((doc.get("filename", x) for doc in documents if doc.get("doc_id") == x), x)
    )
    
    if st.button("Delete Selected Document", type="secondary"):
        if selected_doc:
            with st.spinner("Deleting document..."):
                try:
                    response = requests.delete(f"{API_BASE_URL}/documents/{selected_doc}")
                    
                    if response.status_code == 200:
                        st.success("Document deleted successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: HTTP {response.status_code}")
                        st.code(response.text)
                
                except Exception as e:
                    st.error(f"Error deleting document: {str(e)}")

def settings_page():
    """Settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("API Configuration")
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="Base URL for the FastAPI backend"
    )
    
    st.subheader("Environment Variables")
    st.info("Make sure to set up your environment variables in the .env file:")
    st.code("""
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
    """)
    
    st.subheader("System Information")
    st.write(f"**API URL:** {api_url}")
    st.write(f"**Supported File Types:** {', '.join(SUPPORTED_FILE_TYPES)}")
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("✅ API connection successful!")
            else:
                st.error(f"❌ API connection failed: HTTP {response.status_code}")
        except Exception as e:
            st.error(f"❌ API connection failed: {str(e)}")

if __name__ == "__main__":
    main()
