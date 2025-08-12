#!/usr/bin/env python3
"""
Startup script for Claimsure LLM Query-Retrieval System.
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import API_HOST, API_PORT, API_RELOAD, validate_config

def main():
    """Main startup function."""
    print("🚀 Starting Claimsure LLM Query-Retrieval System...")
    
    # Validate configuration
    try:
        validate_config()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"⚠️  Configuration warning: {e}")
        print("   Some features may be disabled without proper API keys.")
    
    # Start the server
    print(f"🌐 Starting server on {API_HOST}:{API_PORT}")
    print("📖 API documentation will be available at http://localhost:8000/docs")
    print("🔍 Streamlit frontend can be started with: streamlit run src/streamlit_app.py")
    
    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info"
    )

if __name__ == "__main__":
    main()
