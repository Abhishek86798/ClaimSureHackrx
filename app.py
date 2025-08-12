#!/usr/bin/env python3
"""
Claimsure - Main Application Entry Point
FastAPI application for insurance document query system
"""

import os
import uvicorn
from src.main import app

if __name__ == "__main__":
    # Get port from environment variable (for Render)
    port = int(os.getenv("PORT", 8000))
    
    # Configure uvicorn for production deployment
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker to save memory
        log_level="info",
        access_log=True
    )
