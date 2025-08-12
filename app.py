#!/usr/bin/env python3
"""
Claimsure - Main Application Entry Point
FastAPI application for insurance document query system
"""

import uvicorn
from src.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
