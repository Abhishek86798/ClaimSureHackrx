#!/usr/bin/env python3
"""
Claimsure Web Interface Launcher

This script launches the Streamlit web interface for Claimsure.
It sets up the environment and starts the Streamlit app.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_streamlit():
    """Check if Streamlit is installed."""
    try:
        import streamlit
        logger.info(f"✅ Streamlit {streamlit.__version__} is installed")
        return True
    except ImportError:
        logger.error("❌ Streamlit is not installed")
        logger.info("Install it with: pip install streamlit")
        return False

def check_webapp_file():
    """Check if the webapp file exists."""
    webapp_path = Path("app/webapp.py")
    if webapp_path.exists():
        logger.info(f"✅ Webapp file found: {webapp_path}")
        return True
    else:
        logger.error(f"❌ Webapp file not found: {webapp_path}")
        return False

def setup_environment():
    """Set up the environment for the webapp."""
    try:
        # Add src to Python path
        src_path = Path("src").absolute()
        if src_path.exists():
            sys.path.insert(0, str(src_path))
            logger.info(f"✅ Added {src_path} to Python path")
        
        # Check if .env exists
        env_path = Path(".env")
        if env_path.exists():
            logger.info("✅ Environment file found")
        else:
            logger.warning("⚠️ Environment file not found. Using system environment variables.")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to setup environment: {e}")
        return False

def launch_webapp():
    """Launch the Streamlit webapp."""
    try:
        webapp_path = Path("app/webapp.py")
        
        # Set Streamlit configuration
        os.environ["STREAMLIT_SERVER_PORT"] = "8501"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        
        logger.info("🚀 Launching Claimsure Web Interface...")
        logger.info("📱 Opening browser at: http://localhost:8501")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(webapp_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Webapp stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch webapp: {e}")

def main():
    """Main launcher function."""
    logger.info("🏥 Claimsure Web Interface Launcher")
    logger.info("=" * 50)
    
    # Check prerequisites
    if not check_streamlit():
        return False
    
    if not check_webapp_file():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Launch webapp
    launch_webapp()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
