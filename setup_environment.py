#!/usr/bin/env python3
"""
Claimsure Environment Setup Script

This script helps set up the Claimsure environment with proper dependency
management, environment variable configuration, and validation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version matches requirements."""
    required_version = "3.13.5"
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    logger.info(f"Checking Python version...")
    logger.info(f"Required: {required_version}")
    logger.info(f"Current:  {current_version}")
    
    if current_version == required_version:
        logger.info("✅ Python version matches requirements")
        return True
    else:
        logger.warning(f"⚠️ Python version mismatch. Expected {required_version}, got {current_version}")
        logger.warning("This may cause compatibility issues.")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn", 
        "pydantic",
        "anthropic",
        "google.generativeai",
        "openai",
        "huggingface_hub",
        "sentence_transformers",
        "faiss",
        "dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.warning(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        logger.info("✅ .env file already exists")
        return True
    
    if env_example.exists():
        logger.info("Creating .env file from template...")
        shutil.copy(env_example, env_file)
        logger.info("✅ Created .env file from template")
        logger.warning("⚠️ Please edit .env file with your actual API keys")
        return True
    else:
        logger.error("❌ .env.example template not found")
        return False

def validate_env_file():
    """Validate the .env file configuration."""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.error("❌ .env file not found")
        return False
    
    logger.info("Validating .env file...")
    
    # Read and check for required variables
    with open(env_file, 'r') as f:
        content = f.read()
    
    required_vars = [
        "GOOGLE_AI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "HUGGINGFACE_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if f"{var}=" not in content:
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please add these to your .env file")
        return False
    
    logger.info("✅ .env file validation passed")
    return True

def create_directories():
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    directories = [
        "data",
        "data/cache", 
        "data/vector_store",
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return True

def check_api_keys():
    """Check if API keys are properly configured."""
    logger.info("Checking API key configuration...")
    
    # Import config manager to validate keys
    try:
        sys.path.insert(0, 'src')
        from config_manager import ConfigManager
        
        config = ConfigManager()
        available_services = config.get_llm_services()
        
        if available_services:
            logger.info(f"✅ Available LLM services: {', '.join(available_services)}")
        else:
            logger.warning("⚠️ No LLM services configured")
            logger.info("Please add API keys to your .env file")
        
        return len(available_services) > 0
        
    except Exception as e:
        logger.error(f"❌ Error checking API keys: {e}")
        return False

def run_tests():
    """Run basic tests to verify setup."""
    logger.info("Running basic tests...")
    
    try:
        # Test imports
        sys.path.insert(0, 'src')
        
        # Test core modules
        from main import app
        logger.info("✅ FastAPI app imports successfully")
        
        from core.embeddings import EmbeddingSystem
        logger.info("✅ Embedding system imports successfully")
        
        from core.query_processing import QueryProcessor
        logger.info("✅ Query processor imports successfully")
        
        logger.info("✅ All core modules import successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Claimsure Environment Setup")
    logger.info("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create .env file
    env_ok = create_env_file()
    
    # Validate .env file
    env_valid = validate_env_file()
    
    # Create directories
    dirs_ok = create_directories()
    
    # Check API keys
    keys_ok = check_api_keys()
    
    # Run tests
    tests_ok = run_tests()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📋 Setup Summary:")
    logger.info(f"Python Version: {'✅' if python_ok else '❌'}")
    logger.info(f"Dependencies: {'✅' if deps_ok else '❌'}")
    logger.info(f"Environment File: {'✅' if env_ok else '❌'}")
    logger.info(f"Environment Validation: {'✅' if env_valid else '❌'}")
    logger.info(f"Directories: {'✅' if dirs_ok else '❌'}")
    logger.info(f"API Keys: {'✅' if keys_ok else '❌'}")
    logger.info(f"Tests: {'✅' if tests_ok else '❌'}")
    
    all_ok = all([python_ok, deps_ok, env_ok, env_valid, dirs_ok, keys_ok, tests_ok])
    
    if all_ok:
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("You can now run the application with:")
        logger.info("  python src/main.py")
        logger.info("  or")
        logger.info("  uvicorn src.main:app --reload")
    else:
        logger.error("\n❌ Setup incomplete. Please fix the issues above.")
        logger.info("Common fixes:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Add API keys to .env file")
        logger.info("3. Check Python version compatibility")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
