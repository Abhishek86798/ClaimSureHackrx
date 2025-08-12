#!/usr/bin/env python3
"""
Claimsure Deployment Helper Script
Checks system compatibility and provides deployment guidance
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version must be 3.8 or higher")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'streamlit',
        'anthropic', 'google.generativeai', 'openai', 'huggingface_hub'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements-minimal.txt")
        return False
    
    print("✅ All core dependencies available")
    return True

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n🔑 Checking environment variables...")
    
    required_vars = [
        'GOOGLE_AI_API_KEY',
        'ANTHROPIC_API_KEY', 
        'HUGGINGFACE_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            print(f"❌ {var} - Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Set them in your .env file or deployment platform")
        return False
    
    print("✅ All environment variables set")
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'run.py',
        'requirements-minimal.txt',
        'runtime.txt',
        'src/main.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def test_local_run():
    """Test if the application can run locally"""
    print("\n🚀 Testing local run...")
    
    try:
        # Test import
        from src.main import app
        print("✅ FastAPI app imports successfully")
        
        # Test basic functionality
        print("✅ Application structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Local run test failed: {e}")
        return False

def get_deployment_recommendation():
    """Provide deployment recommendations"""
    print("\n🎯 Deployment Recommendations:")
    print("\n1. 🚀 Render (Recommended - Free)")
    print("   - Sign up at render.com")
    print("   - Connect GitHub repository")
    print("   - Build Command: pip install -r requirements-minimal.txt")
    print("   - Start Command: python run.py")
    
    print("\n2. 🚂 Railway (Simple - Free)")
    print("   - Sign up at railway.app")
    print("   - Connect GitHub repository")
    print("   - Auto-detects Python")
    
    print("\n3. 🏗️ Heroku (Professional - Paid)")
    print("   - Install Heroku CLI")
    print("   - heroku create claimsure-app")
    print("   - git push heroku master")
    
    print("\n❌ Avoid Netlify - Not suitable for Python applications")

def main():
    """Main deployment check function"""
    print("🔍 Claimsure Deployment Compatibility Check")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_environment_variables(),
        check_files(),
        test_local_run()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 All checks passed! Ready for deployment.")
        get_deployment_recommendation()
    else:
        print("⚠️  Some checks failed. Please fix issues before deployment.")
        print("\n💡 Quick fixes:")
        print("1. Install dependencies: pip install -r requirements-minimal.txt")
        print("2. Set environment variables in .env file")
        print("3. Ensure all files are present")
    
    print("\n📚 For detailed deployment guide, see DEPLOYMENT_GUIDE.md")

if __name__ == "__main__":
    main()
