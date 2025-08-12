"""
Launcher script for Claimsure Streamlit UI.

Run this script to start the Streamlit interface.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Launch the Streamlit UI."""
    
    # Check if we're in the right directory
    if not Path("src/ui.py").exists():
        print("❌ Error: src/ui.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    print("🚀 Starting Claimsure Streamlit UI...")
    print("=" * 50)
    print("📄 Claimsure - Document Query System")
    print("=" * 50)
    print()
    print("⚠️  Make sure the API server is running first!")
    print("   Run: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print()
    print("🌐 The UI will be available at: http://localhost:8501")
    print("📚 API documentation at: http://localhost:8000/docs")
    print()
    print("=" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/ui.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit UI stopped by user.")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
