#!/usr/bin/env python3
"""
Script to run the FastAPI server
"""
import sys
import os
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# Change to src directory so relative imports work
os.chdir(src_path)

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Second Brain Knowledge Management API...")
    print("📖 Visit http://localhost:8000/docs for interactive API documentation")
    
    # Run the FastAPI app directly
    uvicorn.run(
        "api.main:app",  # Module path to the app
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )