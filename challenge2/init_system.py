#!/usr/bin/env python3
"""
System initialization script for AI Knowledge Base Search & Enrichment.
This script initializes the database and prepares the system for use.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_system():
    """Initialize the system by setting up directories and clearing old data."""
    print("🚀 Initializing AI Knowledge Base System")
    print("=" * 50)
    
    # Define paths
    base_dir = Path(".")
    data_dir = base_dir / "data"
    chromadb_dir = data_dir / "chromadb"
    documents_dir = data_dir / "documents"
    
    try:
        # Remove existing data if it exists
        if data_dir.exists():
            print("🧹 Cleaning existing data...")
            shutil.rmtree(data_dir)
            print("✅ Old data removed")
        
        # Create fresh directories
        print("📁 Creating fresh directories...")
        chromadb_dir.mkdir(parents=True, exist_ok=True)
        documents_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {chromadb_dir}")
        print(f"✅ Created: {documents_dir}")
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print(f"✅ OpenAI API key found: {api_key[:8]}...")
        else:
            print("⚠️ No OpenAI API key found. System will use MockLLM for demonstration.")
            print("   To use real OpenAI API, run: python set_api_key.py")
        
        # Create .gitkeep files to preserve empty directories
        (chromadb_dir / ".gitkeep").touch()
        (documents_dir / ".gitkeep").touch()
        
        print("\n🎉 System initialization complete!")
        print("📋 Next steps:")
        print("   1. Run: python main.py")
        print("   2. Open: http://localhost:8000")
        print("   3. Upload documents and start searching!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

if __name__ == "__main__":
    success = initialize_system()
    if success:
        print("\n✅ System ready to use!")
    else:
        print("\n❌ Initialization failed!")
        exit(1)
