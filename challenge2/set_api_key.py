#!/usr/bin/env python3
"""
Script to set OpenAI API key for the AI Knowledge Base Search & Enrichment system.
"""

import os
import sys

def set_api_key():
    """Set the OpenAI API key."""
    print("🔑 OpenAI API Key Setup")
    print("=" * 40)
    
    # Check if API key is already set
    existing_key = os.getenv("OPENAI_API_KEY")
    if existing_key:
        print(f"✅ OpenAI API key is already set: {existing_key[:8]}...")
        return True
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return False
    
    # Set environment variable
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"✅ API key set successfully: {api_key[:8]}...")
    
    # Save to .env file for persistence
    try:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print("💾 API key saved to .env file")
    except Exception as e:
        print(f"⚠️ Could not save to .env file: {e}")
    
    return True

if __name__ == "__main__":
    if set_api_key():
        print("\n🚀 You can now run the system with: python main.py")
    else:
        print("\n❌ Setup failed. Please try again.")
        sys.exit(1)
