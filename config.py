"""
Central configuration for the HR AI Agent project.
Every other file imports settings from here instead of hardcoding values.
"""

import os
from dotenv import load_dotenv

# Loads variables from your .env file into the environment
load_dotenv()

# ---- API Keys ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Did you create a .env file with "
        "GROQ_API_KEY=your_key_here in the project root?"
    )

# ---- Model settings ----
# Using the 8B model as default during development — it has a SEPARATE daily
# token quota from the 70B model on Groq's free tier, so switching here avoids
# repeatedly hitting the 70B model's 100K tokens/day limit during testing and
# eval runs. Swap back to "llama-3.3-70b-versatile" for final demo/recording
# if you want slightly higher-quality responses once development is done.
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1                    # low temp = more factual, less creative

EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # free, local, runs via sentence-transformers

# ---- RAG settings ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4   # how many chunks to retrieve per query

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

if __name__ == "__main__":
    # Run this file directly to sanity-check your setup:
    # python config.py
    print("Config loaded successfully.")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Groq key loaded: {'yes' if GROQ_API_KEY else 'no'}")
    print(f"Docs directory: {DOCS_DIR}")
