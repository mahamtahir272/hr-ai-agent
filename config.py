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
# Main agent model — needs to be powerful for reliable tool-calling.
# 70B has a 100K tokens/day free limit on Groq.
# DO NOT switch this to 8B — it hallucinates non-existent tool names.
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# Reflection model — used only for scoring draft answers, NOT for generation
# or tool-calling. This is a simpler task the 8B model handles reliably.
# Using a separate model here gives us a separate 500K tokens/day quota,
# effectively decoupling reflection cost from the main agent's budget.
# Two-model architecture: 70B generates, 8B evaluates. Standard production pattern.
REFLECTION_MODEL = "llama-3.1-8b-instant"
REFLECTION_TEMPERATURE = 0.0   # zero temp for deterministic, consistent scoring

# ---- Embedding model ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, local, via sentence-transformers

# ---- RAG settings ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "data", "synthetic")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

if __name__ == "__main__":
    print("Config loaded successfully.")
    print(f"Main LLM model:       {LLM_MODEL}")
    print(f"Reflection model:     {REFLECTION_MODEL}")
    print(f"Embedding model:      {EMBEDDING_MODEL}")
    print(f"Groq key loaded:      {'yes' if GROQ_API_KEY else 'no'}")
    print(f"Docs directory:       {DOCS_DIR}")
