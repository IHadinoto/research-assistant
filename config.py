"""
Configuration module for the research assistant.
Uses environment variables for flexibility.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
VECTOR_DIR = DATA_DIR / "vectorstore"

# Create directories if they don't exist
DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4

# Agent settings
MAX_ITERATIONS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Memory settings
CONVERSATION_MEMORY_KEY = "chat_history"
MAX_MEMORY_MESSAGES = 10
