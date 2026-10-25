import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").lower()  # "openai" or "ollama"

# OpenAI (cloud)
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Ollama (local / offline)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")

# Qdrant: set QDRANT_URL for a server, otherwise an on-disk local store is used
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./.qdrant")
# Embedding sizes differ per provider, so each gets its own collection and chunk file
COLLECTION = f"{os.getenv('QDRANT_COLLECTION', 'knowledge')}_{PROVIDER}"
CHUNKS_FILE = f"./.qdrant/chunks_{PROVIDER}.json"

# Chunking and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))

FALLBACK = "I don't have enough information in the provided knowledge base to answer that."
