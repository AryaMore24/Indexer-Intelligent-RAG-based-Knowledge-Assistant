# Sample Knowledge Base

Qdrant is an open-source vector database written in Rust. It stores embeddings
and supports filtering on payload metadata alongside similarity search.

BM25 is a keyword ranking function based on term frequency and document length.
It works well for exact terms such as product names, error codes and acronyms.

Hybrid retrieval combines BM25 keyword scores with dense vector similarity.
Reciprocal Rank Fusion merges the two ranked lists into a single ranking.

BGE-M3 is a multilingual embedding model that can run locally through Ollama.
