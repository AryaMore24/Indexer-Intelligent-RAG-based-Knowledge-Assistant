# Intelligent RAG Knowledge Assistant

A Retrieval-Augmented Generation system for querying your own documents.

- **Hybrid retrieval**: BM25 keyword search + dense vectors in Qdrant, merged with Reciprocal Rank Fusion
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Two inference modes**
  - Cloud: OpenAI `gpt-4.1` + `text-embedding-3-large`
  - Local/offline: Gemma 4 via Ollama + `bge-m3` embeddings
- **Grounded prompting**: answers come only from retrieved context; otherwise a fixed fallback is returned, which limits hallucination

## Architecture

```
docs -> split -> embed -> Qdrant ----\
                 \-> chunks.json -> BM25 --> RRF fusion -> top-k -> grounded prompt -> LLM
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit
```

**Cloud:** set `PROVIDER=openai` and `OPENAI_API_KEY`.

**Local:**
```bash
ollama pull gemma4
ollama pull bge-m3
# set PROVIDER=ollama in .env
```

## Usage

```bash
python main.py ingest data/
python main.py ask "What does Reciprocal Rank Fusion do?"
python main.py chat
```

Put your own `.txt`, `.md` or `.pdf` files in `data/` and re-run `ingest`.
Each provider keeps a separate index, since embedding sizes differ.

## Layout

```
main.py            CLI
rag/config.py      settings from .env
rag/models.py      LLM and embedding factory (OpenAI / Ollama)
rag/ingest.py      load, chunk, index
rag/retriever.py   hybrid BM25 + Qdrant retriever
rag/chain.py       grounded prompt and answer chain
```
