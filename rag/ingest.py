import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config as C
from .models import get_embeddings


def load_documents(folder: str) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs += TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(path)).load()
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=C.CHUNK_SIZE, chunk_overlap=C.CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def _qdrant_kwargs() -> dict:
    return {"url": C.QDRANT_URL} if C.QDRANT_URL else {"path": C.QDRANT_PATH}


def ingest(folder: str) -> int:
    docs = load_documents(folder)
    if not docs:
        raise SystemExit(f"No .txt/.md/.pdf files found in {folder!r}")
    chunks = split_documents(docs)

    # Dense index in Qdrant
    QdrantVectorStore.from_documents(
        chunks,
        get_embeddings(),
        collection_name=C.COLLECTION,
        force_recreate=True,
        **_qdrant_kwargs(),
    )

    # Chunk copy on disk for the BM25 keyword index
    Path(C.CHUNKS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(C.CHUNKS_FILE).write_text(
        json.dumps([{"content": c.page_content, "metadata": c.metadata} for c in chunks]),
        encoding="utf-8",
    )
    return len(chunks)


def open_store() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=get_embeddings(), collection_name=C.COLLECTION, **_qdrant_kwargs()
    )


def load_chunks() -> list[Document]:
    path = Path(C.CHUNKS_FILE)
    if not path.exists():
        raise SystemExit("No index found. Run: python main.py ingest data/")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Document(page_content=r["content"], metadata=r["metadata"]) for r in raw]
