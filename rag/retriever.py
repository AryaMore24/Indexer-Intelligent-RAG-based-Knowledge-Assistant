import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import config as C
from .ingest import load_chunks, open_store


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """BM25 keyword search + Qdrant dense search, fused with Reciprocal Rank Fusion."""

    def __init__(self, k: int = C.TOP_K, rrf_k: int = 60):
        self.k = k
        self.rrf_k = rrf_k
        self.store = open_store()
        self.chunks = load_chunks()
        self.bm25 = BM25Okapi([_tokenize(c.page_content) for c in self.chunks])

    def _keyword_search(self, query: str, n: int) -> list[Document]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i] for i in ranked if scores[i] > 0]

    def invoke(self, query: str) -> list[Document]:
        n = self.k * 3
        dense = self.store.similarity_search(query, k=n)
        sparse = self._keyword_search(query, n)

        fused: dict[str, float] = {}
        by_key: dict[str, Document] = {}
        for results in (dense, sparse):
            for rank, doc in enumerate(results):
                key = doc.page_content
                by_key[key] = doc
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[: self.k]
        return [by_key[key] for key in top]
