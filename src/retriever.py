from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rank_bm25 import BM25Okapi

from config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    CHROMA_DIR,
    CHUNKS_PATH,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
)
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store


def _tokenize_vi(text: str) -> list[str]:
    # Tokenizer đơn giản (whitespace + lowercase). Có thể thay bằng underthesea/vncorenlp sau.
    return [t for t in text.lower().split() if t]


def _doc_key(doc: Document) -> str:
    # Key ổn định để dedupe (ưu tiên source + chunk info nếu có)
    src = str(doc.metadata.get("source", ""))
    chunk_id = str(doc.metadata.get("chunk_id", ""))
    base = f"{src}::{chunk_id}::{doc.page_content}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _preview(text: str, limit: int = 220) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit].rstrip() + "..."


@dataclass
class _ScoredDoc:
    doc: Document
    score: float


class HybridRetriever(BaseRetriever):
    """Hybrid retriever thủ công: BM25 (sparse) + Dense (Chroma) + weighted score merge."""

    bm25_top_k: int = BM25_TOP_K
    dense_top_k: int = RETRIEVER_TOP_K
    final_top_k: int = RETRIEVER_TOP_K
    bm25_weight: float = BM25_WEIGHT
    dense_weight: float = DENSE_WEIGHT

    def _load_chunks(self) -> list[Document]:
        try:
            with open(CHUNKS_PATH, "rb") as f:
                chunks = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Không tìm thấy file chunks cho BM25 tại `{CHUNKS_PATH}`. "
                "Hãy chạy lại `python ingest.py` để tạo chunks.pkl."
            ) from e

        # chunks được tạo từ split_documents -> thường là list[Document]
        return chunks

    def _bm25_search(self, query: str, chunks: list[Document]) -> list[_ScoredDoc]:
        tokenized_corpus = [_tokenize_vi(d.page_content) for d in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize_vi(query))

        # lấy top k theo score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.bm25_top_k]
        return [_ScoredDoc(doc=chunks[i], score=float(scores[i])) for i in top_indices]

    def _dense_search(self, query: str) -> list[_ScoredDoc]:
        embedding_model = get_embedding_model(EMBEDDING_MODEL)
        vs = load_vector_store(embedding_model, CHROMA_DIR)

        # Chroma: score càng nhỏ càng giống (distance). Ta đổi thành similarity = -distance để dễ normalize.
        results: list[tuple[Document, float]] = vs.similarity_search_with_score(query, k=self.dense_top_k)
        return [_ScoredDoc(doc=d, score=float(-dist)) for (d, dist) in results]

    def _merge(self, bm25_docs: Iterable[_ScoredDoc], dense_docs: Iterable[_ScoredDoc]) -> list[Document]:
        bm25_map: dict[str, _ScoredDoc] = {_doc_key(sd.doc): sd for sd in bm25_docs}
        dense_map: dict[str, _ScoredDoc] = {_doc_key(sd.doc): sd for sd in dense_docs}

        bm25_scores = _minmax_norm({k: v.score for k, v in bm25_map.items()})
        dense_scores = _minmax_norm({k: v.score for k, v in dense_map.items()})

        keys = set(bm25_scores) | set(dense_scores)
        merged: list[tuple[str, float]] = []
        for k in keys:
            s = self.bm25_weight * bm25_scores.get(k, 0.0) + self.dense_weight * dense_scores.get(k, 0.0)
            merged.append((k, s))

        merged.sort(key=lambda x: x[1], reverse=True)

        docs_by_key: dict[str, Document] = {}
        for k in keys:
            if k in dense_map:
                docs_by_key[k] = dense_map[k].doc
            else:
                docs_by_key[k] = bm25_map[k].doc

        return [docs_by_key[k] for (k, _) in merged[: self.final_top_k]]

    def _log_retrieval(self, query: str, docs: list[Document]) -> None:
        print("\n================ RETRIEVAL DEBUG ================")
        print(f"Query: {query}")
        print(f"Retrieved chunks: {len(docs)}")
        for i, doc in enumerate(docs, start=1):
            source = str(doc.metadata.get("source", "unknown"))
            chunk_id = doc.metadata.get("chunk_id")
            header = f"[{i}] source={source}"
            if chunk_id is not None:
                header += f", chunk_id={chunk_id}"
            print(header)
            print(f"    {_preview(doc.page_content)}")
        print("=================================================\n")

    def _get_relevant_documents(self, query: str, *, run_manager: Any | None = None) -> list[Document]:
        chunks = self._load_chunks()
        bm25_hits = self._bm25_search(query, chunks)
        dense_hits = self._dense_search(query)
        final_docs = self._merge(bm25_hits, dense_hits)
        self._log_retrieval(query, final_docs)
        return final_docs


def get_retriever() -> BaseRetriever:
    """Hybrid retriever without LangChain retriever libraries (manual merge)."""
    return HybridRetriever()
