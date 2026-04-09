from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import RERANKER_MODEL, RERANKER_TOP_K


def _preview(text: str, limit: int = 220) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit].rstrip() + "..."


@dataclass
class _RankedDoc:
    doc: Document
    score: float


class CrossEncoderReranker:
    """Manual cross-encoder reranker using sentence-transformers."""

    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = RERANKER_TOP_K):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.top_k = top_k
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs

        pairs = [(query, d.page_content) for d in docs]
        scores = self._model.predict(pairs)

        ranked = [
            _RankedDoc(doc=d, score=float(s))
            for d, s in zip(docs, scores)
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)

        top_docs = [x.doc for x in ranked[: self.top_k]]
        self._log_rerank(query, ranked[: self.top_k], len(docs))
        return top_docs

    def _log_rerank(self, query: str, top_ranked: list[_RankedDoc], original_count: int) -> None:
        print("\n================ RERANK DEBUG ===================")
        print(f"Query: {query}")
        print(f"Candidates before rerank: {original_count}")
        print(f"Top after rerank: {len(top_ranked)}")
        for i, item in enumerate(top_ranked, start=1):
            source = str(item.doc.metadata.get("source", "unknown"))
            chunk_id = item.doc.metadata.get("chunk_id")
            header = f"[{i}] score={item.score:.4f}, source={source}"
            if chunk_id is not None:
                header += f", chunk_id={chunk_id}"
            print(header)
            print(f"    {_preview(item.doc.page_content)}")
        print("=================================================\n")
