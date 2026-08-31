from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .indexing import load_chunks, sha256_file


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_id_source: str
    title: str
    article: str
    source_file: str
    content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_id_source": self.document_id_source,
            "title": self.title,
            "article": self.article,
            "source_file": self.source_file,
            "content": self.content,
        }


class VectorRetriever:
    def __init__(self, chunks_file: Path, index_file: Path, manifest_file: Path):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = load_chunks(chunks_file)
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Phiên bản manifest không được hỗ trợ")
        if self.manifest.get("chunks_sha256") != sha256_file(chunks_file):
            raise ValueError("chunks.jsonl đã thay đổi; cần tạo lại FAISS index")

        self.index = faiss.read_index(str(index_file))
        expected_count = int(self.manifest["chunk_count"])
        expected_dimension = int(self.manifest["embedding_dimension"])
        if self.index.ntotal != expected_count or len(self.chunks) != expected_count:
            raise ValueError("Số vector, manifest và chunk không khớp")
        if self.index.d != expected_dimension:
            raise ValueError("Số chiều vector không khớp manifest")

        self.model = SentenceTransformer(self.manifest["embedding_model"])
        probe = self.model.get_embedding_dimension()
        if probe is not None and int(probe) != expected_dimension:
            raise ValueError("Model embedding không khớp index")

    def search(self, query: str, k: int = 6, min_score: float = 0.55) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        vector = self.model.encode(
            [f"{self.manifest['query_prefix']}{query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        requested = max(1, k)
        # Fetch a larger candidate pool because the recovered corpus frequently
        # contains both PDF and DOCX copies of the same document/article.
        limit = min(requested * 4, len(self.chunks))
        scores, indices = self.index.search(vector, limit)
        results: list[SearchResult] = []
        seen_sections: set[tuple[str, str]] = set()
        for raw_score, position in zip(scores[0], indices[0], strict=True):
            if position < 0 or float(raw_score) < min_score:
                continue
            chunk = self.chunks[int(position)]
            section_key = (
                chunk["document_id"].strip().casefold(),
                (chunk["article"].strip().casefold() or chunk["content_sha256"]),
            )
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=float(raw_score),
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    document_id_source=chunk["document_id_source"],
                    title=chunk["title"],
                    article=chunk["article"],
                    source_file=chunk["source_file"],
                    content=chunk["content"],
                )
            )
            if len(results) >= requested:
                break
        return results
