from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    created_at: str
    embedding_model: str
    embedding_dimension: int
    normalized: bool
    document_prefix: str
    query_prefix: str
    chunk_count: int
    chunks_sha256: str
    index_type: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_chunks(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        chunks = [json.loads(line) for line in stream if line.strip()]
    if not chunks:
        raise ValueError(f"Không có chunk trong {path}")
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("chunk_id không duy nhất; dừng tạo index để tránh lệch ánh xạ")
    return chunks


def build_index(
    chunks_file: Path,
    index_file: Path,
    manifest_file: Path,
    model_name: str,
    batch_size: int = 32,
) -> IndexManifest:
    import faiss
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(chunks_file)
    model = SentenceTransformer(model_name)
    passages = [f"passage: {chunk['content']}" for chunk in chunks]
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("Số embedding không khớp với số chunk")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_file.with_suffix(index_file.suffix + ".tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(index_file)

    manifest = IndexManifest(
        schema_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        embedding_model=model_name,
        embedding_dimension=int(embeddings.shape[1]),
        normalized=True,
        document_prefix="passage: ",
        query_prefix="query: ",
        chunk_count=len(chunks),
        chunks_sha256=sha256_file(chunks_file),
        index_type="IndexFlatIP",
    )
    temporary_manifest = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_file)
    return manifest
