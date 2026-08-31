from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    source_dir: Path
    index_dir: Path
    static_dir: Path
    embedding_model: str
    openrouter_api_key: str | None
    openrouter_model: str | None
    top_k: int
    min_score: float

    @property
    def chunks_file(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def faiss_file(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def manifest_file(self) -> Path:
        return self.index_dir / "manifest.json"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            source_dir=_path_from_env("RAG_SOURCE_DIR", PROJECT_ROOT / "data/raw/Thongtu"),
            index_dir=_path_from_env("RAG_INDEX_DIR", PROJECT_ROOT / "data/index"),
            static_dir=_path_from_env("RAG_STATIC_DIR", PROJECT_ROOT / "static"),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_model=os.getenv("OPENROUTER_MODEL") or None,
            top_k=max(1, int(os.getenv("RAG_TOP_K", "6"))),
            min_score=float(os.getenv("RAG_MIN_SCORE", "0.55")),
        )
