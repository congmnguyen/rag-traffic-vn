from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag_traffic.config import Settings
from rag_traffic.indexing import build_index


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Tạo FAISS index đồng bộ với metadata")
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    manifest = build_index(
        settings.chunks_file,
        settings.faiss_file,
        settings.manifest_file,
        args.model,
        batch_size=max(1, args.batch_size),
    )
    print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
