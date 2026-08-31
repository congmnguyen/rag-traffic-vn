from __future__ import annotations

import argparse
import json

from rag_traffic.config import Settings
from rag_traffic.retriever import VectorRetriever


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Kiểm thử truy xuất FAISS từ dòng lệnh")
    parser.add_argument("query", nargs="?", help="Câu hỏi cần tìm")
    parser.add_argument("-k", type=int, default=settings.top_k)
    parser.add_argument("--min-score", type=float, default=settings.min_score)
    args = parser.parse_args()
    retriever = VectorRetriever(
        settings.chunks_file, settings.faiss_file, settings.manifest_file
    )
    query = args.query or input("Câu hỏi: ").strip()
    results = retriever.search(query, k=max(1, args.k), min_score=args.min_score)
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
