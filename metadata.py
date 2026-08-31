from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag_traffic.config import Settings
from rag_traffic.ingest import ingest


def main() -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Trích xuất và chia đoạn văn bản pháp luật")
    parser.add_argument("--source", default=str(settings.source_dir))
    parser.add_argument("--output", default=str(settings.chunks_file))
    parser.add_argument("--report", default=str(settings.index_dir / "ingestion-report.json"))
    args = parser.parse_args()
    report = ingest(Path(args.source), Path(args.output))
    rendered = json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n"
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
