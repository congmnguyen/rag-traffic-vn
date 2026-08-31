import json
from pathlib import Path

import pytest

from rag_traffic.indexing import load_chunks


def test_load_chunks_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    row = {"chunk_id": "duplicate", "content": "nội dung"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="không duy nhất"):
        load_chunks(path)
