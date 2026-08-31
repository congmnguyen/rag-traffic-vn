from pathlib import Path

from fastapi.testclient import TestClient

from app import SessionStore, create_app
from rag_traffic.config import Settings


def test_sessions_are_isolated_and_bounded() -> None:
    store = SessionStore(max_sessions=2, max_turns=2)
    store.add("a", "q1", "a1")
    store.add("a", "q2", "a2")
    store.add("a", "q3", "a3")
    store.add("b", "q", "b")
    assert store.history("a") == [("q2", "a2"), ("q3", "a3")]
    assert store.history("b") == [("q", "b")]
    assert store.clear("a") is True
    assert store.history("a") == []


def test_app_reports_missing_index_instead_of_crashing(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("ok")
    settings = Settings(
        source_dir=tmp_path / "source",
        index_dir=tmp_path / "missing-index",
        static_dir=static,
        embedding_model="unused",
        openrouter_api_key=None,
        openrouter_model=None,
        top_k=3,
        min_score=0.5,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        response = client.post("/search", json={"query": "tốc độ xe máy"})
    assert health.json()["status"] == "not_ready"
    assert response.status_code == 503
