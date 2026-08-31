from rag_traffic.generation import evidence_only_answer
from rag_traffic.retriever import SearchResult


def test_evidence_only_answer_uses_source_rank() -> None:
    result = SearchResult(
        rank=1,
        score=0.8,
        chunk_id="x",
        document_id="01/2024/TT-BGTVT",
        document_id_source="header",
        title="Văn bản thử nghiệm",
        article="Điều 1. Phạm vi",
        source_file="document.docx",
        content="Nội dung làm căn cứ.",
    )
    answer = evidence_only_answer([result])
    assert "[1] Điều 1. Phạm vi" in answer
    assert "đối chiếu toàn văn" in answer
