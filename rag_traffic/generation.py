from __future__ import annotations

from collections.abc import Sequence

import httpx

from .retriever import SearchResult


SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật giao thông Việt Nam.
Chỉ sử dụng các đoạn tài liệu được cung cấp. Mỗi nhận định pháp lý phải có trích dẫn dạng [1], [2].
Không suy đoán hiệu lực của văn bản. Nếu tài liệu chưa đủ hoặc có thể đã lỗi thời, nói rõ điều đó.
Phân biệt loại phương tiện, hành vi, tuyến đường và thời điểm áp dụng khi tài liệu có nêu.
Đây là thông tin tra cứu, không thay thế tư vấn pháp lý chuyên nghiệp."""


def evidence_only_answer(results: Sequence[SearchResult]) -> str:
    if not results:
        return (
            "Không tìm thấy đoạn tài liệu đủ tương đồng để trả lời. "
            "Hãy diễn đạt cụ thể hành vi, loại phương tiện và thời điểm cần tra cứu."
        )
    lines = ["Các đoạn tài liệu liên quan nhất:"]
    for result in results[:3]:
        label = result.article or result.document_id
        excerpt = " ".join(result.content.split())[:500]
        lines.append(f"[{result.rank}] {label}: {excerpt}")
    lines.append(
        "Chưa cấu hình mô hình sinh câu trả lời; vui lòng đối chiếu toàn văn và tình trạng hiệu lực của văn bản."
    )
    return "\n\n".join(lines)


def generate_answer(
    query: str,
    results: Sequence[SearchResult],
    api_key: str | None,
    model: str | None,
    history: Sequence[tuple[str, str]] = (),
) -> tuple[str, str]:
    if not results or not api_key or not model:
        return evidence_only_answer(results), "evidence_only"

    context = "\n\n".join(
        f"[{item.rank}] Văn bản: {item.document_id}\n"
        f"Tiêu đề: {item.title}\nĐiều/mục: {item.article}\n"
        f"Tệp nguồn: {item.source_file}\nNội dung: {item.content}"
        for item in results
    )
    recent_history = "\n".join(
        f"Người dùng: {question}\nTrợ lý: {answer}" for question, answer in history[-3:]
    )
    user_prompt = (
        f"Lịch sử gần đây:\n{recent_history or '(không có)'}\n\n"
        f"Câu hỏi mới: {query}\n\nTài liệu truy xuất:\n{context}"
    )
    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer, "rag"
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        fallback = evidence_only_answer(results)
        return f"{fallback}\n\nKhông gọi được mô hình sinh câu trả lời: {exc}", "evidence_only"
