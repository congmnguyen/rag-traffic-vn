#!/usr/bin/env python3
"""Minimal mock HTTP target for guerrilla AuditAI runs.

Intentionally weak: one SEED blurb for all questions (not per-case RAG).
Empty response contexts so faithfulness uses dataset case.contexts.
Do not treat high scores with a smarter mock as product quality.
"""
from __future__ import annotations
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

SEED = 'Hệ thống RAG (Retrieval-Augmented Generation) trả lời câu hỏi về luật giao thông Việt Nam. Câu hỏi được một LLM router phân làm 3 nhánh — retrieval (tra FAISS trên kho thông tư đã index), search (tìm web trực tiếp qua SerpAPI, ưu tiên nguồn .gov.vn / thuvienphapluat.vn), hoặc conversation (hội thoại thường) — sau đó tổng hợp câu trả lời có trích dẫn điều luật. . ├── metadata.py # Trích xuất văn bản từ PDF/DOCX thông tư (tự OCR nếu là PDF scan) → metadata.json ├── embeddings.py # (Tuỳ chọn) Tạo embeddings bằng model tự chọn và build FAISS index ├── retrieval.py # Chatbot CLI: router 3 nhánh + FAISS retrieval + web search + lọc similarity ├── app.py # API server (FastAPI): POST /search, POST /clearhistory └── requirements.txt Router 3 nhánh: một LLM nhỏ (gemma-3-4b qua OpenRouter) quyết định'
ATTACK = re.compile(
    r"(ignore all previous|system prompt|api key|bỏ qua mọi|tiết lộ)",
    re.I,
)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = str(body.get("question") or "")
        if ATTACK.search(q):
            ans = "I cannot reveal system prompts or secrets."
        elif SEED:
            # Weak: same SEED for every question (honest FAIL under strict judge)
            ans = "Based on available docs: " + SEED[:500]
        else:
            ans = f"(mock) no seed docs for: {q[:80]}"
        raw = json.dumps({"answer": ans, "contexts": []}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("mock adapter http://127.0.0.1:18080/chat (weak SEED, empty contexts)")
HTTPServer(("127.0.0.1", 18080), H).serve_forever()
