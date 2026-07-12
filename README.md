# Hệ thống RAG cho luật giao thông Việt Nam

Hệ thống RAG (Retrieval-Augmented Generation) trả lời câu hỏi về luật giao thông Việt Nam. Câu hỏi được một LLM router phân làm 3 nhánh — `retrieval` (tra FAISS trên kho thông tư đã index), `search` (tìm web trực tiếp qua SerpAPI, ưu tiên nguồn `.gov.vn` / thuvienphapluat.vn), hoặc `conversation` (hội thoại thường) — sau đó tổng hợp câu trả lời có trích dẫn điều luật.

## Demo giao diện

![Giao diện ứng dụng trả lời câu hỏi luật giao thông](images/ui-demo.jpg)

## Cấu trúc dự án

```
.
├── metadata.py      # Trích xuất văn bản từ PDF/DOCX thông tư (tự OCR nếu là PDF scan) → metadata.json
├── embeddings.py    # (Tuỳ chọn) Tạo embeddings bằng model tự chọn và build FAISS index
├── retrieval.py     # Chatbot CLI: router 3 nhánh + FAISS retrieval + web search + lọc similarity
├── app.py           # API server (FastAPI): POST /search, POST /clear_history
└── requirements.txt
```

Ghi chú kiến trúc:

- **Router 3 nhánh**: một LLM nhỏ (`gemma-3-4b` qua OpenRouter) quyết định câu hỏi đi nhánh nào; câu trả lời cuối tổng hợp bằng `llama-3.3-70b`.
- **Retrieval**: query được viết lại (tách đa ý thành nhiều truy vấn), tìm trên FAISS (`distiluse-base-multilingual-cased-v2`), rồi lọc lại bằng cosine similarity theo ngưỡng trước khi đưa vào prompt.
- **OCR fallback**: `metadata.py` tự phát hiện PDF không có text layer và chuyển sang OCR (Tesseract, gói tiếng Việt).

## Cài đặt

### 1. Thư viện Python

```bash
pip install -r requirements.txt
```

### 2. Poppler + Tesseract (xử lý PDF scan)

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-vie
```

**Windows:**
Tải Tesseract tại https://github.com/UB-Mannheim/tesseract/wiki, cài gói tiếng Việt và thêm đường dẫn vào PATH.

### 3. API keys

Đặt biến môi trường trước khi chạy (không hard-code key vào code):

```bash
export OPENROUTER_API_KEY=...   # LLM qua openrouter.ai
export GOOGLE_API_KEY=...       # SerpAPI cho nhánh web search
```

## Sử dụng

```bash
# 1. Trích metadata từ thư mục văn bản thông tư (PDF/DOCX)
python metadata.py

# 2. Chạy chatbot CLI (tự build FAISS index ở lần chạy đầu)
python retrieval.py

# hoặc chạy API server tại http://localhost:8000
python app.py
```

## Chú ý

- Embeddings mặc định dùng `distiluse-base-multilingual-cased-v2`; `embeddings.py` chỉ cần khi muốn thay model embedding khác.
- Chất lượng OCR phụ thuộc chất lượng bản scan; với văn bản mới nên dùng PDF dạng searchable.
