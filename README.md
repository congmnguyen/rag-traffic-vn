# RAG Traffic VN

Baseline RAG có thể tái lập để tra cứu văn bản pháp luật giao thông Việt Nam. Hệ thống trích xuất PDF/DOC/DOCX, chia theo điều, tạo embedding bằng `intfloat/multilingual-e5-small`, tìm cosine bằng FAISS và trả nguồn theo từng đoạn.

> Dữ liệu hiện được phục hồi từ bản lưu tháng 3/2025. Chưa có kiểm kê đầy đủ tình trạng hiệu lực, văn bản sửa đổi hoặc bãi bỏ. Kết quả chỉ dùng để tra cứu và phải được đối chiếu với nguồn pháp luật chính thức.

Kiểm tra nội bộ hiện không tìm thấy `168/2024/NĐ-CP` hoặc `36/2024/QH15` trong corpus. Đây là khoảng trống dữ liệu đã biết, không phải lỗi retrieval.

## Những gì đã thay đổi so với prototype

- Dữ liệu nguồn, index sinh ra và archive được tách khỏi mã nguồn, đồng thời bị Git bỏ qua.
- Mỗi chunk có ID và SHA-256 duy nhất; bản trùng chính xác được loại khi ingest.
- File có tên “Dự thảo” bị loại khỏi index mặc định và được ghi vào báo cáo ingestion.
- FAISS dùng cosine đúng cách: embedding được chuẩn hóa rồi tìm bằng `IndexFlatIP`.
- `manifest.json` lưu model, chiều vector, số chunk và hash metadata. API từ chối nạp index bị lệch.
- API vẫn khởi động và báo `/health` rõ ràng nếu chưa tạo index.
- Lịch sử được tách theo `session_id`, có giới hạn bộ nhớ thay vì dùng một biến toàn cục cho mọi người.
- Không có API key trong code. Khi chưa cấu hình LLM, hệ thống chạy ở chế độ `evidence_only`.
- Frontend hiển thị nguồn có cấu trúc và không render HTML do LLM sinh ra.

## Cấu trúc

```text
rag_traffic/                 # ingestion, indexing, retrieval và generation
tests/                       # unit tests offline
static/index.html            # giao diện
data/raw/Thongtu/            # dữ liệu đã giải nén (không commit)
data/index/                  # chunks + FAISS + manifest (có thể tái tạo)
data/legacy/                 # metadata/index cũ để đối chiếu
data/archives/               # ZIP gốc; có chứa secret lịch sử, không commit
metadata.py                  # CLI ingestion
embeddings.py                # CLI tạo index
retrieval.py                 # CLI tìm kiếm
app.py                       # FastAPI
```

## Cài đặt

Khuyến nghị `uv` và PyTorch CPU để tránh tải các gói CUDA không cần thiết:

```bash
make install
```

Hoặc thực hiện thủ công:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python torch \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

Để OCR PDF scan trên Ubuntu/Debian:

```bash
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-vie
```

Nếu chưa có Tesseract, ingestion vẫn hoàn tất các tài liệu đọc được và ghi danh sách lỗi vào `data/index/ingestion-report.json`.

Khi không có quyền `sudo`, pipeline cũng tự nhận bản cài user-local tại `~/.local/opt/tesseract`. Có thể đổi vị trí bằng `RAG_TESSERACT_ROOT` hoặc trỏ thẳng binary qua `RAG_TESSERACT_CMD`.

## Tạo dữ liệu và index

```bash
make ingest
make index
```

Các file được sinh:

```text
data/index/chunks.jsonl
data/index/ingestion-report.json
data/index/faiss.index
data/index/manifest.json
```

[Model card E5](https://huggingface.co/intfloat/multilingual-e5-small) yêu cầu prefix `passage:` và `query:` cho tác vụ retrieval, kể cả với ngôn ngữ không phải tiếng Anh. Pipeline tự thêm đúng hai prefix này. FAISS dùng `IndexFlatIP` trên vector đã chuẩn hóa để inner product tương đương cosine, theo [tài liệu FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## Chạy

Tìm thử trên CLI:

```bash
.venv/bin/python retrieval.py "Quy định về tốc độ tối đa của xe máy?"
```

Chạy web/API:

```bash
make serve
```

Mở `http://127.0.0.1:8000`. Kiểm tra trạng thái tại `GET /health` và OpenAPI tại `/docs`.

### Bật phần sinh câu trả lời

Sao chép `.env.example` thành `.env`, sau đó điền cả hai biến:

```dotenv
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
```

Không có hai biến này, endpoint `/search` vẫn hoạt động nhưng chỉ trả các đoạn bằng chứng. Tên model không được hard-code vì model trên gateway có thể thay đổi theo thời gian.

## Kiểm thử

```bash
make test
```

Unit tests không cần gọi LLM. Trước khi dùng thực tế cần bổ sung một bộ câu hỏi–đáp án chuẩn có điều khoản làm căn cứ, rồi đo ít nhất Recall@k, MRR và độ chính xác trích dẫn.

## An toàn dữ liệu

- Không commit `data/`, ZIP hoặc `.env` lên repository công khai.
- Archive `src-*.zip` được phục hồi có chứa một OpenRouter key lịch sử. Key đó phải được thu hồi và không được giải nén source code vào repository.
- Kho hiện có cả dự thảo và nhiều bản PDF/DOCX của cùng văn bản. Chưa được phép suy luận hiệu lực chỉ từ ngày hoặc tên file.
