# Recovery audit — 2026-08-31

Ba archive được phục hồi từ Google Drive và đã vượt qua kiểm tra CRC.

| Archive | SHA-256 | Nội dung |
|---|---|---|
| `Thongtu.zip` | `77afa53a170e33b33c60b649122179b101c2815e43076f2bb8430f92c1aba10c` | 188 file nguồn: 119 DOCX, 67 PDF, 1 DOC và 1 lock file |
| `corpus_crawl.zip` | `8f1492902ab46eb841c5e4644b169057e01bbc74420be4ba103cd6aaa5ed1a27` | `corpus.txt`, `crawl.ipynb` |
| `src-20260831T115015Z-1-001.zip` | `8ea698934dbf1e890b4c3ee5326d3a797759104ec43de086327d3e89bc32f7df` | source prototype, frontend, metadata và FAISS legacy |

Archive `src-*` chứa một OpenRouter API key viết trực tiếp trong source. Không giải nén source này vào repository và phải coi key lịch sử là đã lộ.

## Legacy index

- `metadata.json`: 6.617 hàng.
- `segment_ids.json`: 6.617 ID và khớp thứ tự metadata.
- Chỉ có 5.221 `segment_id` duy nhất, tức mapping cũ có ID trùng.
- `faiss_index.bin`: 6.617 vector, 768 chiều, tạo bởi checkpoint fine-tune không được phục hồi.

Vì thiếu checkpoint tương ứng, index legacy chỉ được giữ để điều tra và không được API nạp.

## Ingestion baseline mới

- 187 file nguồn hợp lệ được nhìn thấy sau khi bỏ lock file.
- 185 file được xử lý, gồm 13 PDF scan bằng Tesseract `vie`.
- Hai file có tên “Dự thảo” bị loại có chủ đích.
- Không còn lỗi ingestion.
- 5.126 chunk được tạo; 193 chunk trùng chính xác bị loại.
- 5.126 `chunk_id` và `content_sha256` đều duy nhất.

Báo cáo máy đọc đầy đủ được tạo tại `data/index/ingestion-report.json` mỗi lần chạy `metadata.py`.

## Khoảng trống đã biết

Tìm kiếm toàn bộ `chunks.jsonl` và tên file nguồn không thấy `168/2024/NĐ-CP` hoặc `36/2024/QH15`. Corpus chưa đủ để trả lời đáng tin cậy các câu hỏi về chế tài và quy định hiện hành sau thời điểm dữ liệu cũ được thu thập. Cần xây dựng một corpus chính thức có trạng thái hiệu lực trước khi bật cho người dùng thực tế.
