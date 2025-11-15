# CLAUDE.md - AI Assistant Guide for RAG Traffic VN

> **Last Updated**: 2025-11-15
> **Purpose**: Comprehensive guide for AI assistants working on this RAG system for Vietnamese traffic law documents

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Architecture & Data Flow](#architecture--data-flow)
4. [Key Components](#key-components)
5. [Development Workflows](#development-workflows)
6. [Conventions & Best Practices](#conventions--best-practices)
7. [Known Issues & Warnings](#known-issues--warnings)
8. [AI Assistant Guidelines](#ai-assistant-guidelines)
9. [Quick Reference](#quick-reference)

---

## Project Overview

### Purpose
A **Retrieval-Augmented Generation (RAG)** system for Vietnamese traffic law documents. This is a legal assistant chatbot that combines vector search, web search, and conversational AI to answer questions about Vietnamese legal documents, specifically focused on traffic regulations (Nghị định 168/2024/NĐ-CP).

### Technology Stack
- **Language**: Python 3.10
- **Web Framework**: FastAPI
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Transformers + Custom fine-tuned model
- **LLM**: OpenRouter API (Llama 3.3 70B Instruct)
- **Web Search**: SerpAPI (Google Search)
- **Document Processing**: pdfminer.six, python-docx, pytesseract (OCR)
- **Memory**: LangChain ConversationBufferMemory

### Core Features
1. **Semantic search** in Vietnamese legal documents
2. **Multi-hop query decomposition** for complex questions
3. **Hybrid search**: Vector retrieval + Web search fallback
4. **Conversation memory** for contextual follow-ups
5. **Intelligent query routing** (retrieval/search/conversation)

---

## Codebase Structure

```
rag-traffic-vn/
├── app.py                  # FastAPI web server (MAIN ENTRY POINT)
├── retrieval.py            # Vector retrieval engine ⚠️ HAS MERGE CONFLICTS
├── metadata.py             # Document processing & segmentation
├── embeddings.py           # Vector embedding generation
├── requirements.txt        # Python dependencies
│
├── README.md               # Vietnamese user documentation
├── HUONG_DAN.md           # Vietnamese troubleshooting guide
├── CLAUDE.md              # This file (AI assistant guide)
│
├── .gitignore             # Git ignore patterns
├── images/                # UI screenshots
│   └── ui-demo.jpg
│
└── [Generated files - may not exist in fresh clone]
    ├── metadata.json       # Processed document segments
    ├── faiss_index.bin    # FAISS vector index
    └── segment_ids.json   # Segment ID mappings
```

### File Responsibilities

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| **app.py** | Web API layer, orchestrates all components | `POST /search`, `POST /clear_history`, `route_query()` |
| **retrieval.py** | Core retrieval logic | `VectorRetriever`, `WebSearcher`, `MemoryManager` |
| **metadata.py** | Document extraction & segmentation | `extract_text_from_pdf()`, `split_into_segments()` |
| **embeddings.py** | Vector embedding generation | `create_embeddings()`, `mean_pooling()` |

---

## Architecture & Data Flow

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI App (app.py)                     │
│                     POST /search endpoint                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Query Router                           │
│         (Heuristics + LLM-based classification)            │
└──────────┬──────────────────┬─────────────────┬────────────┘
           │                  │                 │
    "retrieval"          "search"         "conversation"
           │                  │                 │
           ▼                  ▼                 ▼
┌──────────────────┐  ┌─────────────┐  ┌──────────────┐
│ VectorRetriever  │  │ WebSearcher │  │MemoryManager │
│                  │  │             │  │              │
│ 1. Query         │  │ 1. SerpAPI  │  │ 1. Chat      │
│    Decomposition │  │    Google   │  │    History   │
│ 2. FAISS Search  │  │    Search   │  │ 2. LLM       │
│ 3. Similarity    │  │ 2. LLM      │  │    Response  │
│    Filtering     │  │    Summary  │  │              │
│ 4. LLM Synthesis │  │             │  │              │
└──────────────────┘  └─────────────┘  └──────────────┘
           │                  │                 │
           └──────────────────┴─────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Response + Conversation Memory                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Pipeline (Document Processing)

```
Source Documents (PDF/DOCX in Thongtu/ folder)
           │
           ▼
[metadata.py] Extract text (OCR if needed)
           │
           ▼
[metadata.py] Extract metadata (ID, title, agency, date)
           │
           ▼
[metadata.py] Split into segments (by Article/Chapter, 256 tokens)
           │
           ▼
metadata.json (thousands of segments with metadata)
           │
           ▼
[embeddings.py] Generate vector embeddings (batch_size=32)
           │
           ▼
faiss_index.bin + segment_ids.json
           │
           ▼
[retrieval.py] FAISS index loaded for queries
```

---

## Key Components

### 1. FastAPI Application (app.py)

**Main Entry Point** for the web server.

#### API Endpoints

```python
GET  /                  # Serves index.html (frontend)
POST /search            # Main query endpoint
POST /clear_history     # Clears conversation history
```

#### Request/Response Models

```python
class QueryRequest(BaseModel):
    query: str

class RouterOutput(BaseModel):
    destination: str  # 'search' | 'retrieval' | 'conversation'
    reason: str
```

#### Configuration

```python
# Server
host = "0.0.0.0"
port = 8000
cors_origins = ["*"]  # All origins allowed

# LLM
model = "meta-llama/llama-3.3-70b-instruct:free"
api_base = "https://openrouter.ai/api/v1"
temperature = 0

# Embeddings
model = "sentence-transformers/distiluse-base-multilingual-cased-v2"
```

### 2. Vector Retrieval Engine (retrieval.py)

⚠️ **WARNING**: Contains unresolved git merge conflicts between HEAD and merged branches.

#### Key Classes

##### VectorRetriever

Core retrieval logic with multi-hop query processing.

**Key Methods**:
- `process_query(query)`: Decomposes complex queries into sub-queries using LLM
- `retrieve(queries, k=5, similarity_threshold=0.5)`: FAISS search with cosine filtering
- `get_results(query)`: End-to-end retrieval with deduplication

**Multi-Hop Example**:
```python
# Input: "So sánh mức phạt cho ô tô và xe máy?"
# Output: [
#   "*Mức phạt cho ô tô*",
#   "*Mức phạt cho xe máy*"
# ]
```

##### WebSearcher

Google Search integration via SerpAPI.

**Features**:
- Targets Vietnamese government sites (.gov.vn, thuvienphapluat.vn)
- Specifically searches for "Nghị định 168/2024/NĐ-CP"
- LLM-based result summarization

**Search Pattern**:
```python
query = f"{user_query} Nghị định 168/2024/NĐ-CP site:(.gov.vn OR thuvienphapluat.vn)"
```

##### MemoryManager

Conversation history management using LangChain.

**Methods**:
- `add_user_message(message)`
- `add_ai_message(message)`
- `get_chat_history()`

#### Query Routing Logic

```python
def route_query(query: str, chat_history: str) -> str:
    """
    Routes to:
    - "search": Keywords like "2025", "mới nhất", "tiền", "phạt bao nhiêu"
    - "retrieval": Legal questions about existing documents
    - "conversation": General chat, greetings, follow-ups
    """
    # 1. Fast path: Keyword heuristics
    if has_search_keywords(query):
        return "search"

    # 2. Fallback: LLM classification
    return llm_classify(query, chat_history)
```

### 3. Document Processing (metadata.py)

Extracts and segments legal documents.

#### Key Functions

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from PDF.
    Fallback to OCR (pytesseract) for scanned documents.
    """

def split_into_segments(text: str, max_tokens: int = 256) -> List[str]:
    """
    Intelligently splits by:
    1. "Điều X:" (Article X)
    2. "Chương X:" (Chapter X)
    3. Paragraph boundaries if no structure found
    """

def extract_metadata(text: str) -> Dict:
    """
    Extracts:
    - document_id: Regex for "XXX/YYYY/NĐ-CP"
    - title: First meaningful line
    - issuing_agency: "Bộ ...", "Chính phủ", etc.
    - date: "ngày X tháng Y năm Z"
    """
```

#### Output Format (metadata.json)

```json
{
  "document_id": "168/2024/NĐ-CP",
  "title": "Nghị định về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ",
  "issuing_agency": "Chính phủ",
  "date": "ngày 17 tháng 12 năm 2024",
  "segment_id": "168/2024/NĐ-CP_0",
  "content": "Điều 1. Phạm vi điều chỉnh..."
}
```

### 4. Vector Embedding Generation (embeddings.py)

Creates FAISS index from document segments.

#### Process

```python
def create_embeddings():
    """
    1. Load metadata.json
    2. Load transformer model (custom checkpoint)
    3. Batch process segments (batch_size=32)
    4. Create FAISS L2 index (768-dim vectors)
    5. Save to faiss_index.bin + segment_ids.json
    """
```

#### Model Configuration

```python
# Custom fine-tuned model (hardcoded path)
model_path = "/home/cong/workspace/chatbot-retrieval-based/kaggle/working/mlm_model/checkpoint-6996"

# Vector dimension
embedding_dim = 768

# FAISS index type
index = faiss.IndexFlatL2(embedding_dim)
```

---

## Development Workflows

### Initial Setup

```bash
# 1. Clone repository
git clone <repo-url>
cd rag-traffic-vn

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install system dependencies (Ubuntu/Debian)
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-vie

# 5. Set up environment variables (RECOMMENDED - currently hardcoded)
export OPENROUTER_API_KEY="sk-or-v1-..."
export SERPAPI_KEY="..."
```

### Data Processing Pipeline

```bash
# Option 1: Manual processing
python metadata.py      # Creates metadata.json
python embeddings.py    # Creates faiss_index.bin + segment_ids.json

# Option 2: Automated (if run_processing.py exists)
python run_processing.py
```

### Running the Application

```bash
# Development server
python app.py

# Production server (recommended)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing Queries

```bash
# Interactive CLI testing (if available in retrieval.py)
python retrieval.py

# API testing
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Mức phạt cho xe máy chạy quá tốc độ?"}'
```

### Git Workflow

```bash
# Current branch
git branch  # Should show: claude/claude-md-...

# Standard workflow
git add <files>
git commit -m "Descriptive message"
git push -u origin <branch-name>

# IMPORTANT: Branch must start with 'claude/' and match session ID
```

---

## Conventions & Best Practices

### Code Style

#### Python Conventions
- **Encoding**: UTF-8 with Vietnamese support
- **Indentation**: 4 spaces (not tabs)
- **Line length**: Generally < 120 characters
- **Naming**:
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

#### Type Hints
```python
# Preferred (though not consistently used in codebase)
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from PDF file."""
    pass
```

### LLM Integration Patterns

#### Prompt Engineering

**System Prompts**:
- Always reference "Nghị định 168/2024/NĐ-CP" when relevant
- Distinguish between cars (ô tô) and motorcycles (xe máy)
- Cite sources from thuvienphapluat.vn or .gov.vn domains

**Example Prompt Structure**:
```python
system_prompt = """
Bạn là trợ lý tư vấn pháp luật giao thông Việt Nam.
Tài liệu tham khảo chính: Nghị định 168/2024/NĐ-CP.
Phân biệt rõ ràng giữa ô tô và xe máy.
Trích dẫn nguồn cụ thể từ các điều luật.
"""
```

#### Temperature Settings
- **Routing/Classification**: `temperature=0` (deterministic)
- **Query Decomposition**: `temperature=0` (consistent)
- **Answer Synthesis**: `temperature=0-0.3` (factual)
- **Conversation**: `temperature=0.5-0.7` (natural)

### Vector Search Patterns

#### Similarity Thresholds

```python
# Conservative (high precision)
similarity_threshold = 0.7

# Balanced (recommended)
similarity_threshold = 0.5

# Recall-focused (more results)
similarity_threshold = 0.3
```

#### Top-K Selection

```python
# Initial FAISS retrieval
k = 10  # Get more candidates

# Post-filtering
final_results = [r for r in results if r['score'] > threshold][:5]
```

### Error Handling

#### Current Pattern (limited)
```python
try:
    result = process_query(query)
except Exception as e:
    print(f"Error: {e}")
    # No structured logging
```

#### Recommended Pattern
```python
import logging

logger = logging.getLogger(__name__)

try:
    result = process_query(query)
except FAISSError as e:
    logger.error(f"FAISS search failed: {e}")
    # Fallback to web search
except LLMError as e:
    logger.error(f"LLM inference failed: {e}")
    # Return cached/default response
```

### Configuration Management

#### Current (Hardcoded) ⚠️
```python
# app.py
openrouter_key = "sk-or-v1-..."  # SECURITY ISSUE
serpapi_key = "..."              # SECURITY ISSUE
```

#### Recommended
```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
```

### Vietnamese Language Processing

#### Text Normalization
```python
# Handle common variations
text = text.replace("ô tô", "xe ô tô")
text = text.replace("xe máy", "xe mô tô")
```

#### Segment Boundaries
```python
# Prioritize legal structure markers
patterns = [
    r"Điều \d+:",      # Article
    r"Chương [IVX]+:", # Chapter
    r"Khoản \d+",      # Clause
    r"Điểm [a-z]\)",   # Point
]
```

---

## Known Issues & Warnings

### Critical Issues

#### 1. Merge Conflicts in retrieval.py ⚠️⚠️⚠️

**Status**: UNRESOLVED

**Location**: `/home/user/rag-traffic-vn/retrieval.py`

**Description**: Git merge conflicts between two versions:
- HEAD version: More complete with `EnhancedChatbot` class
- Merged version: Alternative implementation

**Action Required**:
```bash
# Review conflicts
git diff retrieval.py

# Resolve manually or choose one version
git checkout --ours retrieval.py    # Keep HEAD version
git checkout --theirs retrieval.py  # Keep merged version
```

#### 2. Hardcoded API Keys ⚠️🔒

**Security Risk**: HIGH

**Locations**:
- `app.py`: OpenRouter API key
- `retrieval.py` (if present): SerpAPI key

**Remediation**:
```python
# Create .env file (add to .gitignore)
OPENROUTER_API_KEY=sk-or-v1-...
SERPAPI_KEY=...

# Load in code
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")
```

#### 3. Hardcoded Paths ⚠️

**Portability Issue**: HIGH

**Examples**:
```python
# embeddings.py
model_path = "/home/cong/workspace/chatbot-retrieval-based/..."
```

**Solution**: Use relative paths or environment variables

```python
import os
model_path = os.getenv("MODEL_PATH", "./models/checkpoint-6996")
```

### Non-Critical Issues

#### 4. Missing Test Coverage

**Status**: No automated tests

**Recommendation**: Add pytest-based tests
```bash
pip install pytest pytest-cov
mkdir tests/
touch tests/test_retrieval.py tests/test_metadata.py
```

#### 5. Inconsistent Embedding Models

**Issue**: Different models referenced:
- `app.py`: `distiluse-base-multilingual-cased-v2`
- `embeddings.py`: Custom checkpoint-6996

**Impact**: Embeddings may not match if models are different

**Solution**: Standardize on one model across all components

#### 6. Missing Frontend

**Issue**: `index.html` referenced but not in repository

**Impact**: GET / endpoint returns 404

**Solution**: Add frontend or remove endpoint

#### 7. No Logging

**Issue**: Limited error handling, no structured logging

**Recommendation**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

#### 8. CORS Wide Open

**Security**: `allow_origins=["*"]`

**Recommendation**: Restrict to specific domains in production
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

---

## AI Assistant Guidelines

### When Working on This Codebase

#### 1. Before Making Changes

- [ ] Check for merge conflicts (especially in `retrieval.py`)
- [ ] Verify current branch starts with `claude/`
- [ ] Read relevant sections of this CLAUDE.md
- [ ] Understand data flow for affected components

#### 2. Code Modification Guidelines

**DO**:
- ✅ Use Vietnamese language in prompts and comments when appropriate
- ✅ Maintain consistency with existing code style
- ✅ Test with Vietnamese legal queries
- ✅ Preserve backward compatibility with existing data files
- ✅ Add docstrings for new functions
- ✅ Consider both cars (ô tô) and motorcycles (xe máy) in legal contexts

**DON'T**:
- ❌ Commit API keys or secrets
- ❌ Break existing FAISS index compatibility
- ❌ Change embedding model without regenerating index
- ❌ Remove Vietnamese language support
- ❌ Push to main branch without PR
- ❌ Ignore merge conflicts

#### 3. Testing Recommendations

```python
# Test queries to validate functionality
test_queries = [
    "Mức phạt xe máy chạy quá tốc độ?",          # Retrieval
    "So sánh phạt ô tô và xe máy?",              # Multi-hop
    "Quy định mới nhất năm 2025?",               # Web search
    "Cảm ơn bạn!",                                # Conversation
]

for query in test_queries:
    response = client.post("/search", json={"query": query})
    print(f"Query: {query}")
    print(f"Route: {response.json()['route']}")
    print(f"Answer: {response.json()['answer'][:100]}...")
    print("-" * 80)
```

#### 4. Common Tasks

##### Adding New Document Sources

```python
# 1. Add PDFs/DOCX to Thongtu/ folder (gitignored)
# 2. Run metadata extraction
python metadata.py

# 3. Regenerate embeddings
python embeddings.py

# 4. Verify index
python retrieval.py
```

##### Improving Retrieval Quality

```python
# Options:
1. Adjust similarity_threshold in VectorRetriever
2. Modify query decomposition prompts
3. Add more semantic variations in preprocessing
4. Fine-tune embedding model on legal domain
```

##### Adding New Routing Rules

```python
# In route_query() function:

# 1. Add keyword heuristics (fast path)
if any(kw in query.lower() for kw in ["new_keyword", "..."]):
    return "search"

# 2. Update LLM classification prompt
system_prompt += "Route to 'search' if query mentions: ..."
```

##### Debugging Retrieval Issues

```bash
# 1. Check if index exists
ls -lh faiss_index.bin segment_ids.json metadata.json

# 2. Verify index dimensions
python -c "import faiss; idx = faiss.read_index('faiss_index.bin'); print(idx.ntotal, idx.d)"

# 3. Test query directly
python -c "from retrieval import VectorRetriever; vr = VectorRetriever(); print(vr.get_results('test query'))"

# 4. Check LLM connectivity
curl https://openrouter.ai/api/v1/models
```

#### 5. Security Checklist

Before committing:
- [ ] No API keys in code
- [ ] No absolute paths to user directories
- [ ] Sensitive data in .gitignore
- [ ] Input validation for user queries
- [ ] Rate limiting considerations
- [ ] Error messages don't expose internals

#### 6. Documentation Updates

When modifying code:
- [ ] Update this CLAUDE.md if architecture changes
- [ ] Update README.md for user-facing changes
- [ ] Update HUONG_DAN.md for troubleshooting steps
- [ ] Add inline comments for complex logic
- [ ] Update requirements.txt if dependencies change

---

## Quick Reference

### File Locations

```bash
# Code
app.py              # Web server
retrieval.py        # Core logic (HAS CONFLICTS!)
metadata.py         # Document processing
embeddings.py       # Vector generation

# Data (generated)
metadata.json       # Segmented documents
faiss_index.bin     # Vector index
segment_ids.json    # ID mappings

# Config
requirements.txt    # Dependencies
.gitignore         # Git ignore patterns
```

### Key Commands

```bash
# Setup
pip install -r requirements.txt

# Process documents
python metadata.py && python embeddings.py

# Run server
python app.py

# Test
curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query":"test"}'

# Git
git status
git add .
git commit -m "message"
git push -u origin claude/...
```

### Environment Variables (Recommended)

```bash
# Create .env file
OPENROUTER_API_KEY=sk-or-v1-...
SERPAPI_KEY=...
MODEL_PATH=./models/checkpoint-6996
HOST=0.0.0.0
PORT=8000
```

### Useful URLs

- **OpenRouter**: https://openrouter.ai/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **Vietnamese Legal Database**: https://thuvienphapluat.vn/
- **Nghị định 168/2024/NĐ-CP**: [Official government source]

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial CLAUDE.md creation |

---

## Contact & Support

For questions or issues:
1. Check HUONG_DAN.md for troubleshooting
2. Review this CLAUDE.md for context
3. Check git commit history: `git log --oneline`
4. Search existing issues/PRs

---

**Last Updated**: 2025-11-15
**Maintained By**: AI Assistant (Claude)
**Repository**: rag-traffic-vn
