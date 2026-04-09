from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Provider: "openai" | "gemini"
EMBEDDING_PROVIDER = "openai"
LLM_PROVIDER = "openai"

# OpenAI defaults
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# Gemini defaults (chỉ dùng khi *_PROVIDER = "gemini")
GEMINI_EMBEDDING_MODEL = "text-embedding-004"
GEMINI_LLM_MODEL = "gemini-2.0-flash"
LLM_TEMPERATURE = 0.0

RETRIEVER_TOP_K = 5
BM25_TOP_K = 5

# Trọng số hybrid search: [BM25, Dense]  — tổng = 1.0
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6

CHUNKS_PATH = BASE_DIR / "chroma_db" / "chunks.pkl"

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về chính sách của ứng dụng XanhSM dành cho tài xế.
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp bên dưới.
Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói rõ rằng bạn không có thông tin để trả lời.
Luôn trả lời bằng tiếng Việt, rõ ràng và chính xác.

Ngữ cảnh:
{context}
"""
