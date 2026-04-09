# XanhSM Policy RAG System

Hệ thống truy xuất thông tin chính sách XanhSM dành cho tài xế, sử dụng Retrieval-Augmented Generation (RAG).

## Cấu trúc dự án

```
├── app.py                   # Streamlit chatbot UI
├── ingest.py                # Pipeline: load → chunk → embed → store
├── config.py                # Cấu hình tập trung
├── requirements.txt         # Dependencies
├── .env.example             # Template biến môi trường
├── data/                    # Tài liệu chính sách gốc (.txt)
├── chroma_db/               # Vector store (tạo sau khi chạy ingest)
└── src/
    ├── document_loader.py   # Load tài liệu từ thư mục data/
    ├── text_splitter.py     # Chia nhỏ tài liệu thành chunks
    ├── embeddings.py        # Khởi tạo embedding model
    ├── vector_store.py      # Tạo & load ChromaDB vector store
    ├── retriever.py         # Retriever từ vector store
    └── chain.py             # RAG chain (retriever → prompt → LLM)
```

## Cài đặt

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env` từ template:

```bash
cp .env.example .env
```

Điền `OPENAI_API_KEY` vào file `.env`.

## Sử dụng

### 1. Index dữ liệu (chạy 1 lần hoặc khi dữ liệu thay đổi)

```bash
python ingest.py
```

### 2. Chạy chatbot

```bash
streamlit run app.py
```
