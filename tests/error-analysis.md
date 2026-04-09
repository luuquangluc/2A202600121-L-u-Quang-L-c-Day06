# Error Analysis — XanhSM RAG System

## 1. Failure Taxonomy

RAG có 3 tầng lỗi:

1. Retrieval errors
2. Generation errors
3. Data errors

---

## 2. Error 1 — Retrieval Miss

### Root cause
- embedding không hiểu tiếng Việt
- query khác wording

### Example
"huỷ nhiều có sao không" → miss

### Impact
- LLM không có context → hallucinate

### Fix
- hybrid search
- better embeddings

---

## 3. Error 2 — Chunking Failure

### Root cause
- chunk split sai

### Example
- "50 cuốc" ở chunk A
- "200k" ở chunk B

### Impact
- incomplete answer

### Fix
- overlap
- semantic chunking

---

## 4. Error 3 — Irrelevant Retrieval

### Root cause
- similarity search yếu

### Impact
- noise → confuse model

### Fix
- reranking

---

## 5. Error 4 — Hallucination

### Root cause
- context thiếu
- prompt không strict

### Impact
🚨 Critical

### Fix
- strict grounding prompt

---

## 6. Error 5 — Context Ignored

### Root cause
- prompt yếu

### Fix
- force citation

---

## 7. Key Insight

> "Improving RAG = improving retrieval, not LLM"