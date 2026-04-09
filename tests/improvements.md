# Improvements — XanhSM RAG System

## 1. Chunking Optimization

- size: 300–500 tokens
- overlap: 50–100

Impact:
- ↑ recall

---

## 2. Hybrid Search

- semantic + keyword

Impact:
- handle slang

---

## 3. Reranking Layer

- retrieve top-10
- rerank → top-3

Impact:
- ↑ precision

---

## 4. Strict Prompt

Rules:
- only answer from context
- else say "không biết"

---

## 5. Citation Enforcement

Example:
"Theo chính sách ngày 01/04/2026..."

---

## 6. Confidence Routing

if no context:
    say "không chắc"

---

## 7. Continuous Evaluation

- run test set daily
- track metrics

---

## Priority

1. Retrieval
2. Chunking
3. Prompt