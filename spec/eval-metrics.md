# Eval Metrics — XanhSM Policy RAG System

## 1. Mục tiêu evaluation

Hệ thống RAG có 2 nhiệm vụ chính:
1. **Retrieve đúng thông tin từ knowledge base**
2. **Generate câu trả lời KHÔNG sai và KHÔNG bịa**

→ Vì vậy evaluation phải tách rõ:
- Retrieval quality
- Generation quality
- End-to-end UX

---

## 2. Kiến trúc pipeline (để hiểu metric đo ở đâu)

User Query
   ↓
Retriever (ChromaDB)
   ↓
Top-k Chunks
   ↓
LLM (Prompt + Context)
   ↓
Final Answer

→ Metrics sẽ đo ở từng stage:
- Retrieval (trước LLM)
- Generation (sau LLM)

---

## 3. Core Metrics

## 3.1 Context Recall (MOST IMPORTANT)

### Definition
Retriever có tìm được chunk chứa câu trả lời đúng hay không

### Why important
Nếu retrieval fail → LLM chắc chắn sai hoặc hallucinate

### Cách đo

Với mỗi query:
- Ground truth: biết câu trả lời nằm ở document nào
- Check top-k chunks:
  - nếu có chunk chứa answer → success

### Formula

Context Recall = (# queries có relevant chunk) / (total queries)

### Ví dụ

Query: "Hủy cuốc có bị khóa không?"

→ Nếu retrieve được chunk chứa:
- >15% warning
- >25% khóa

→ count = success

### Threshold
- ≥ 90%

### Red flag (critical)
- KB có answer nhưng không retrieve được

---

## 3.2 Context Precision

### Definition
Trong top-k chunks, bao nhiêu chunk là relevant

### Formula

Precision = relevant_chunks / total_chunks

### Trade-off
- k lớn → recall cao, precision thấp
- k nhỏ → precision cao, miss info

### Threshold
- ≥ 70%

### Insight (điểm ăn nói demo)
> "RAG không chỉ cần recall cao mà còn cần precision để tránh noise làm model confuse"

---

## 3.3 Answer Accuracy

### Definition
Câu trả lời có đúng policy không

### Label schema

- ✅ Correct (đúng hoàn toàn)
- ⚠️ Partial (thiếu 1 phần)
- ❌ Incorrect (sai)
- 🚨 Hallucinated (bịa)

### Formula

Accuracy = (Correct + 0.5 × Partial) / Total

### Threshold
- ≥ 90%

### Red flag
- Hallucinated policy → FAIL hệ thống

---

## 3.4 Groundedness (RAG-specific)

### Definition
Câu trả lời có **chỉ dựa trên context** không

### Cách đo

- So sánh answer với retrieved context
- Check:
  - Có nằm trong context không
  - Có thêm thông tin ngoài không

### Good example
"50 cuốc → 200k" (có trong doc)

### Bad example
"60 cuốc → 300k" (không có trong doc)

### Threshold
- ≥ 95%

### Insight (nói demo)
> “Groundedness quan trọng hơn fluency — đúng còn hơn hay”

---

## 3.5 No-Hallucination Rate

### Definition
% câu trả lời KHÔNG bịa

### Formula

No-Hallucination = 1 - (Hallucinated / Total)

### Threshold
- ≥ 98%

---

## 3.6 Latency

### Definition
Thời gian end-to-end

### Breakdown
- retrieval: ~100–300ms
- LLM: ~1–3s

### Threshold
- ≤ 5s

---

## 4. Evaluation Dataset

### Size
- 20–30 queries

### Coverage
- thưởng
- phạt
- thu nhập
- edge cases
- no-answer cases

### Important (ăn điểm)
Phải có:
- 20% queries KHÔNG có answer → test hallucination

---

## 5. Evaluation Pipeline

1. Chuẩn bị test set
2. Run toàn bộ queries
3. Log:
   - retrieved chunks
   - final answer
4. Human label:
   - correctness
   - grounding
5. Tính metrics

---

## 6. Final Threshold Summary

| Metric | Target |
|------|-------|
| Context Recall | ≥ 90% |
| Context Precision | ≥ 70% |
| Accuracy | ≥ 90% |
| Groundedness | ≥ 95% |
| No Hallucination | ≥ 98% |

---

## 7. Key Insight (nói demo)

- 80% lỗi đến từ retrieval, không phải LLM
- Nếu fail → fix retriever trước, không fix prompt