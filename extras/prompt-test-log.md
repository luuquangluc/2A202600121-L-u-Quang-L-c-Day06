# Prompt Test Log — RAG System

## Setup

- k = 3
- embedding = OpenAI
- chunk size = 400

---

## Test Results

### Query 1
"Bao nhiêu cuốc có thưởng?"
→ Retrieved: correct
→ Answer: correct
→ Result: ✅

---

### Query 2
"Huỷ nhiều có sao không?"
→ Retrieved: partial
→ Answer: slightly vague
→ Result: ⚠️

---

### Query 3
"100 cuốc thưởng bao nhiêu?"
→ Retrieved: none
→ Answer: "không có thông tin"
→ Result: ✅

---

### Query 4
"tiền"
→ Retrieved: none
→ Answer: clarify
→ Result: ✅

---

## Metrics

- Recall: ~90%
- Accuracy: ~88%
- Issues:
  - paraphrase retrieval
  - chunk boundary