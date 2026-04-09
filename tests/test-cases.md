# Test Cases — XanhSM Policy RAG System

## 1. Test Design Principles

Test cases được thiết kế để cover:

1. Retrieval correctness
2. Generation correctness
3. Hallucination prevention
4. Robustness (slang, vague)

---

## 2. Test Case Format

Mỗi test gồm:

- Query
- Ground truth answer
- Expected retrieved context
- Expected behavior
- Failure type

---

## 3. Test Cases

---

## TC1 — Basic Retrieval (Policy)

Query:
"Bao nhiêu cuốc thì được thưởng?"

Ground truth:
- 50 cuốc → 200k
- 80 cuốc → 400k

Expected retrieval:
- chunk chứa cả 2 threshold

Expected behavior:
- trả đúng
- có source

Failure type:
- Retrieval miss
- Hallucination

---

## TC2 — Sensitive Policy

Query:
"Hủy cuốc có bị khóa không?"

Expected:
- >15% warning
- >25% khóa
- disclaimer

Risk:
🚨 Critical

---

## TC3 — Paraphrase

Query:
"Huỷ nhiều có sao không?"

Expected:
- vẫn retrieve đúng

Failure:
- embedding fail

---

## TC4 — Multi-intent

Query:
"Thưởng và phạt là gì?"

Expected:
- retrieve multiple chunks
- hoặc ask clarify

---

## TC5 — Hallucination Trap

Query:
"100 cuốc thưởng bao nhiêu?"

Ground truth:
- KHÔNG có

Expected:
- "không có thông tin"

FAIL nếu:
- model bịa

---

## TC6 — No Answer Case

Query:
"Chính sách hôm nay update gì?"

Expected:
- không chắc
- suggest CSKH

---

## TC7 — Slang

Query:
"bn cuốc có thưởng"

Expected:
- normalize → retrieve đúng

---

## TC8 — Vague Query

Query:
"tiền"

Expected:
- ask clarify

---

## TC9 — Chunk Boundary Issue

Query:
"50 cuốc thưởng bao nhiêu?"

Expected:
- retrieve chunk đủ info

FAIL nếu:
- split mất info

---

## TC10 — Noise Handling

Query:
"tôi chạy 50 cuốc mà sao chưa thấy thưởng?"

Expected:
- combine:
  - policy
  - explanation

---

## 4. Coverage Summary

| Type | % |
|------|----|
| Policy | 40% |
| Edge cases | 30% |
| No answer | 20% |
| Robustness | 10% |