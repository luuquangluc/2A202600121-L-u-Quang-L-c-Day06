import json
import time
import os
from tqdm import tqdm
from datetime import datetime

from src.chain import build_rag_chain
from src.evaluator import llm_judge, extract_score

# ===== CONFIG =====
DATASET_PATH = "test_dataset.json"
OUTPUT_DIR = "test_logs"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_FILE = f"{OUTPUT_DIR}/results_{timestamp}.json"
REPORT_FILE = f"{OUTPUT_DIR}/report_{timestamp}.txt"

# ===== LOAD =====
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

rag_chain = build_rag_chain()

# ===== UTILS =====
def normalize(text):
    return text.lower().strip()

def soft_match(pred, gt):
    pred = normalize(pred)
    gt = normalize(gt)
    return int(gt in pred or pred in gt)

def keyword_overlap(pred, gt):
    gt_words = set(normalize(gt).split())
    pred_words = set(normalize(pred).split())
    return len(gt_words & pred_words) / max(len(gt_words), 1)

results = []

# ===== RUN =====
for sample in tqdm(dataset):
    question = sample["question"]
    gt = sample["ground_truth"]

    response = rag_chain.invoke(question)

    # ---- metrics ----
    acc = soft_match(response, gt)
    overlap = keyword_overlap(response, gt)

    # ---- LLM judge ----
    judge_text = llm_judge(response, gt)
    judge_score = extract_score(judge_text)

    # 🔥 FIX: override accuracy bằng LLM
    if judge_score is not None and judge_score >= 4:
        acc = 1

    results.append({
        "id": sample["id"],
        "question": question,
        "prediction": response,
        "ground_truth": gt,
        "accuracy": acc,
        "overlap": overlap,
        "llm_score": judge_score,
        "judge_raw": judge_text
    })

    time.sleep(1)

# ===== SUMMARY =====
valid_llm_scores = [r["llm_score"] for r in results if r["llm_score"] is not None]

avg_acc = sum(r["accuracy"] for r in results) / len(results)
avg_overlap = sum(r["overlap"] for r in results) / len(results)
avg_llm = sum(valid_llm_scores) / len(valid_llm_scores) if valid_llm_scores else 0

# ===== ERROR ANALYSIS =====
error_cases = [r for r in results if r["accuracy"] == 0]

# ===== SAVE =====
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("===== EVALUATION REPORT =====\n")
    f.write(f"Time: {timestamp}\n")
    f.write(f"Total samples: {len(results)}\n\n")

    f.write("----- METRICS -----\n")
    f.write(f"Accuracy: {avg_acc:.2f}\n")
    f.write(f"Keyword Overlap: {avg_overlap:.2f}\n")
    f.write(f"LLM Score: {avg_llm:.2f}\n\n")

    f.write("----- ERROR CASES -----\n")
    for r in error_cases:
        f.write("\n---\n")
        f.write(f"Q: {r['question']}\n")
        f.write(f"Pred: {r['prediction']}\n")
        f.write(f"GT: {r['ground_truth']}\n")
        f.write(f"Overlap: {r['overlap']:.2f} | LLM: {r['llm_score']}\n")

    f.write("\n----- FULL RESULTS -----\n")
    for r in results:
        f.write("\n---\n")
        f.write(f"[{r['id']}] {r['question']}\n")
        f.write(f"Pred: {r['prediction']}\n")
        f.write(f"GT: {r['ground_truth']}\n")
        f.write(f"Acc={r['accuracy']} | Overlap={r['overlap']:.2f} | LLM={r['llm_score']}\n")

print("\n===== DONE =====")
print(f"Saved results to: {RESULT_FILE}")
print(f"Saved report to: {REPORT_FILE}")