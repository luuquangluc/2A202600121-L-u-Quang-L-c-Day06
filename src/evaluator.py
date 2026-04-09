from openai import OpenAI
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def llm_judge(pred, gt):
    prompt = f"""
You are an evaluator.

Compare the answer with the ground truth.

Answer: {pred}
Ground truth: {gt}

Score from 1 to 5:
1 = completely wrong
2 = mostly wrong
3 = partially correct
4 = mostly correct
5 = fully correct

ONLY return a number (1-5).
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content.strip()


def extract_score(text):
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else 0


def semantic_accuracy(score, threshold=4):
    """
    Convert LLM score → binary accuracy
    """
    return 1 if score >= threshold else 0