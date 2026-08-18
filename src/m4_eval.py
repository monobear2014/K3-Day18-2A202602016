from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _clean(value) -> float:
    """Ép về float, NaN/None → 0.0 (RAGAS trả NaN khi metric không tính được)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f  # NaN != NaN


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    Trả về 4 metric aggregate + per_question. Bọc toàn bộ trong try/except vì RAGAS
    cần OPENAI_API_KEY và network — khi lỗi vẫn phải trả đúng schema để pipeline chạy tiếp.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_clean(row.get("faithfulness", 0.0)),
                answer_relevancy=_clean(row.get("answer_relevancy", 0.0)),
                context_precision=_clean(row.get("context_precision", 0.0)),
                context_recall=_clean(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]

        aggregate = {}
        for m in _METRIC_NAMES:
            values = [getattr(r, m) for r in per_question]
            aggregate[m] = sum(values) / len(values) if values else 0.0

        return {**aggregate, "per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {m: 0.0 for m in _METRIC_NAMES} | {"per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree.

    Diagnostic Tree: metric thấp nhất chỉ ra tầng nào của pipeline hỏng
    (generation vs retrieval vs ranking vs prompt).
    """
    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating — câu trả lời chứa thông tin không có trong context",
            "Siết prompt ('CHỈ dùng context'), giảm temperature, thêm câu bắt buộc trích dẫn nguồn",
        ),
        "context_recall": (
            "Missing relevant chunks — retriever không lấy đủ thông tin của ground truth",
            "Cải thiện chunking (parent-child expand), tăng top_k, thêm BM25 vào hybrid",
        ),
        "context_precision": (
            "Too many irrelevant chunks — chunk đúng bị xếp dưới chunk nhiễu",
            "Thêm/siết reranking, lọc metadata theo version, giảm top_k sau rerank",
        ),
        "answer_relevancy": (
            "Answer doesn't match question — trả lời lạc đề hoặc quá chung chung",
            "Cải thiện prompt template, thêm query rewriting, yêu cầu trả lời trực tiếp câu hỏi",
        ),
    }

    scored = []
    for r in eval_results:
        metrics = {m: getattr(r, m) for m in _METRIC_NAMES}
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "avg_score": round(avg, 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
