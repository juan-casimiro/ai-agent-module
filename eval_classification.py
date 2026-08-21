#!/usr/bin/env python3
"""Measures classify_question routing accuracy against a hand-labeled set.

Calls classify_question directly — not app.invoke() — for two reasons:
it's the real production classification path, and it needs no running
RAG service. This deliberately bypasses condense_question: conversational
reference resolution is a separate concern (see classification_set.json's
schema_notes) and is not exercised here. A question's gold_category
reflects what classify_question should do with that exact string, not
what a multi-turn conversation might resolve it to.

Usage:
    python eval_classification.py
    python eval_classification.py --dataset classification_set.json --out eval_results/classification_results.json
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hello_langgraph import Category, classify_question

DEFAULT_DATASET = Path("./classification_set.json")
DEFAULT_OUT = Path("./eval_results/classification_results.json")

CATEGORIES = [c.value for c in Category]  # fixed order for the confusion matrix


def run_one(question: dict) -> dict:
    """Classify a single question and record predicted vs. gold."""
    state = {"resolved_question": question["question"]}
    result = classify_question(state)
    predicted = result["classification"]
    gold = question["gold_category"]

    return {
        "id": question["id"],
        "question": question["question"],
        "trap_class": question["trap_class"],
        "gold_category": gold,
        "predicted_category": predicted,
        "correct": predicted == gold,
    }


def build_confusion_matrix(results: list[dict]) -> dict:
    """gold -> predicted -> count, zero-filled for every category pair."""
    matrix = {gold: {pred: 0 for pred in CATEGORIES} for gold in CATEGORIES}
    for r in results:
        matrix[r["gold_category"]][r["predicted_category"]] += 1
    return matrix


def print_confusion_matrix(matrix: dict) -> None:
    header = "gold \\ pred".ljust(14) + "".join(c.ljust(14) for c in CATEGORIES)
    print(header)
    for gold in CATEGORIES:
        row = gold.ljust(14) + "".join(str(matrix[gold][pred]).ljust(14) for pred in CATEGORIES)
        print(row)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0.0

    print(f"\nOverall accuracy: {correct}/{total} ({accuracy:.1%})\n")

    print("Per-trap-class accuracy:")
    by_trap: dict[str, list[dict]] = {}
    for r in results:
        by_trap.setdefault(r["trap_class"], []).append(r)
    for trap in sorted(by_trap):
        trap_results = by_trap[trap]
        trap_correct = sum(r["correct"] for r in trap_results)
        trap_total = len(trap_results)
        print(f"  {trap:<32} {trap_correct}/{trap_total} ({trap_correct/trap_total:.1%})")

    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\nMisroutes ({len(misses)}):")
        for r in misses:
            print(f"  [{r['id']}] {r['trap_class']}: gold={r['gold_category']} "
                  f"predicted={r['predicted_category']}")
            print(f"       {r['question']}")

    print("\nConfusion matrix (gold rows, predicted columns):")
    print_confusion_matrix(build_confusion_matrix(results))


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure classify_question routing accuracy.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                         help=f"Path to labeled question set (default: {DEFAULT_DATASET})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                         help=f"Path to write results JSON (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        return 1

    data = json.loads(args.dataset.read_text())
    questions = data["questions"]
    print(f"Loaded {len(questions)} questions from {args.dataset}\n")

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] classifying {q['id']}...", end=" ", flush=True)
        r = run_one(q)
        results.append(r)
        print("OK" if r["correct"] else f"MISS (predicted {r['predicted_category']})")

    print_summary(results)

    total = len(results)
    correct = sum(r["correct"] for r in results)
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion_matrix": build_confusion_matrix(results),
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
