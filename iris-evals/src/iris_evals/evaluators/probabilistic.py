"""
probabilistic.py — Accuracy, Precision, Recall, F1 for classification agents.

Runs every golden record through the agent, compares the predicted label
to expected_label, then computes per-class and macro-averaged metrics.

Suitable for: P1.2 AP Mailbox Triage (Invoice/Statement/Dunning/VendorQuery/Spam)
and any agent that classifies inputs into discrete categories.
"""

from __future__ import annotations

import time
from collections import defaultdict

from iris_evals.schemas import EvalConfig, EvalResult, EvalType, GoldenRecord
from iris_evals._transport import call_agent


async def run(config: EvalConfig, records: list[GoldenRecord]) -> tuple[list[EvalResult], dict]:
    results: list[EvalResult] = []

    # confusion matrix: {actual_label: {predicted_label: count}}
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in records:
        t0 = time.monotonic()
        try:
            raw = await call_agent(config, rec.input)
            latency_ms = (time.monotonic() - t0) * 1000

            predicted = _extract_label(raw)
            expected  = (rec.expected_label or "").strip()

            passed = predicted.upper() == expected.upper()
            confusion[expected.upper()][predicted.upper()] += 1

            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.PROBABILISTIC,
                passed=passed,
                score=1.0 if passed else 0.0,
                actual_output=predicted,
                expected_output=expected,
                reason="" if passed else f"Expected '{expected}', got '{predicted}'",
                latency_ms=latency_ms,
            ))
        except Exception as exc:
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.PROBABILISTIC,
                passed=False,
                reason=f"Agent call failed: {exc}",
            ))

    metrics = _compute_metrics(confusion, results)
    return results, metrics


def _extract_label(raw: dict) -> str:
    """Pull document_type / category / label from agent response."""
    for key in ("document_type", "category", "label", "classification", "intent"):
        if key in raw:
            return str(raw[key])
    # nested result key
    result = raw.get("result") or {}
    for key in ("document_type", "category", "label"):
        if key in result:
            return str(result[key])
    return "UNKNOWN"


def _compute_metrics(
    confusion: dict[str, dict[str, int]],
    results: list[EvalResult],
) -> dict:
    labels = set(confusion.keys())
    per_class: dict[str, dict] = {}

    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[other].get(label, 0) for other in labels if other != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0.0)

        per_class[label] = {"precision": round(precision, 4),
                            "recall":    round(recall, 4),
                            "f1":        round(f1, 4),
                            "support":   sum(confusion[label].values())}

    total     = len(results)
    correct   = sum(1 for r in results if r.passed)
    accuracy  = correct / total if total else 0.0
    macro_p   = sum(v["precision"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    macro_r   = sum(v["recall"]    for v in per_class.values()) / len(per_class) if per_class else 0.0
    macro_f1  = sum(v["f1"]        for v in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "accuracy":         round(accuracy, 4),
        "macro_precision":  round(macro_p, 4),
        "macro_recall":     round(macro_r, 4),
        "macro_f1":         round(macro_f1, 4),
        "per_class":        per_class,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
