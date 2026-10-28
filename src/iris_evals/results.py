"""
results.py — EvalReport display and export helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from iris_evals.schemas import EvalReport, EvalType


_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def print_report(report: EvalReport) -> None:
    """Print a human-readable summary to stdout (ASCII only, no box chars)."""
    _line()
    print(f"{_BOLD}IRIS Evals — {report.agent_name}{_RESET}")
    print(f"Run ID: {report.run_id}  |  {report.timestamp}")
    _line()

    color = _GREEN if report.pass_rate >= 0.90 else (_YELLOW if report.pass_rate >= 0.70 else _RED)
    print(
        f"Total: {report.total_cases}  "
        f"Passed: {_GREEN}{report.passed}{_RESET}  "
        f"Failed: {_RED}{report.failed}{_RESET}  "
        f"Pass Rate: {color}{report.pass_rate * 100:.1f}%{_RESET}"
    )
    _line()

    # Per-type summary
    for eval_type in report.eval_types_run:
        key = eval_type.value
        m = report.metrics.get(key) or {}
        print(f"  [{_CYAN}{key.upper()}{_RESET}]")

        if "accuracy" in m:
            print(f"    Accuracy:        {m['accuracy'] * 100:.1f}%")
            print(f"    Macro Precision: {m['macro_precision'] * 100:.1f}%")
            print(f"    Macro Recall:    {m['macro_recall'] * 100:.1f}%")
            print(f"    Macro F1:        {m['macro_f1'] * 100:.1f}%")
        elif "average_judge_score" in m:
            print(f"    Avg Judge Score: {m['average_judge_score']:.3f}")
            print(f"    Pass Rate:       {m['pass_rate'] * 100:.1f}%")
        elif "avg_latency_ms" in m:
            print(f"    Avg Latency:  {m['avg_latency_ms']:.0f} ms")
            print(f"    P95 Latency:  {m['p95_latency_ms']:.0f} ms")
            print(f"    Pass Rate:    {m['pass_rate'] * 100:.1f}%")
        elif "pass_rate" in m:
            print(f"    Pass Rate: {m['pass_rate'] * 100:.1f}%  "
                  f"({m.get('passed', '?')}/{m.get('total', '?')})")

    _line()

    # Failures detail
    failures = [r for r in report.results if not r.passed]
    if failures:
        print(f"{_RED}FAILURES:{_RESET}")
        for r in failures:
            print(f"  [{r.eval_type.value}] {r.record_id}: {r.reason}")
        _line()
    else:
        print(f"{_GREEN}All cases passed.{_RESET}")
        _line()


def save_json(report: EvalReport, path: str) -> None:
    """Save full report as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"Report saved to: {path}")


def _line() -> None:
    print("-" * 60)
