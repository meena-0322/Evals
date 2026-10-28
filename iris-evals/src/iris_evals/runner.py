"""
runner.py — EvalSuite orchestrator.

Loads the golden dataset, dispatches to the correct evaluator(s),
aggregates results, and returns a unified EvalReport.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import uuid
from pathlib import Path

from iris_evals.schemas import (
    EvalConfig, EvalReport, EvalResult, EvalType, GoldenRecord
)
from iris_evals import evaluators


class EvalSuite:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def run(self) -> EvalReport:
        """Synchronous entry point — wraps the async runner."""
        return asyncio.run(self.arun())

    async def arun(self) -> EvalReport:
        config = self.config
        records = _load_golden(config.golden_dataset_path)
        all_results: list[EvalResult] = []
        all_metrics: dict = {}

        for eval_type in config.evaluators:

            if eval_type == EvalType.DETERMINISTIC:
                res = await evaluators.deterministic.run(config, records)
                all_results.extend(res)
                all_metrics["deterministic"] = _aggregate(res)

            elif eval_type == EvalType.PROBABILISTIC:
                res, metrics = await evaluators.probabilistic.run(config, records)
                all_results.extend(res)
                all_metrics["probabilistic"] = metrics

            elif eval_type == EvalType.LLM_AS_JUDGE:
                if not config.judge_rubric:
                    raise ValueError("LLM_AS_JUDGE selected but no judge_rubric in EvalConfig")
                res, metrics = await evaluators.llm_judge.run(
                    config, records, config.judge_rubric
                )
                all_results.extend(res)
                all_metrics["llm_as_judge"] = metrics

            elif eval_type == EvalType.TRACE:
                res = await evaluators.trace.run(config, records)
                all_results.extend(res)
                all_metrics["trace"] = _aggregate(res)

            elif eval_type == EvalType.SAFETY:
                safety_records = (
                    config.safety_cases
                    or [r for r in records if "safety" in r.tags]
                    or records
                )
                res = await evaluators.safety.run(config, safety_records)
                all_results.extend(res)
                all_metrics["safety"] = _aggregate(res)

            elif eval_type == EvalType.PERFORMANCE:
                res, metrics = await evaluators.performance.run(config, records)
                all_results.extend(res)
                all_metrics["performance"] = metrics

        total  = len(all_results)
        passed = sum(1 for r in all_results if r.passed)

        return EvalReport(
            agent_name=config.agent_name,
            eval_types_run=list(config.evaluators),
            total_cases=total,
            passed=passed,
            failed=total - passed,
            pass_rate=round(passed / total, 4) if total else 0.0,
            results=all_results,
            metrics=all_metrics,
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )


# ── Golden dataset loader ──────────────────────────────────────────────────────

def _load_golden(path: str | None) -> list[GoldenRecord]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenRecord(**r) for r in data]


def _aggregate(results: list[EvalResult]) -> dict:
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total":     total,
        "passed":    passed,
        "failed":    total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }
