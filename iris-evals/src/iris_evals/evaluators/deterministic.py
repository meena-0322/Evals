"""
deterministic.py — Exact-match evaluation against a golden dataset.

Sends each golden record to the agent, compares actual output to
expected_output / expected_verdict. No LLM involved.

Suitable for: P1.11 Duplicate Detection, P2.5 Tolerance Resolution,
and any fully deterministic rule-based agent.
"""

from __future__ import annotations

import time
from iris_evals.schemas import EvalConfig, EvalResult, EvalType, GoldenRecord
from iris_evals._transport import call_agent


async def run(config: EvalConfig, records: list[GoldenRecord]) -> list[EvalResult]:
    results: list[EvalResult] = []

    for rec in records:
        t0 = time.monotonic()
        try:
            raw = await call_agent(config, rec.input)
            latency_ms = (time.monotonic() - t0) * 1000

            actual = _extract_verdict_or_text(raw)
            expected = rec.expected_verdict or rec.expected_output or ""

            passed = _matches(actual, expected)
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.DETERMINISTIC,
                passed=passed,
                score=1.0 if passed else 0.0,
                actual_output=actual,
                expected_output=expected,
                reason="" if passed else f"Expected '{expected}', got '{actual}'",
                latency_ms=latency_ms,
            ))
        except Exception as exc:
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.DETERMINISTIC,
                passed=False,
                reason=f"Agent call failed: {exc}",
            ))

    return results


def _extract_verdict_or_text(raw: dict) -> str:
    """Pull verdict / invoice_resolution / text from the agent response."""
    # structured SAP agent response shapes
    for key in ("verdict", "invoice_resolution", "resolution", "status"):
        if key in raw:
            return str(raw[key])
    # A2A artifact text fallback
    if "result" in raw:
        parts = raw["result"].get("artifacts", [{}])[0].get("parts", [{}])
        return parts[0].get("text", str(raw)) if parts else str(raw)
    return str(raw)


def _matches(actual: str, expected: str) -> bool:
    return actual.strip().upper() == expected.strip().upper()
