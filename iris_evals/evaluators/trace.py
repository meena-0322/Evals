"""
trace.py — Tool-call trace evaluator.

Verifies that the agent called the correct OData tools (and in the
correct order) for a given input. Does not need a golden dataset —
you declare expected tool sequences per intent in eval_config.py.

Works by inspecting the OpenTelemetry spans exported during the run
OR by intercepting a structured trace_log returned by the agent.

Suitable for: any agent — verifies P1.11 called list_supplier_invoices
before deciding, P2.5 called SAP T169G before resolving, etc.
"""

from __future__ import annotations

import time
from iris_evals.schemas import EvalConfig, EvalResult, EvalType, GoldenRecord
from iris_evals._transport import call_agent


async def run(
    config: EvalConfig,
    records: list[GoldenRecord],
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for rec in records:
        expected_tools = rec.expected_tools
        if not expected_tools:
            # skip records that don't declare expected tools
            continue

        t0 = time.monotonic()
        try:
            raw = await call_agent(config, rec.input)
            latency_ms = (time.monotonic() - t0) * 1000

            actual_tools = _extract_tools_called(raw)
            passed, reason = _check_sequence(expected_tools, actual_tools)

            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.TRACE,
                passed=passed,
                score=1.0 if passed else 0.0,
                actual_output=str(actual_tools),
                expected_output=str(expected_tools),
                reason=reason,
                latency_ms=latency_ms,
                metadata={"actual_tools": actual_tools, "expected_tools": expected_tools},
            ))
        except Exception as exc:
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.TRACE,
                passed=False,
                reason=f"Trace eval error: {exc}",
            ))

    return results


def _extract_tools_called(raw: dict) -> list[str]:
    """
    Pull tool call list from agent response.
    Agents that support trace export include an 'observability' or 'trace_log'
    key in their response. Falls back to an empty list if not present.
    """
    obs = raw.get("observability") or {}

    # IRIS agent observability block
    if "tools_called" in obs:
        return obs["tools_called"]

    # flat response with trace_log
    if "trace_log" in raw:
        return [step.get("tool") for step in raw["trace_log"] if "tool" in step]

    # structured result
    result = raw.get("result") or {}
    if "tools_called" in result:
        return result["tools_called"]

    return []


def _check_sequence(expected: list[str], actual: list[str]) -> tuple[bool, str]:
    """
    Check that all expected tools appear in actual, in order.
    Extra tools in actual are allowed (agent may do more).
    """
    if not actual:
        return False, f"No tools called — expected {expected}"

    pos = 0
    for tool in expected:
        found = False
        while pos < len(actual):
            if actual[pos] == tool:
                pos += 1
                found = True
                break
            pos += 1
        if not found:
            return False, f"Expected tool '{tool}' not found in sequence {actual}"

    return True, ""
