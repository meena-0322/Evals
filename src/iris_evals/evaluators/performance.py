"""
performance.py — Latency and token-usage evaluator.

Runs every record through the agent, measures wall-clock latency,
and optionally checks token counts from the observability block.

Does NOT need a golden dataset — any set of representative inputs works.

Suitable for: all agents before CF push to verify SLA compliance.
"""

from __future__ import annotations

import time

from iris_evals.schemas import EvalConfig, EvalResult, EvalType, GoldenRecord
from iris_evals._transport import call_agent


async def run(
    config: EvalConfig,
    records: list[GoldenRecord],
) -> tuple[list[EvalResult], dict]:
    results: list[EvalResult] = []
    latencies: list[float] = []

    for rec in records:
        max_ms = rec.max_latency_ms or config.max_latency_ms

        t0 = time.monotonic()
        try:
            raw = await call_agent(config, rec.input)
            latency_ms = (time.monotonic() - t0) * 1000
        except Exception as exc:
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.PERFORMANCE,
                passed=False,
                reason=f"Agent call failed: {exc}",
            ))
            continue

        tokens = _extract_tokens(raw)
        latencies.append(latency_ms)

        within_latency = latency_ms <= max_ms
        within_tokens  = (tokens is None or config.max_tokens is None
                          or tokens <= config.max_tokens)
        passed = within_latency and within_tokens

        reasons = []
        if not within_latency:
            reasons.append(f"Latency {latency_ms:.0f}ms > limit {max_ms:.0f}ms")
        if not within_tokens and tokens is not None:
            reasons.append(f"Tokens {tokens} > limit {config.max_tokens}")

        results.append(EvalResult(
            record_id=rec.id,
            eval_type=EvalType.PERFORMANCE,
            passed=passed,
            score=min(1.0, max_ms / latency_ms) if latency_ms > 0 else 1.0,
            latency_ms=latency_ms,
            tokens_used=tokens,
            reason="; ".join(reasons),
            metadata={"max_latency_ms": max_ms},
        ))

    metrics = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        metrics = {
            "avg_latency_ms":  round(sum(latencies) / n, 1),
            "min_latency_ms":  round(min(latencies), 1),
            "max_latency_ms":  round(max(latencies), 1),
            "p50_latency_ms":  round(latencies_sorted[n // 2], 1),
            "p95_latency_ms":  round(latencies_sorted[int(n * 0.95)], 1),
            "pass_rate":       round(sum(1 for r in results if r.passed) / n, 4),
        }

    return results, metrics


def _extract_tokens(raw: dict) -> int | None:
    obs = raw.get("observability") or {}
    if "tokens_used" in obs:
        return int(obs["tokens_used"])
    if "usage" in raw:
        usage = raw["usage"]
        return usage.get("total_tokens") or usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return None
