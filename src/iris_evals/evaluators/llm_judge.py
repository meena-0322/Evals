"""
llm_judge.py — LLM-as-a-Judge evaluator using Claude.

For each record, sends the agent's actual output to Claude along with
a rubric. Claude scores each criterion on scale_min–scale_max and
returns reasoning. The weighted average determines pass/fail.

Suitable for: free-text response quality, explanation groundedness,
auto-reply tone — any output that cannot be evaluated by exact match.
"""

from __future__ import annotations

import json
import os
import time

from iris_evals import aicore
from iris_evals.schemas import (
    EvalConfig, EvalResult, EvalType, GoldenRecord, JudgeRubric
)
from iris_evals._transport import call_agent


async def run(
    config: EvalConfig,
    records: list[GoldenRecord],
    rubric: JudgeRubric,
) -> tuple[list[EvalResult], dict]:
    results: list[EvalResult] = []

    for rec in records:
        t0 = time.monotonic()
        try:
            raw = await call_agent(config, rec.input)
            latency_ms = (time.monotonic() - t0) * 1000

            actual = _extract_text(raw)
            scores, reasoning = _judge(rubric, str(rec.input), actual, config)

            weighted_avg = _weighted_average(rubric, scores)
            passed = weighted_avg >= rubric.pass_threshold

            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.LLM_AS_JUDGE,
                passed=passed,
                score=round(weighted_avg / rubric.scale_max, 4),
                actual_output=actual,
                reason=reasoning,
                latency_ms=latency_ms,
                metadata={
                    "criterion_scores": scores,
                    "weighted_avg": round(weighted_avg, 3),
                    "threshold": rubric.pass_threshold,
                    "prompt_version": os.environ.get("PROMPT_VERSION", "unknown"),
                },
            ))
        except Exception as exc:
            results.append(EvalResult(
                record_id=rec.id,
                eval_type=EvalType.LLM_AS_JUDGE,
                passed=False,
                reason=f"Judge error: {exc}",
            ))

    avg_score = (sum(r.score or 0 for r in results) / len(results)) if results else 0.0
    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    metrics = {
        "average_judge_score": round(avg_score, 4),
        "pass_rate": round(pass_rate, 4),
        "rubric_criteria": [c.name for c in rubric.criteria],
    }
    return results, metrics


def _judge(
    rubric: JudgeRubric,
    user_input: str,
    agent_output: str,
    config: EvalConfig,
) -> tuple[dict[str, float], str]:
    criteria_text = "\n".join(
        f"- {c.name} (weight {c.weight}): {c.description}"
        for c in rubric.criteria
    )
    scale = f"{rubric.scale_min}–{rubric.scale_max}"

    system = (
        "You are an expert evaluator for enterprise SAP AP AI agents. "
        "Score the agent's response on each criterion honestly and strictly."
    )
    prompt = f"""You are evaluating an AI agent response for: {config.agent_name}

USER INPUT:
{user_input}

AGENT RESPONSE:
{agent_output}

EVALUATION CRITERIA (score each {scale}):
{criteria_text}

Return ONLY valid JSON in this exact shape:
{{
  "scores": {{"<criterion_name>": <score>, ...}},
  "reasoning": "<one paragraph explaining all scores>"
}}"""

    text = aicore.chat_complete(
        system=system,
        user=prompt,
        model_name=os.environ.get("AICORE_JUDGE_MODEL", "anthropic--claude-4.6-sonnet"),
        max_tokens=512,
        temperature=0.0,
    ).strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text)
    return parsed["scores"], parsed.get("reasoning", "")


def _weighted_average(rubric: JudgeRubric, scores: dict[str, float]) -> float:
    total_weight = sum(c.weight for c in rubric.criteria)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(
        scores.get(c.name, rubric.scale_min) * c.weight
        for c in rubric.criteria
    )
    return weighted_sum / total_weight


def _extract_text(raw: dict) -> str:
    if "result" in raw:
        parts = raw["result"].get("artifacts", [{}])[0].get("parts", [{}])
        return parts[0].get("text", str(raw)) if parts else str(raw)
    for key in ("text", "response", "output", "message"):
        if key in raw:
            return str(raw[key])
    return str(raw)
