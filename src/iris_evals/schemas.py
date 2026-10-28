"""
schemas.py — Core data models for iris-evals.

All eval input/output shapes are defined here so every evaluator
and the runner share the same types.
"""

from __future__ import annotations

import enum
from typing import Any
from pydantic import BaseModel, Field


# ── Eval type catalogue ────────────────────────────────────────────────────────

class EvalType(str, enum.Enum):
    DETERMINISTIC  = "deterministic"   # exact match against golden dataset
    PROBABILISTIC  = "probabilistic"   # accuracy / precision / recall / F1
    LLM_AS_JUDGE   = "llm_as_judge"    # Claude scores free-text output
    AGENT_AS_JUDGE = "agent_as_judge"  # another A2A agent acts as judge
    TRACE          = "trace"           # verifies correct tools were called
    SAFETY         = "safety"          # business-rule / escalation assertions
    PERFORMANCE    = "performance"     # latency + token-count checks


# ── Golden dataset record ──────────────────────────────────────────────────────

class GoldenRecord(BaseModel):
    """One test case in the golden dataset."""
    id: str
    input: str | dict                          # raw text or structured payload
    expected_output: str | None = None         # DETERMINISTIC: exact expected text
    expected_label: str | None  = None         # PROBABILISTIC: class label
    expected_verdict: str | None = None        # e.g. DUPLICATE / CLEAR / RESOLVED
    expected_tools: list[str] = Field(default_factory=list)  # TRACE: ordered tool list
    safety_rule: str | None = None             # SAFETY: rule description
    expected_safe: bool = True                 # SAFETY: should agent comply?
    tags: list[str] = Field(default_factory=list)
    max_latency_ms: float | None = None        # PERFORMANCE: per-record override


# ── Judge rubric (LLM_AS_JUDGE) ────────────────────────────────────────────────

class JudgeCriterion(BaseModel):
    name: str
    description: str
    weight: float = 1.0   # relative weight when computing weighted score


class JudgeRubric(BaseModel):
    criteria: list[JudgeCriterion]
    scale_min: int = 1
    scale_max: int = 5
    pass_threshold: float = 3.5   # minimum weighted average to pass


# ── Per-agent eval configuration ───────────────────────────────────────────────

class EvalConfig(BaseModel):
    agent_name: str
    agent_url: str = "http://localhost:8080"

    evaluators: list[EvalType] = Field(default_factory=list)

    # DETERMINISTIC / PROBABILISTIC
    golden_dataset_path: str | None = None

    # LLM_AS_JUDGE
    judge_rubric: JudgeRubric | None = None
    judge_model: str = "claude-sonnet-4-6"   # model used as judge

    # AGENT_AS_JUDGE
    judge_agent_url: str | None = None

    # TRACE — expected tool sequence per intent keyword
    expected_tool_sequences: dict[str, list[str]] = Field(default_factory=dict)

    # SAFETY — list of safety test records (can also live in golden_dataset)
    safety_cases: list[GoldenRecord] = Field(default_factory=list)

    # PERFORMANCE
    max_latency_ms: float = 5000.0
    max_tokens: int | None = None

    # Thresholds (override defaults)
    thresholds: dict[str, float] = Field(default_factory=dict)

    # A2A transport — set if agent uses JSON-RPC message/send
    use_a2a: bool = True

    # Pre-loaded records injected by run_self_eval() — skips file loading in runner
    golden_records: list[Any] = Field(default_factory=list, exclude=True)


# ── Individual eval result ─────────────────────────────────────────────────────

class EvalResult(BaseModel):
    record_id: str
    eval_type: EvalType
    passed: bool
    score: float | None = None        # 0.0–1.0 where applicable
    actual_output: str | None = None
    expected_output: str | None = None
    reason: str = ""
    latency_ms: float | None = None
    tokens_used: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Aggregated report ──────────────────────────────────────────────────────────

class EvalReport(BaseModel):
    agent_name: str
    eval_types_run: list[EvalType]
    total_cases: int
    passed: int
    failed: int
    pass_rate: float                  # 0.0–1.0
    results: list[EvalResult]
    metrics: dict[str, Any] = Field(default_factory=dict)  # per-type aggregates
    prompt_version: str | None = None
    run_id: str = ""
    timestamp: str = ""

    @property
    def overall_passed(self) -> bool:
        return self.failed == 0

    @property
    def summary(self) -> str:
        pct = round(self.pass_rate * 100, 1)
        status = "PASS" if self.overall_passed else "FAIL"
        return f"{status} {pct}% ({self.passed}/{self.total_cases}) run_id={self.run_id}"
