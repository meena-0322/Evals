"""
registry.py — Built-in golden datasets, rubrics, and threshold profiles.

All IRIS SAP Joule agent types are registered here.
Agent projects need ZERO data files — everything lives in this module.
"""

from __future__ import annotations

from iris_evals.schemas import GoldenRecord, JudgeCriterion, JudgeRubric

# ── Threshold profiles ─────────────────────────────────────────────────────────

THRESHOLD_PROFILES: dict[str, dict[str, float]] = {
    "deterministic_strict": {
        "deterministic": 1.0,
        "safety":        1.0,
        "trace":         1.0,
    },
    "llm_defaults": {
        "llm_as_judge":  0.75,
        "probabilistic": 0.85,
        "safety":        1.0,
        "performance":   0.90,
    },
    "relaxed": {
        "deterministic": 0.90,
        "probabilistic": 0.80,
        "safety":        1.0,
    },
}

# ── Built-in rubrics ───────────────────────────────────────────────────────────

RUBRICS: dict[str, JudgeRubric] = {
    "ap_auto_reply": JudgeRubric(
        criteria=[
            JudgeCriterion(name="accuracy",       description="Response correctly identifies document type and required action", weight=2.0),
            JudgeCriterion(name="completeness",   description="All required fields (vendor, amount, PO) are addressed",         weight=1.5),
            JudgeCriterion(name="tone",           description="Professional, concise, no hallucinated data",                   weight=1.0),
            JudgeCriterion(name="escalation",     description="Correctly escalates ambiguous or high-risk cases",              weight=1.5),
            JudgeCriterion(name="sap_alignment",  description="Suggested action aligns with SAP workflow state",              weight=2.0),
        ],
        pass_threshold=3.5,
    ),
    "general_quality": JudgeRubric(
        criteria=[
            JudgeCriterion(name="accuracy",     description="Answer is factually correct",         weight=2.0),
            JudgeCriterion(name="completeness", description="All parts of the question are addressed", weight=1.5),
            JudgeCriterion(name="clarity",      description="Response is clear and well-structured",  weight=1.0),
            JudgeCriterion(name="safety",       description="No PII, no hallucinated references",     weight=2.0),
        ],
        pass_threshold=3.5,
    ),
}

# ── Golden datasets are agent-owned — no hardcoded data here ──────────────────
# Each agent passes its own golden_dataset in the /run-eval request body.
# Deterministic evals REQUIRE a golden_dataset; LLM-based evals do not.

_DATASETS: dict[str, list[dict]] = {}

# ── Registry — maps agent name → eval config ───────────────────────────────────
# Agents registered here get a rubric and threshold profile automatically.
# Add an entry when a new agent type is onboarded; dataset stays in the agent repo.

AGENT_REGISTRY: dict[str, dict] = {}


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_dataset(agent_name: str) -> list[GoldenRecord]:
    """Return built-in golden records for an agent type."""
    entry = AGENT_REGISTRY.get(agent_name)
    if not entry:
        return []
    raw = _DATASETS.get(entry["dataset_key"], [])
    return [GoldenRecord(**r) for r in raw]


def get_rubric(agent_name: str) -> JudgeRubric | None:
    """Return built-in judge rubric for an agent type, or None."""
    entry = AGENT_REGISTRY.get(agent_name)
    if not entry or not entry["rubric_key"]:
        return None
    return RUBRICS.get(entry["rubric_key"])


def get_thresholds(agent_name: str) -> dict[str, float]:
    """Return threshold profile for an agent type."""
    entry = AGENT_REGISTRY.get(agent_name)
    if not entry:
        return THRESHOLD_PROFILES["deterministic_strict"]
    profile = entry.get("threshold_profile", "deterministic_strict")
    return THRESHOLD_PROFILES.get(profile, THRESHOLD_PROFILES["deterministic_strict"])


def register_agent(
    name: str,
    dataset: list[dict],
    rubric: JudgeRubric | None = None,
    threshold_profile: str = "deterministic_strict",
) -> None:
    """Register a new agent type at runtime (for custom agents not in the built-in list)."""
    _DATASETS[name] = dataset
    AGENT_REGISTRY[name] = {
        "dataset_key":       name,
        "rubric_key":        name if rubric else None,
        "threshold_profile": threshold_profile,
    }
    if rubric:
        RUBRICS[name] = rubric
