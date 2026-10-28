"""
iris-evals — Reusable evaluation framework for IRIS SAP Joule A2A agents.

── Startup self-eval (simplest usage) ──────────────────────────────────────────
    from iris_evals import run_self_eval

    @app.on_event("startup")
    async def startup():
        report = await run_self_eval(agent_url="http://localhost:8080")
        if not report.overall_passed:
            logger.warning(f"[IRIS EVALS] {report.summary}")

── Background loop ──────────────────────────────────────────────────────────────
    from iris_evals import self_eval_loop

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(self_eval_loop(agent_url="http://localhost:8080", interval_seconds=300))

── Health endpoint ───────────────────────────────────────────────────────────────
    @app.get("/health/evals")
    async def eval_health():
        report = await run_self_eval(agent_url="http://localhost:8080")
        return {"passed": report.overall_passed, "summary": report.summary}

── Manual / CLI ─────────────────────────────────────────────────────────────────
    iris-evals run --agent http://localhost:8080
    iris-evals run-pipeline --pipeline pipeline.json
"""

from iris_evals.runner    import EvalSuite
from iris_evals.self_eval import run_self_eval, self_eval_loop
from iris_evals.schemas   import (
    EvalConfig,
    EvalType,
    EvalReport,
    EvalResult,
    GoldenRecord,
    JudgeRubric,
    JudgeCriterion,
)
from iris_evals.results   import print_report, save_json
from iris_evals.registry  import register_agent

__all__ = [
    # self-eval (primary API)
    "run_self_eval",
    "self_eval_loop",
    # manual suite
    "EvalSuite",
    # schemas
    "EvalConfig",
    "EvalType",
    "EvalReport",
    "EvalResult",
    "GoldenRecord",
    "JudgeRubric",
    "JudgeCriterion",
    # output
    "print_report",
    "save_json",
    # registry extension
    "register_agent",
]
