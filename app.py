"""
iris-evals service — standalone BTP CF deployment.

Exposes a simple HTTP API so any IRIS agent can trigger evals
without bundling the library inside its own container.

Endpoints:
  POST /run-eval          — run evals for an agent (pass agent_url + card JSON)
  GET  /health            — liveness check
"""

import asyncio
import json
import logging
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from iris_evals import run_self_eval
from iris_evals.schemas import EvalConfig, EvalType, GoldenRecord
from iris_evals.registry import get_dataset, get_rubric, get_thresholds
import iris_evals.aicore as _aicore
import base64, time, httpx as _httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_evals_service")


# ── Patch aicore.chat_complete to use templating_module_config (SAP AI Core requirement) ──
def _patched_chat_complete(system, user, *, model_name="gpt-4o", max_tokens=512, temperature=0.0):
    url = os.environ["ORCHESTRATION_URL"].rstrip("/") + "/completion"
    resource_group = os.environ.get("AICORE_RESOURCE_GROUP", "default")
    payload = {
        "orchestration_config": {
            "module_configurations": {
                "llm_module_config": {
                    "model_name": model_name,
                    "model_params": {"max_tokens": max_tokens, "temperature": temperature},
                },
                "templating_module_config": {
                    "template": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ]
                },
            }
        },
    }
    resp = _httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {_aicore._get_token()}",
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = (data.get("orchestration_result", data).get("choices", [{}]))
    return choices[0].get("message", {}).get("content", "")

_aicore.chat_complete = _patched_chat_complete


# ── Endpoints ─────────────────────────────────────────────────────────────────

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "iris-evals"})


async def run_eval(request: Request) -> JSONResponse:
    """
    POST /run-eval
    Body:
    {
      "agent_url": "https://my-agent.cfapps.eu10.hana.ondemand.com",
      "agent_card": { ...full agent-card.json content... }
    }
    Response:
    {
      "passed": true,
      "pass_rate": 1.0,
      "summary": "...",
      "run_id": "a3f2b8c1"
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    agent_url = body.get("agent_url")
    agent_card = body.get("agent_card")

    if not agent_url or not agent_card:
        return JSONResponse(
            {"error": "agent_url and agent_card are required"},
            status_code=400,
        )

    # Build config directly from request body — bypasses registry defaults
    from iris_evals.schemas import EvalConfig, EvalType, GoldenRecord, JudgeRubric, JudgeCriterion
    from iris_evals.runner import EvalSuite

    agent_name = agent_card.get("name", "unknown-agent")
    raw_types  = agent_card.get("x_evals", {}).get("evaluators", ["llm_as_judge"])
    eval_types = [EvalType(t) for t in raw_types]

    # Golden dataset from request body
    raw_dataset = body.get("golden_dataset", [])
    records = [GoldenRecord(**r) for r in raw_dataset]

    # Rubric from request body
    rubric = None
    raw_rubric = body.get("rubric")
    if raw_rubric and EvalType.LLM_AS_JUDGE in eval_types:
        rubric = JudgeRubric(
            criteria=[JudgeCriterion(**c) for c in raw_rubric.get("criteria", [])],
            scale_min=raw_rubric.get("scale_min", 1),
            scale_max=raw_rubric.get("scale_max", 5),
            pass_threshold=raw_rubric.get("pass_threshold", 3.5),
        )

    config = EvalConfig(
        agent_name   = agent_name,
        agent_url    = agent_url,
        evaluators   = eval_types,
        judge_rubric = rubric,
        judge_model  = os.environ.get("AICORE_JUDGE_MODEL", "anthropic--claude-4.6-sonnet"),
        use_a2a      = True,
        golden_records = records,
    )

    try:
        from iris_evals.self_eval import _SelfEvalSuite
        suite  = _SelfEvalSuite(config)
        report = await suite.arun()
    except Exception as exc:
        logger.error("eval error: %s", exc, exc_info=True)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse({
        "passed":    report.overall_passed,
        "pass_rate": report.pass_rate,
        "summary":   report.summary,
        "run_id":    report.run_id,
        "metrics":   report.metrics,
        "results":   [r.model_dump() for r in report.results],
    })


# ── App ───────────────────────────────────────────────────────────────────────

app = Starlette(routes=[
    Route("/health", health,   methods=["GET"]),
    Route("/run-eval", run_eval, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # CF injects PORT automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
