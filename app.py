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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iris_evals_service")


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

    # Write card to a temp file so run_self_eval can load it
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(agent_card, f)
        card_path = f.name

    try:
        report = await run_self_eval(agent_url=agent_url, card_path=card_path)
    except Exception as exc:
        logger.error("run_self_eval error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        pathlib.Path(card_path).unlink(missing_ok=True)

    return JSONResponse({
        "passed":    report.overall_passed,
        "pass_rate": report.pass_rate,
        "summary":   report.summary,
        "run_id":    report.run_id,
        "metrics":   report.metrics,
    })


# ── App ───────────────────────────────────────────────────────────────────────

app = Starlette(routes=[
    Route("/health", health,   methods=["GET"]),
    Route("/run-eval", run_eval, methods=["POST"]),
])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
