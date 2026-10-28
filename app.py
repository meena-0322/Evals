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
      "agent_url":     "https://my-agent.cfapps.eu10.hana.ondemand.com",
      "agent_card":    { ...full agent-card.json content... },
      "golden_dataset": [ ...list of golden records... ]   // required for deterministic evals
    }
    Response:
    {
      "passed": true,
      "pass_rate": 1.0,
      "summary": "...",
      "run_id": "a3f2b8c1",
      "metrics": { ... }
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    agent_url      = body.get("agent_url")
    agent_card     = body.get("agent_card")
    golden_dataset = body.get("golden_dataset")  # optional list of golden records

    if not agent_url or not agent_card:
        return JSONResponse(
            {"error": "agent_url and agent_card are required"},
            status_code=400,
        )

    # Enforce: deterministic evals require a golden_dataset
    evaluators_requested = agent_card.get("x_evals", {}).get("evaluators", ["deterministic"])
    if "deterministic" in evaluators_requested and not golden_dataset:
        return JSONResponse(
            {"error": "golden_dataset is required when using deterministic evals. "
                      "Pass a list of golden records in the request body, "
                      "or switch to llm_as_judge in your agent card x_evals.evaluators."},
            status_code=400,
        )

    # Write card (with injected dataset path if provided) to temp files
    import tempfile, pathlib
    tmp_files = []

    try:
        # If golden_dataset provided, write it to a temp JSON file and inject path into card
        if golden_dataset:
            ds_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
            json.dump(golden_dataset, ds_file)
            ds_file.close()
            tmp_files.append(ds_file.name)
            # Inject dataset path into card so self_eval can find it
            agent_card.setdefault("x_evals", {})["dataset"] = ds_file.name

        card_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(agent_card, card_file)
        card_file.close()
        tmp_files.append(card_file.name)

        report = await run_self_eval(agent_url=agent_url, card_path=card_file.name)
    except Exception as exc:
        logger.error("run_self_eval error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        for f in tmp_files:
            pathlib.Path(f).unlink(missing_ok=True)

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
