# iris-evals

Self-evaluation library for Applexus IRIS SAP Joule agents.

Install once — every agent evaluates itself on startup, runs a background loop every 5 minutes, and exposes a `/health/evals` CI gate endpoint. All golden datasets, rubrics, and thresholds are bundled inside the library. Zero eval data files needed per agent.

---

## Installation

```bash
pip install git+https://github.com/applexus/iris-evals.git@v1.0.0
```

Add to your agent's `requirements.txt`:

```
iris-evals @ git+https://github.com/applexus/iris-evals.git@v1.0.0
```

---

## Quick Start

Add three blocks to your agent's `app.py`:

```python
import asyncio, os
from iris_evals import run_self_eval, self_eval_loop

AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8080")

@app.on_event("startup")
async def startup_self_eval():
    report = await run_self_eval(agent_url=AGENT_URL)
    if not report.overall_passed:
        logger.warning("[IRIS EVALS] Self-eval FAILED — %s", report.summary)

@app.on_event("startup")
async def start_eval_loop():
    asyncio.create_task(
        self_eval_loop(agent_url=AGENT_URL, interval_seconds=300)
    )

@app.route("/health/evals", methods=["GET"])
async def eval_health(request):
    from starlette.responses import JSONResponse
    report = await run_self_eval(agent_url=AGENT_URL)
    return JSONResponse({"passed": report.overall_passed, "summary": report.summary})
```

Add to your agent's `agent-card.json`:

```json
"x_evals": {
  "evaluators": ["deterministic", "safety"],
  "dataset": "builtin",
  "threshold_profile": "deterministic_strict"
}
```

---

## 7 Eval Types

| # | Type | Needs AI | Use when |
|---|------|----------|----------|
| 1 | `deterministic` | No | Rule-based agents — exact verdict match |
| 2 | `probabilistic` | No | Classifiers — accuracy / F1 ≥ 85% |
| 3 | `llm_as_judge` | Yes (AI Core) | Free-text output quality — rubric scored |
| 4 | `agent_as_judge` | Yes (A2A) | Complex SAP domain — peer agent reviews |
| 5 | `trace` | No | Multi-step agents — correct tool call sequence |
| 6 | `safety` | No | Every agent — PII + injection checks |
| 7 | `performance` | No | SLA-critical — latency p95 ≤ 5000ms |

**Quick picks:**
- Rule-based agent → `1, 6`
- Triage / classifier → `2, 3, 6`
- Multi-step → `1, 5, 6`
- SLA-critical → `1, 5, 6, 7`

---

## How it works

```
cf push
  └─ pip installs iris-evals from GitHub
       └─ startup: run_self_eval() calls agent via A2A with golden records
            └─ CI polls GET /health/evals → passed: true → promote
                 └─ background: self_eval_loop() re-runs every 5 minutes
```

iris-evals runs **inside your agent's process** as an asyncio task — no separate BTP service, no extra cost.

---

## LLM Judge (type 3)

Uses SAP AI Core orchestration. Requires these env vars in your agent's `manifest.yml` (gitignored):

```
AICORE_CLIENT_ID
AICORE_CLIENT_SECRET
AICORE_AUTH_URL
AICORE_BASE_URL
AICORE_RESOURCE_GROUP
ORCHESTRATION_URL
AICORE_JUDGE_MODEL     (default: gpt-4o)
```

If your agent already has AI Core credentials, iris-evals picks them up automatically — no extra config.

---

## Skill Integration

The `iris-sap-agent-skill` asks **Q_EVAL** as its last interview question:

```
Which evaluation types does this agent need?
→ deterministic / probabilistic / llm_as_judge / agent_as_judge / trace / safety / performance
```

Answering generates all wiring automatically — `x_evals` block, `app.py` startup/loop/health, and `requirements.txt` entry.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Project Structure

```
src/iris_evals/
├── aicore.py          SAP AI Core orchestration client
├── registry.py        Built-in golden datasets, rubrics, thresholds
├── schemas.py         EvalType, GoldenRecord, EvalConfig, EvalReport
├── self_eval.py       run_self_eval() and self_eval_loop()
├── runner.py          EvalSuite base class
├── _transport.py      A2A JSON-RPC call helper
└── evaluators/
    ├── deterministic.py
    ├── probabilistic.py
    ├── llm_judge.py
    ├── safety.py
    ├── trace.py
    └── performance.py
```

---

## Versioning

| Version | Notes |
|---------|-------|
| v1.0.0 | Initial release — 7 eval types, built-in registry, AI Core LLM judge |

Tag a new version and update `requirements.txt` in each agent to upgrade:

```bash
git tag v1.0.1
git push --tags
```
