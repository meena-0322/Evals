"""
_transport.py — Agent call transport layer.

All evaluators call the agent through this single function.
Supports:
  - A2A JSON-RPC (message/send) — default for IRIS agents
  - Plain HTTP POST (for non-A2A agents or direct endpoint testing)

The agent URL and transport mode come from EvalConfig.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from iris_evals.schemas import EvalConfig

_TIMEOUT = 60.0


async def call_agent(config: EvalConfig, input_data: str | dict) -> dict:
    """Send input to the agent and return the parsed response dict."""
    payload = input_data if isinstance(input_data, str) else json.dumps(input_data)

    if config.use_a2a:
        return await _call_a2a(config.agent_url, payload)
    else:
        return await _call_http(config.agent_url, payload)


async def _call_a2a(base_url: str, text: str) -> dict:
    """
    SAP Joule A2A JSON-RPC 2.0 transport.
    Mirrors the pattern used by a2a_client.py in every IRIS agent.
    """
    request_id = str(uuid.uuid4())
    body = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": {
                "messageId":  str(uuid.uuid4()),
                "contextId":  str(uuid.uuid4()),
                "role":       "user",
                "parts":      [{"kind": "text", "text": text}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(base_url, json=body)
        resp.raise_for_status()
        rpc = resp.json()

    result    = rpc.get("result") or {}
    artifacts = result.get("artifacts", [])
    if not artifacts:
        return {"raw_rpc": rpc}

    parts = artifacts[0].get("parts", [{}])
    text_out = parts[0].get("text", "") if parts else ""
    try:
        parsed = json.loads(text_out)
        parsed["result"] = result          # keep A2A envelope for trace evaluator
        return parsed
    except json.JSONDecodeError:
        return {"result": result, "text": text_out}


async def _call_http(url: str, text: str) -> dict[str, Any]:
    """Plain HTTP POST — for non-A2A endpoints."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json={"input": text})
        resp.raise_for_status()
        return resp.json()
