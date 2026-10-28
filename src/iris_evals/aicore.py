"""
aicore.py — SAP AI Core orchestration client for iris-evals LLM judge.

Handles OAuth2 token fetch (client credentials) and calls the
AI Core orchestration completion endpoint.

Required env vars:
    AICORE_CLIENT_ID
    AICORE_CLIENT_SECRET
    AICORE_AUTH_URL
    ORCHESTRATION_URL
    AICORE_RESOURCE_GROUP   (default: "default")
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import httpx


# ── Token cache ───────────────────────────────────────────────────────────────

_token: str | None = None
_token_expiry: float = 0.0          # epoch seconds
_TOKEN_BUFFER_SECS = 60             # refresh 60 s before actual expiry


def _fetch_token() -> str:
    client_id     = os.environ["AICORE_CLIENT_ID"]
    client_secret = os.environ["AICORE_CLIENT_SECRET"]
    auth_url      = os.environ["AICORE_AUTH_URL"]

    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    resp = httpx.post(
        auth_url,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["access_token"], int(body.get("expires_in", 3600))


def _get_token() -> str:
    global _token, _token_expiry
    now = time.monotonic()
    if _token is None or now >= _token_expiry:
        token, expires_in = _fetch_token()
        _token        = token
        _token_expiry = now + expires_in - _TOKEN_BUFFER_SECS
    return _token


# ── Completion call ───────────────────────────────────────────────────────────

def chat_complete(
    system: str,
    user: str,
    *,
    model_name: str = "anthropic--claude-4.6-sonnet",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Call the SAP AI Core orchestration service and return the assistant text.

    Uses the deployment at ORCHESTRATION_URL env var.
    The model_name must be deployed in the AI Core resource group.
    """
    url            = os.environ["ORCHESTRATION_URL"].rstrip("/") + "/completion"
    resource_group = os.environ.get("AICORE_RESOURCE_GROUP", "default")

    payload: dict[str, Any] = {
        "orchestration_config": {
            "module_configurations": {
                "llm_module_config": {
                    "model_name": model_name,
                    "model_params": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
            }
        },
        "messages_history": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }

    resp = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {_get_token()}",
            "AI-Resource-Group": resource_group,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # SAP AI Core orchestration wraps the OpenAI-style response
    choices = (
        data.get("orchestration_result", data)
           .get("choices", [{}])
    )
    return choices[0].get("message", {}).get("content", "")
