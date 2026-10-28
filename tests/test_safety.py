"""Tests for the safety evaluator."""

import pytest
from unittest.mock import AsyncMock, patch

from iris_evals.evaluators.safety import run, _check_safety, _flatten
from iris_evals.schemas import EvalConfig, EvalType, GoldenRecord


@pytest.fixture
def config():
    return EvalConfig(agent_name="test-agent", evaluators=[EvalType.SAFETY])


def _rec(rid, safe, rule="test rule"):
    return GoldenRecord(id=rid, input="test", expected_safe=safe, safety_rule=rule, tags=["safety"])


@pytest.mark.asyncio
async def test_safe_input_passes_when_not_blocked(config):
    rec = _rec("s1", expected_safe=True)
    with patch("iris_evals.evaluators.safety.call_agent",
               new=AsyncMock(return_value={"verdict": "CLEAR"})):
        results = await run(config, [rec])
    assert results[0].passed


@pytest.mark.asyncio
async def test_safe_input_fails_when_wrongly_blocked(config):
    rec = _rec("s2", expected_safe=True, rule="legitimate invoice must clear")
    with patch("iris_evals.evaluators.safety.call_agent",
               new=AsyncMock(return_value={"verdict": "BLOCKED"})):
        results = await run(config, [rec])
    assert not results[0].passed
    assert "incorrectly BLOCKED" in results[0].reason


@pytest.mark.asyncio
async def test_unsafe_input_must_be_blocked(config):
    rec = _rec("s3", expected_safe=False, rule="duplicate must be blocked")
    with patch("iris_evals.evaluators.safety.call_agent",
               new=AsyncMock(return_value={"verdict": "DUPLICATE"})):
        results = await run(config, [rec])
    assert results[0].passed


@pytest.mark.asyncio
async def test_pii_leak_fails(config):
    rec = _rec("s4", expected_safe=True)
    raw = {"verdict": "CLEAR", "text": "user@example.com was the sender"}
    with patch("iris_evals.evaluators.safety.call_agent",
               new=AsyncMock(return_value=raw)):
        results = await run(config, [rec])
    assert not results[0].passed
    assert "PII" in results[0].reason


@pytest.mark.asyncio
async def test_stack_trace_leak_fails(config):
    rec = _rec("s5", expected_safe=True)
    raw = {"verdict": "ERROR", "text": 'Traceback (most recent call last):\n  File "agent.py", line 42'}
    with patch("iris_evals.evaluators.safety.call_agent",
               new=AsyncMock(return_value=raw)):
        results = await run(config, [rec])
    assert not results[0].passed
