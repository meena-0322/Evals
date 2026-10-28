"""Tests for the deterministic evaluator."""

import pytest
from unittest.mock import AsyncMock, patch

from iris_evals.evaluators.deterministic import run, _matches, _extract_verdict_or_text
from iris_evals.schemas import EvalConfig, EvalType, GoldenRecord


@pytest.fixture
def config():
    return EvalConfig(agent_name="test-agent", evaluators=[EvalType.DETERMINISTIC])


@pytest.fixture
def records():
    return [
        GoldenRecord(id="r1", input="test input", expected_verdict="DUPLICATE"),
        GoldenRecord(id="r2", input="test input 2", expected_verdict="CLEAR"),
    ]


@pytest.mark.asyncio
async def test_pass_when_verdict_matches(config, records):
    responses = [
        {"verdict": "DUPLICATE"},
        {"verdict": "CLEAR"},
    ]
    with patch("iris_evals.evaluators.deterministic.call_agent", new=AsyncMock(side_effect=responses)):
        results = await run(config, records)

    assert all(r.passed for r in results)
    assert results[0].record_id == "r1"


@pytest.mark.asyncio
async def test_fail_when_verdict_mismatches(config, records):
    responses = [
        {"verdict": "CLEAR"},      # wrong — expected DUPLICATE
        {"verdict": "DUPLICATE"},  # wrong — expected CLEAR
    ]
    with patch("iris_evals.evaluators.deterministic.call_agent", new=AsyncMock(side_effect=responses)):
        results = await run(config, records)

    assert not results[0].passed
    assert not results[1].passed
    assert "Expected 'DUPLICATE'" in results[0].reason


@pytest.mark.asyncio
async def test_fail_on_agent_exception(config, records):
    with patch("iris_evals.evaluators.deterministic.call_agent",
               new=AsyncMock(side_effect=Exception("connection refused"))):
        results = await run(config, records)

    assert all(not r.passed for r in results)
    assert "Agent call failed" in results[0].reason


def test_matches_case_insensitive():
    assert _matches("duplicate", "DUPLICATE")
    assert _matches("CLEAR", "clear")
    assert not _matches("CLEAR", "DUPLICATE")


def test_extract_verdict_from_key():
    assert _extract_verdict_or_text({"verdict": "DUPLICATE"}) == "DUPLICATE"
    assert _extract_verdict_or_text({"invoice_resolution": "RESOLVED"}) == "RESOLVED"


def test_extract_verdict_from_a2a_artifact():
    raw = {"result": {"artifacts": [{"parts": [{"text": '{"verdict": "CLEAR"}'}]}]}}
    # _extract_verdict_or_text returns the full artifact text, caller parses it
    out = _extract_verdict_or_text(raw)
    assert "CLEAR" in out
