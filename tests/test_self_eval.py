"""Tests for run_self_eval() — no agent-card.json file needed."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from iris_evals.self_eval import _load_agent_card, _parse_x_evals, run_self_eval
from iris_evals.schemas import EvalType


SAMPLE_CARD = {
    "name": "iris-ap-p111-duplicate-invoice-detection",
    "description": "Detects duplicate invoices",
    "x_evals": {
        "evaluators": ["deterministic", "safety"],
        "dataset": "builtin",
        "threshold_profile": "deterministic_strict",
    },
}


def test_parse_x_evals_builtin(tmp_path):
    config = _parse_x_evals(SAMPLE_CARD, agent_url="http://localhost:8080")
    assert config.agent_name == "iris-ap-p111-duplicate-invoice-detection"
    assert EvalType.DETERMINISTIC in config.evaluators
    assert EvalType.SAFETY in config.evaluators
    records = config.golden_records
    assert len(records) > 0


def test_parse_x_evals_thresholds():
    config = _parse_x_evals(SAMPLE_CARD, agent_url="http://localhost:8080")
    assert config.thresholds.get("deterministic") == 1.0
    assert config.thresholds.get("safety") == 1.0


def test_parse_x_evals_custom_threshold_override():
    card = {**SAMPLE_CARD, "x_evals": {**SAMPLE_CARD["x_evals"], "thresholds": {"deterministic": 0.8}}}
    config = _parse_x_evals(card, agent_url="http://localhost:8080")
    assert config.thresholds["deterministic"] == 0.8


def test_load_agent_card_explicit(tmp_path):
    card_file = tmp_path / "agent-card.json"
    card_file.write_text(json.dumps(SAMPLE_CARD))
    loaded = _load_agent_card(card_file)
    assert loaded["name"] == SAMPLE_CARD["name"]


def test_load_agent_card_missing_raises():
    with pytest.raises(FileNotFoundError):
        _load_agent_card("/nonexistent/path/agent-card.json")


@pytest.mark.asyncio
async def test_run_self_eval_pass(tmp_path):
    card_file = tmp_path / "agent-card.json"
    card_file.write_text(json.dumps(SAMPLE_CARD))

    # All deterministic calls return DUPLICATE or CLEAR as expected
    def mock_response(config, input_data):
        text = str(input_data)
        if "p111-det-001" in str(id(input_data)):
            return {"verdict": "DUPLICATE"}
        return {"verdict": "CLEAR"}

    with patch(
        "iris_evals.evaluators.deterministic.call_agent",
        new=AsyncMock(return_value={"verdict": "DUPLICATE"}),
    ), patch(
        "iris_evals.evaluators.safety.call_agent",
        new=AsyncMock(return_value={"verdict": "CLEAR"}),
    ):
        report = await run_self_eval(
            agent_url="http://localhost:8080",
            card_path=card_file,
        )

    assert report.agent_name == "iris-ap-p111-duplicate-invoice-detection"
    assert report.total_cases > 0
    assert "run_id" in report.summary


@pytest.mark.asyncio
async def test_run_self_eval_calls_on_fail(tmp_path):
    card_file = tmp_path / "agent-card.json"
    card_file.write_text(json.dumps(SAMPLE_CARD))

    on_fail_called = []

    with patch(
        "iris_evals.evaluators.deterministic.call_agent",
        new=AsyncMock(return_value={"verdict": "WRONG_VERDICT"}),
    ), patch(
        "iris_evals.evaluators.safety.call_agent",
        new=AsyncMock(return_value={"verdict": "CLEAR"}),
    ):
        report = await run_self_eval(
            agent_url="http://localhost:8080",
            card_path=card_file,
            on_fail=lambda r: on_fail_called.append(r),
        )

    assert len(on_fail_called) == 1
    assert on_fail_called[0].agent_name == "iris-ap-p111-duplicate-invoice-detection"
