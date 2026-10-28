"""Tests for the probabilistic evaluator."""

import pytest
from unittest.mock import AsyncMock, patch

from iris_evals.evaluators.probabilistic import run, _compute_metrics
from iris_evals.schemas import EvalConfig, EvalType, GoldenRecord


@pytest.fixture
def config():
    return EvalConfig(agent_name="test-triage", evaluators=[EvalType.PROBABILISTIC])


@pytest.fixture
def records():
    return [
        GoldenRecord(id="c1", input="Invoice attached", expected_label="Invoice"),
        GoldenRecord(id="c2", input="Payment query",    expected_label="VendorQuery"),
        GoldenRecord(id="c3", input="Statement SOA",    expected_label="Statement"),
        GoldenRecord(id="c4", input="Invoice attached", expected_label="Invoice"),
    ]


@pytest.mark.asyncio
async def test_perfect_accuracy(config, records):
    responses = [
        {"document_type": "Invoice"},
        {"document_type": "VendorQuery"},
        {"document_type": "Statement"},
        {"document_type": "Invoice"},
    ]
    with patch("iris_evals.evaluators.probabilistic.call_agent",
               new=AsyncMock(side_effect=responses)):
        results, metrics = await run(config, records)

    assert metrics["accuracy"] == 1.0
    assert all(r.passed for r in results)


@pytest.mark.asyncio
async def test_partial_accuracy(config, records):
    responses = [
        {"document_type": "Invoice"},
        {"document_type": "Invoice"},   # wrong — expected VendorQuery
        {"document_type": "Statement"},
        {"document_type": "Invoice"},
    ]
    with patch("iris_evals.evaluators.probabilistic.call_agent",
               new=AsyncMock(side_effect=responses)):
        results, metrics = await run(config, records)

    assert metrics["accuracy"] == 0.75
    assert not results[1].passed


def test_compute_metrics_perfect():
    from collections import defaultdict
    from iris_evals.schemas import EvalResult, EvalType as ET

    confusion = defaultdict(lambda: defaultdict(int))
    confusion["A"]["A"] = 3
    confusion["B"]["B"] = 2
    results = [
        EvalResult(record_id=f"r{i}", eval_type=ET.PROBABILISTIC, passed=True)
        for i in range(5)
    ]
    metrics = _compute_metrics(confusion, results)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
