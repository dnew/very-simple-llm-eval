"""Tests for the Evaluator class."""

import pytest

from eval_framework import Evaluator


def test_evaluator_initialization():
    """Test that the evaluator initializes correctly."""
    evaluator = Evaluator()
    assert evaluator.api_key is None


def test_evaluator_with_api_key():
    """Test that the evaluator initializes with an API key."""
    api_key = "test-key"
    evaluator = Evaluator(api_key=api_key)
    assert evaluator.api_key == api_key


def test_evaluate_basic():
    """Test basic evaluation functionality."""
    evaluator = Evaluator()
    test_cases = ["What is 2+2?", "Explain quantum computing"]
    metrics = ["accuracy", "response_time"]
    
    results = evaluator.evaluate(
        model="gpt-4",
        test_cases=test_cases,
        metrics=metrics,
    )
    
    assert len(results) == len(test_cases)
    for result in results:
        assert result.model == "gpt-4"
        assert result.test_case in test_cases
        assert set(result.metrics.keys()) == set(metrics) 