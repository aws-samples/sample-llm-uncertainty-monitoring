# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for the calibration aggregator's compute_calibration logic.

Validates that:
- q_hat returned by compute_calibration equals compute_threshold applied
  to the same LAC scores computed directly.
- The per-sample diagnostics contain the expected fields.
- Edge cases (single sample, all-same logprobs) are handled correctly.

Uses Hypothesis for property-based testing with random logprobs and
target_coverage values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.special import softmax

from llm_uncertainty_monitoring.calibration.aggregator import compute_calibration
from llm_uncertainty_monitoring.conformal import compute_threshold

# ---------------------------------------------------------------------------
# Helpers: build synthetic Batch Transform output lines
# ---------------------------------------------------------------------------


def _build_lines(
    logprobs_by_index: dict[int, list[float]],
    answers: dict[int, int],
) -> list[dict]:
    """Build synthetic Batch Transform output lines.

    Each line simulates the joined output format: input metadata +
    SageMakerOutput with details.prefill containing the logprob.
    """
    lines = []
    for index, logprobs in logprobs_by_index.items():
        answer = answers[index]
        for k, logprob in enumerate(logprobs):
            lines.append(
                {
                    "index": index,
                    "k": k,
                    "answer": answer,
                    "inputs": f"prompt for index {index} token {k}",
                    "SageMakerOutput": {
                        "details": {
                            "prefill": [
                                {"logprob": -999.0},  # earlier tokens (irrelevant)
                                {"logprob": logprob},  # last token = the one we care about
                            ]
                        }
                    },
                }
            )
    return lines


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# K in [2, 6], N in [3, 50]
k_strategy = st.integers(min_value=2, max_value=6)
n_strategy = st.integers(min_value=3, max_value=20)
target_coverage_strategy = st.floats(min_value=0.5, max_value=0.99)


@st.composite
def calibration_inputs(draw: st.DrawFn):
    """Generate random calibration inputs: K, N, logprobs, answers, target_coverage."""
    k = draw(k_strategy)
    n = draw(n_strategy)
    target_coverage = draw(target_coverage_strategy)

    # Ensure N is large enough for the target_coverage
    alpha = 1.0 - target_coverage
    min_n = math.ceil((1.0 - alpha) / alpha) if alpha > 0 else 1
    if n < min_n:
        n = min_n

    logprobs_by_index: dict[int, list[float]] = {}
    answers: dict[int, int] = {}
    for i in range(n):
        # Random logprobs (negative values, typical for LLM outputs)
        logprobs = draw(
            st.lists(
                st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False),
                min_size=k,
                max_size=k,
            )
        )
        logprobs_by_index[i] = logprobs
        answers[i] = draw(st.integers(min_value=0, max_value=k - 1))

    return k, n, logprobs_by_index, answers, target_coverage


# ---------------------------------------------------------------------------
# Property test: q_hat matches direct computation
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(inputs=calibration_inputs())
def test_compute_calibration_matches_direct_threshold(inputs):
    """q_hat from compute_calibration equals compute_threshold on directly-computed LAC scores."""
    k, n, logprobs_by_index, answers, target_coverage = inputs

    lines = _build_lines(logprobs_by_index, answers)
    q_hat, diagnostics = compute_calibration(lines, target_coverage)

    # Compute LAC scores directly for comparison
    direct_scores = []
    for i in range(n):
        logprobs = np.array(logprobs_by_index[i])
        probs = softmax(logprobs)
        lac_score = 1.0 - probs[answers[i]]
        direct_scores.append(lac_score)

    expected_q_hat = compute_threshold(direct_scores, target_coverage)

    assert (
        abs(q_hat - expected_q_hat) < 1e-10
    ), f"q_hat mismatch: got {q_hat}, expected {expected_q_hat}"


# ---------------------------------------------------------------------------
# Property test: diagnostics structure
# ---------------------------------------------------------------------------


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(inputs=calibration_inputs())
def test_compute_calibration_diagnostics_structure(inputs):
    """Each diagnostic entry has the expected fields and valid values."""
    k, n, logprobs_by_index, answers, target_coverage = inputs

    lines = _build_lines(logprobs_by_index, answers)
    _, diagnostics = compute_calibration(lines, target_coverage)

    assert len(diagnostics) == n

    for diag in diagnostics:
        assert "index" in diag
        assert "lac_score" in diag
        assert "probs" in diag
        assert "answer" in diag

        # LAC score is in [0, 1]
        assert 0.0 <= diag["lac_score"] <= 1.0

        # Probs sum to 1
        assert abs(sum(diag["probs"]) - 1.0) < 1e-6

        # Probs length equals K
        assert len(diag["probs"]) == k

        # Answer is valid index
        assert 0 <= diag["answer"] < k


# ---------------------------------------------------------------------------
# Example-based tests
# ---------------------------------------------------------------------------


def test_compute_calibration_simple_case():
    """Smoke test with a small known input."""
    # 3 samples, K=2, answer always 0
    logprobs_by_index = {
        0: [-0.5, -2.0],  # softmax → [0.82, 0.18], LAC = 1 - 0.82 = 0.18
        1: [-1.0, -1.0],  # softmax → [0.50, 0.50], LAC = 1 - 0.50 = 0.50
        2: [-2.0, -0.5],  # softmax → [0.18, 0.82], LAC = 1 - 0.18 = 0.82
    }
    answers = {0: 0, 1: 0, 2: 0}
    lines = _build_lines(logprobs_by_index, answers)

    q_hat, diagnostics = compute_calibration(lines, target_coverage=0.5)

    assert len(diagnostics) == 3
    # Scores sorted: [0.18, 0.50, 0.82]
    # For coverage=0.5, alpha=0.5, rank=ceil(4*0.5)=2 → sorted[1] = 0.50
    assert abs(q_hat - 0.5) < 0.01


def test_compute_calibration_insufficient_samples():
    """Raises ValueError when N is too small for the target_coverage."""
    # 1 sample, coverage=0.99 → needs N >= 99
    logprobs_by_index = {0: [-1.0, -1.0]}
    answers = {0: 0}
    lines = _build_lines(logprobs_by_index, answers)

    with pytest.raises(ValueError, match="calibration set too small"):
        compute_calibration(lines, target_coverage=0.99)


def test_compute_calibration_missing_prefill():
    """Raises ValueError when a record is missing prefill logprobs."""
    lines = [
        {
            "index": 0,
            "k": 0,
            "answer": 0,
            "SageMakerOutput": {"details": {"prefill": []}},
        }
    ]

    with pytest.raises(ValueError, match="missing details.prefill"):
        compute_calibration(lines, target_coverage=0.9)
