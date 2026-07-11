# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit and property tests for ``llm_uncertainty_monitoring.conformal``.

Covers the LAC quantile formula, the prediction-set construction rules
(non-emptiness, membership, monotonicity in the threshold), and a handful
of input-validation boundary cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from llm_uncertainty_monitoring.conformal import (
    LACScore,
    build_prediction_set,
    compute_threshold,
)

# Common Hypothesis settings: 100 generated cases per property, no per-example
# deadline (Dirichlet sampling is borderline by default), and silence the
# "too slow" health check on small CI runners.
PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def calibration_scores_and_coverage(draw: st.DrawFn) -> tuple[list[float], float]:
    """Sample a non-empty list of LAC-style scores in ``[0, 1]`` and a
    coverage such that ``compute_threshold`` succeeds (rank <= N)."""
    n = draw(st.integers(min_value=1, max_value=500))
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    coverage = draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    alpha = 1.0 - coverage
    assume(math.ceil((n + 1) * (1.0 - alpha)) <= n)
    return scores, coverage


@st.composite
def probability_simplex(draw: st.DrawFn) -> np.ndarray:
    """Sample a Dirichlet-distributed probability vector of random length
    ``K`` in ``[2, 8]``."""
    k = draw(st.integers(min_value=2, max_value=8))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k))
    return p / p.sum()


# ---------------------------------------------------------------------------
# LAC quantile (compute_threshold)
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(scores_and_coverage=calibration_scores_and_coverage())
def test_compute_threshold_matches_formula(
    scores_and_coverage: tuple[list[float], float],
) -> None:
    """``compute_threshold`` returns the empirical
    ``ceil((N + 1) * (1 - alpha))``-th smallest score, lies in ``[0, 1]``,
    and at least that many scores are ``<= q_hat``."""
    scores, coverage = scores_and_coverage
    n = len(scores)
    alpha = 1.0 - coverage
    rank = math.ceil((n + 1) * (1.0 - alpha))

    q_hat = compute_threshold(scores, coverage)

    assert q_hat == pytest.approx(sorted(scores)[rank - 1])
    assert 0.0 <= q_hat <= 1.0
    assert sum(1 for s in scores if s <= q_hat) >= rank


# ---------------------------------------------------------------------------
# Prediction set (build_prediction_set)
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    p=probability_simplex(),
    q_hat=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_prediction_set_is_nonempty_and_matches_threshold_rule(p: np.ndarray, q_hat: float) -> None:
    """The prediction set is non-empty, sorted, and either consists of the
    indices ``k`` with ``p[k] >= 1 - q_hat`` or falls back to
    ``[argmax(p)]`` when no index meets the threshold."""
    pred_set = build_prediction_set(p.tolist(), q_hat)
    assert pred_set, "prediction set must be non-empty"
    assert pred_set == sorted(pred_set), "prediction set must be sorted"

    threshold = 1.0 - q_hat
    expected = [int(k) for k in range(p.size) if p[k] >= threshold]
    if expected:
        assert pred_set == sorted(expected)
    else:
        assert pred_set == [int(np.argmax(p))]


@PROPERTY_SETTINGS
@given(
    p=probability_simplex(),
    q1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    q2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_prediction_set_grows_monotonically_with_q_hat(p: np.ndarray, q1: float, q2: float) -> None:
    """Increasing the threshold can only add indices: for ``q1 <= q2``,
    ``build_prediction_set(p, q1)`` is a subset of
    ``build_prediction_set(p, q2)`` whenever the inequality-defined set at
    ``q1`` is non-empty (no fallback). When the set at ``q1`` is empty,
    the ``argmax(p)`` fallback is still in the set at ``q2``."""
    if q1 > q2:
        q1, q2 = q2, q1

    s1 = set(build_prediction_set(p.tolist(), q1))
    s2 = set(build_prediction_set(p.tolist(), q2))

    inequality_set_at_q1 = {int(k) for k in range(p.size) if p[k] >= 1.0 - q1}
    if inequality_set_at_q1:
        assert s1 <= s2
    else:
        assert int(np.argmax(p)) in s2


def test_prediction_set_falls_back_to_argmax_when_threshold_too_strict() -> None:
    """When no probability satisfies ``p[k] >= 1 - q_hat``, the prediction
    set falls back to ``[argmax(p)]`` so it is never empty."""
    p = [0.4, 0.45, 0.15]  # 1 - q_hat = 0.99 > max(p), argmax = 1
    assert build_prediction_set(p, q_hat=0.01) == [1]


# ---------------------------------------------------------------------------
# Boundary error cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [-0.1, 0.0, 1.0, 1.5, float("nan")])
def test_compute_threshold_rejects_coverage_outside_open_unit_interval(
    coverage: float,
) -> None:
    with pytest.raises(ValueError, match="target_coverage"):
        compute_threshold([0.1, 0.2, 0.3], coverage)


def test_compute_threshold_rejects_calibration_set_too_small() -> None:
    """target_coverage = 0.95, alpha = 0.05, N = 5  ->  rank = 6 > N."""
    with pytest.raises(ValueError) as exc:
        compute_threshold([0.1, 0.2, 0.3, 0.4, 0.5], target_coverage=0.95)
    msg = str(exc.value)
    assert "N=5" in msg
    assert "alpha=" in msg
    # Minimum N required for alpha = 0.05 is ceil(0.95 / 0.05) = 19.
    assert "required N>=19" in msg


def test_build_prediction_set_rejects_probability_vector_not_summing_to_one() -> None:
    bad_p = [0.4, 0.4, 0.4]  # sums to 1.2
    with pytest.raises(ValueError, match=r"sum=1\.2"):
        build_prediction_set(bad_p, q_hat=0.5)


@pytest.mark.parametrize("q_hat", [-0.1, 1.1, float("nan")])
def test_build_prediction_set_rejects_q_hat_outside_unit_interval(
    q_hat: float,
) -> None:
    with pytest.raises(ValueError, match="q_hat"):
        build_prediction_set([0.5, 0.3, 0.2], q_hat=q_hat)


def test_lac_score_rejects_y_true_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"y_true must be in \[0, 3\)"):
        LACScore().score([0.5, 0.3, 0.2], y_true=3)
