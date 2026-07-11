# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Pure conformal-prediction core (no AWS calls).

Implements split-conformal prediction with a pluggable nonconformity score.
The default :class:`LACScore` reproduces the Least Ambiguous set-valued
Classifier of Sadinle et al., *Journal of the American Statistical
Association*, 2019.

Public API
----------
- :class:`NonconformityScore` -- protocol for plug-in scores.
- :class:`LACScore` -- LAC implementation (default for the POC).
- :func:`compute_threshold` -- score-agnostic split-conformal quantile.
- :func:`build_prediction_set`, :func:`compute_lac_quantile` -- thin
  notebook-friendly aliases over the LAC implementation, kept so the demo
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

# from llm_uncertainty_monitoring.conformal import *, only those names are imported
__all__ = [
    "ProbabilityVector",
    "SUM_TOLERANCE",
    "NonconformityScore",
    "LACScore",
    "compute_threshold",
    "build_prediction_set",
    "compute_lac_quantile",
]


ProbabilityVector = Sequence[float]
"""A vector ``p`` of length ``K`` whose entries lie in ``[0, 1]`` and sum
to ``1`` within :data:`SUM_TOLERANCE`."""

SUM_TOLERANCE: float = 1e-6
"""Numerical slack accepted when validating that a probability vector sums
to one. Calibrated against the softmax precision of the prefill scoring
pipeline (Req 5.2)."""


def _validate_probability_vector(p: ProbabilityVector) -> np.ndarray[Any, Any]:
    """Validate ``p`` and return it as a 1-D ``np.ndarray``.

    Raises ``ValueError`` (with the offending sum or shape) if ``p`` is
    not a finite vector of length >= 2 summing to ``1`` within
    :data:`SUM_TOLERANCE`. (Req 5.2)
    """
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"probability vector must be 1-D, got shape {arr.shape!r}")
    if arr.size < 2:
        raise ValueError(f"probability vector must have length >= 2, got length {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("probability vector contains non-finite entries")
    total = float(arr.sum())
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(
            f"probability vector must sum to 1 within {SUM_TOLERANCE}, " f"got sum={total!r}"
        )
    return arr


def _validate_q_hat(q_hat: float) -> float:
    """Coerce and validate ``q_hat in [0, 1]`` (Req 5.3)."""
    q = float(q_hat)
    if math.isnan(q) or q < 0.0 or q > 1.0:
        raise ValueError(f"q_hat must be in [0, 1], got q_hat={q_hat!r}")
    return q


@runtime_checkable
class NonconformityScore(Protocol):
    """Pluggable conformal nonconformity score.

    Implementations expose two pure operations:

    - :meth:`score(p, y_true)` produces a calibration-time nonconformity
      score (larger = more nonconforming).
    - :meth:`build_prediction_set(p, q_hat)` produces the inference-time
      prediction set given the calibrated threshold.

    Adding a new score (e.g. APS, RAPS) requires implementing this
    interface only; calibration and inference code consume the protocol
    and stay unchanged.
    """

    name: str

    def score(self, p: ProbabilityVector, y_true: int) -> float:
        """Calibration-time score for ``(p, y_true)``."""
        ...

    def build_prediction_set(self, p: ProbabilityVector, q_hat: float) -> list[int]:
        """Inference-time prediction set for threshold ``q_hat``."""
        ...


class LACScore:
    """Least Ambiguous set-valued Classifier (Sadinle et al., 2019).

    - score:                   ``s_i = 1 - p_i[y_i]``
    - prediction set:          ``{ k in [0, K) : p[k] >= 1 - q_hat }``
    - empty-set fallback:      ``{argmax(p)}`` (Req 5.4)
    """

    name: str = "LAC"

    def score(self, p: ProbabilityVector, y_true: int) -> float:
        """Return ``1 - p[y_true]``.

        Raises:
            ValueError: if ``p`` is not on the simplex within tolerance,
                or if ``y_true`` is outside ``[0, len(p))``.
        """
        arr = _validate_probability_vector(p)
        k = arr.shape[0]
        if not (0 <= int(y_true) < k):
            raise ValueError(f"y_true must be in [0, {k}), got y_true={y_true!r}")
        return float(1.0 - arr[int(y_true)])

    def build_prediction_set(self, p: ProbabilityVector, q_hat: float) -> list[int]:
        """Return the LAC prediction set for ``q_hat``.

        ``{ k in [0, K) : p[k] >= 1 - q_hat }``. If empty, falls back to
        ``[argmax(p)]`` (Req 5.4). Returns a sorted ``list[int]`` so the
        output is deterministic for serialization.
        """
        arr = _validate_probability_vector(p)
        q = _validate_q_hat(q_hat)
        threshold = 1.0 - q
        included = [int(k) for k, prob in enumerate(arr) if prob >= threshold]
        if not included:
            included = [int(np.argmax(arr))]
        return sorted(included)


def compute_threshold(
    calibration_scores: Sequence[float],
    target_coverage: float,
) -> float:
    """Split-conformal quantile (score-agnostic).

    For a list of ``N`` calibration scores and a target coverage
    ``1 - alpha``, return the empirical quantile

    .. math::

       \\hat{q} = s_{\\lceil (N+1)(1-\\alpha) \\rceil}

    of the sorted scores (1-indexed) where ``alpha = 1 - target_coverage``
    (Req 4.3).

    Raises:
        ValueError: if ``target_coverage`` is not strictly between 0 and 1
            (Req 4.6); or if ``N`` is too small for the requested
            coverage, in which case the message reports ``N``, ``alpha``,
            and the minimum required ``N`` (Req 4.7); or if any score is
            non-finite.
    """
    coverage = float(target_coverage)
    if not (0.0 < coverage < 1.0):
        raise ValueError(
            "target_coverage must be in the open interval (0, 1), "
            f"got target_coverage={target_coverage!r}"
        )
    alpha = 1.0 - coverage

    scores = np.asarray(list(calibration_scores), dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("calibration_scores must contain at least one score, got 0")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration_scores contains non-finite entries")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        # Inverting the rank constraint:
        #   ceil((N + 1) * (1 - alpha)) <= N
        #   <=> N >= ceil((1 - alpha) / alpha)
        min_n_required = math.ceil((1.0 - alpha) / alpha) if alpha > 0 else math.inf
        raise ValueError(
            f"calibration set too small for target_coverage={coverage!r}: "
            f"N={n}, alpha={alpha!r}, required N>={min_n_required}"
        )

    sorted_scores = np.sort(scores)
    return float(sorted_scores[rank - 1])


def build_prediction_set(p: ProbabilityVector, q_hat: float) -> list[int]:
    """Notebook-friendly alias for :meth:`LACScore.build_prediction_set`.

    Kept so the demo notebook reads like the original prototype (Req 5.1).
    """
    return LACScore().build_prediction_set(p, q_hat)


def compute_lac_quantile(lac_scores: Sequence[float], target_coverage: float) -> float:
    """Notebook-friendly alias for :func:`compute_threshold`.

    LAC is currently the only score in the POC, so the alias delegates
    directly. Renaming to ``compute_threshold`` keeps the door open for
    future scores.
    """
    return compute_threshold(lac_scores, target_coverage)
