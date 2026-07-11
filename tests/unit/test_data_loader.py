# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit and property tests for ``llm_uncertainty_monitoring.data_loader``.

Covers the split invariants (pairwise-disjoint indices, size conservation,
calibration-fraction respect), the difficult-question selection rule, and
a handful of input-validation boundary cases.

We do *not* hit the network here: ``load_mmlu`` is exercised by the smoke
script (Task 16) against the real ``cais/mmlu`` dataset. These tests
build small synthetic ``Dataset`` objects in memory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import datasets
import pytest
from datasets import Dataset
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_uncertainty_monitoring.data_loader import (
    SplitResult,
    extract_difficult,
    persist_splits_to_s3,
    split_remaining,
)

# Silence the noisy per-row "Filter" progress bars when ``Dataset.filter``
# runs inside Hypothesis property loops.
datasets.disable_progress_bar()

PROPERTY_SETTINGS = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n_per_subject: int, subjects: list[str]) -> Dataset:
    """Build a small synthetic MMLU-shaped Dataset with K=4 choices."""
    rows = []
    counter = 0
    for subject in subjects:
        for _ in range(n_per_subject):
            rows.append(
                {
                    "index": counter,
                    "subject": subject,
                    "question": f"Q{counter}",
                    "choices": ["A", "B", "C", "D"],
                    "answer": counter % 4,
                }
            )
            counter += 1
    return Dataset.from_list(rows)


def _probs_with_max_per_index(
    dataset: Dataset, max_prob_by_index: dict[int, float]
) -> dict[int, list[float]]:
    """Build a ``probs_by_index`` dict where each row has the requested max
    probability and the remainder split evenly across the other classes."""
    out: dict[int, list[float]] = {}
    for idx in dataset["index"]:
        m = max_prob_by_index[idx]
        rest = (1.0 - m) / 3.0
        out[idx] = [m, rest, rest, rest]
    return out


# ---------------------------------------------------------------------------
# Property: split invariants
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    n_per_subject=st.integers(min_value=4, max_value=30),
    n_subjects=st.integers(min_value=1, max_value=4),
    n_difficult_per_subject=st.integers(min_value=0, max_value=3),
    calibration_fraction=st.floats(min_value=0.1, max_value=0.9),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_splits_partition_input_indices_disjointly(
    n_per_subject: int,
    n_subjects: int,
    n_difficult_per_subject: int,
    calibration_fraction: float,
    seed: int,
) -> None:
    """The three returned splits have pairwise-disjoint ``index`` sets,
    their union equals the input ``index`` set, and the calibration size
    equals ``round(calibration_fraction * (|input| - |difficult|))``."""
    subjects = [f"s{i}" for i in range(n_subjects)]
    dataset = _make_dataset(n_per_subject, subjects)
    # Distinct max-probs per row so difficulty ranking is well-defined.
    probs_by_index = _probs_with_max_per_index(
        dataset, {idx: 0.30 + (idx * 0.001) for idx in dataset["index"]}
    )

    difficult = extract_difficult(
        dataset, probs_by_index, n_per_group=n_difficult_per_subject, group_column="subject"
    )
    calibration, test = split_remaining(
        dataset, difficult, calibration_fraction=calibration_fraction, seed=seed
    )

    diff_idx = set(difficult["index"])
    cal_idx = set(calibration["index"])
    test_idx = set(test["index"])

    # Pairwise-disjoint, union equals the input.
    assert diff_idx.isdisjoint(cal_idx)
    assert diff_idx.isdisjoint(test_idx)
    assert cal_idx.isdisjoint(test_idx)
    assert diff_idx | cal_idx | test_idx == set(dataset["index"])

    # Calibration size matches the requested fraction within ±1.
    expected_cal = round(calibration_fraction * (len(dataset) - len(diff_idx)))
    assert abs(len(cal_idx) - expected_cal) <= 1


# ---------------------------------------------------------------------------
# Example tests
# ---------------------------------------------------------------------------


def test_extract_difficult_picks_lowest_max_prob_per_group() -> None:
    """``extract_difficult`` picks, per group, the rows with the smallest
    max(probs) — i.e. the rows where the model is least confident."""
    dataset = _make_dataset(n_per_subject=5, subjects=["s0", "s1"])
    # Per subject, the lowest max-prob is the smallest index.
    probs_by_index = _probs_with_max_per_index(
        dataset, {idx: 0.30 + 0.05 * (idx % 5) for idx in dataset["index"]}
    )

    difficult = extract_difficult(dataset, probs_by_index, n_per_group=2, group_column="subject")
    assert sorted(difficult["index"]) == [0, 1, 5, 6]


# ---------------------------------------------------------------------------
# Boundary error cases
# ---------------------------------------------------------------------------


def test_extract_difficult_rejects_missing_probs_entry() -> None:
    dataset = _make_dataset(2, ["s"])
    probs = {0: [0.5, 0.5, 0.0, 0.0]}  # missing index 1
    with pytest.raises(ValueError, match="missing entries"):
        extract_difficult(dataset, probs, n_per_group=1, group_column="subject")


@pytest.mark.parametrize("fraction", [-0.1, 0.05, 0.95])
def test_split_remaining_rejects_calibration_fraction_outside_range(
    fraction: float,
) -> None:
    dataset = _make_dataset(10, ["s"])
    difficult = dataset.select(range(0))  # empty
    with pytest.raises(ValueError, match="calibration_fraction"):
        split_remaining(dataset, difficult, calibration_fraction=fraction)


# ---------------------------------------------------------------------------
# persist_splits_to_s3
# ---------------------------------------------------------------------------


def test_persist_splits_to_s3_writes_three_jsonl_files() -> None:
    """Three files are uploaded to the right S3 keys with JSONL bodies.
    Uses a ``MagicMock`` S3 client so no AWS call is made."""
    dataset = _make_dataset(2, ["s"])
    splits = SplitResult(
        difficult=dataset.select([0]),
        calibration=dataset.select([1]),
        test=dataset.select([]),
    )

    s3_client = MagicMock()
    uris = persist_splits_to_s3(splits, "s3://DOC-EXAMPLE-BUCKET/mmlu/", s3_client=s3_client)

    assert uris == {
        "difficult": "s3://DOC-EXAMPLE-BUCKET/mmlu/difficult.jsonl",
        "calibration": "s3://DOC-EXAMPLE-BUCKET/mmlu/cal.jsonl",
        "test": "s3://DOC-EXAMPLE-BUCKET/mmlu/test.jsonl",
    }
    assert s3_client.put_object.call_count == 3


def test_persist_splits_to_s3_rejects_non_s3_uri() -> None:
    splits = SplitResult(
        difficult=Dataset.from_list([]),
        calibration=Dataset.from_list([]),
        test=Dataset.from_list([]),
    )
    with pytest.raises(ValueError, match="s3://"):
        persist_splits_to_s3(splits, "https://bucket/key/", s3_client=MagicMock())
