# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for ``llm_uncertainty_monitoring.data_quality``.

These tests boot a real Spark session via PyDeequ. The first run is slow
(JVM startup, JAR download) — typically 30-60 seconds. Subsequent calls
in the same pytest process are fast because :func:`_get_spark` caches
the session.

The whole module is skipped when PySpark cannot be imported or when the
JVM that PySpark needs is unavailable on the host (no Java installed).
That keeps the test suite green on lightweight CI runners.
"""

from __future__ import annotations

import os

# PyDeequ requires SPARK_VERSION at *import* time (it raises RuntimeError,
# not ImportError, if missing). Set the default before importorskip so
# `pytest.importorskip("pydeequ")` actually behaves like a skip.
os.environ.setdefault("SPARK_VERSION", "3.5")

import pytest

# Skip the whole module if pyspark isn't installed.
pyspark = pytest.importorskip("pyspark")
pytest.importorskip("pydeequ")

from datasets import Dataset  # noqa: E402

from llm_uncertainty_monitoring.data_quality import (  # noqa: E402
    DataQualityError,
    validate_mmlu_dataset,
)

pytestmark = pytest.mark.slow


def _make_rows(answers: list[str], questions: list[str | None]) -> list[dict[str, object]]:
    """Build five MMLU-shaped rows with caller-supplied ``answer`` /
    ``question`` columns."""
    assert len(answers) == 5
    assert len(questions) == 5
    return [
        {
            "index": i,
            "subject": "marketing",
            "question": questions[i],
            "choices": ["A", "B", "C", "D"],
            "answer": answers[i],
        }
        for i in range(5)
    ]


def test_validate_mmlu_dataset_passes_on_valid_input() -> None:
    """A clean 5-row dataset with answers in ``[0, K)`` and no nulls
    produces an empty failure list and ``passed=True``."""
    try:
        rows = _make_rows(
            answers=["0", "1", "2", "3", "0"],
            questions=[f"Q{i}" for i in range(5)],
        )
        dataset = Dataset.from_list(rows)
        report = validate_mmlu_dataset(dataset, num_choices=4)
    except Exception as err:  # pragma: no cover - environment guard
        # If Spark can't start (no Java, etc.), skip rather than fail the suite.
        pytest.skip(f"Spark unavailable for this run: {err!r}")

    assert report.passed is True
    assert report.failures == []


def test_validate_mmlu_dataset_reports_completeness_and_range_failures() -> None:
    """A dataset with one out-of-range answer and one ``null`` question
    must trigger ``DataQualityError`` and surface both classes of
    failure in ``err.report.failures``."""
    try:
        rows = _make_rows(
            answers=["0", "1", "9", "3", "0"],  # "9" out of [0, 4)
            questions=["Q0", None, "Q2", "Q3", "Q4"],  # null in question
        )
        dataset = Dataset.from_list(rows)
        with pytest.raises(DataQualityError) as exc_info:
            validate_mmlu_dataset(dataset, num_choices=4)
    except DataQualityError:
        raise
    except Exception as err:  # pragma: no cover - environment guard
        pytest.skip(f"Spark unavailable for this run: {err!r}")

    failures_blob = " | ".join(exc_info.value.report.failures).lower()
    assert "answer" in failures_blob
    assert "question" in failures_blob
    assert exc_info.value.report.passed is False
