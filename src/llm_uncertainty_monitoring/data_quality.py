# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Data-quality validation for MMLU-shaped datasets, backed by PyDeequ.

This module is a *dev-only* gate: it is invoked from the offline data-prep
scripts and from the calibration script before ``q_hat`` is computed.
The runtime Lambdas do **not** import it. PyDeequ pulls in PySpark which
requires a JVM, so importing this module is intentionally heavy.

Constraints checked against each split:

- non-empty (``hasSize > 0``);
- ``question``, ``choices``, ``answer``, and ``subject`` are complete
  (no nulls);
- the ``answer`` column is restricted to ``[str(0), ..., str(K-1)]``
  where ``K = num_choices`` — the contract is "answers are in 0..K-1
  only";
- ``index`` is unique within the split.

``validate_split_result`` adds one cross-split check: the ``index`` sets of
``difficult``, ``calibration``, and ``test`` are pairwise disjoint.

A typical call looks like::

    from llm_uncertainty_monitoring.data_quality import (
        DataQualityError,
        validate_split_result,
    )

    try:
        reports = validate_split_result(splits)
    except DataQualityError as err:
        print(err.report.failures)
        raise
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# PyDeequ reads SPARK_VERSION at import time and raises RuntimeError
# (not ImportError) when the env var is missing. Set a sensible default
# before any pydeequ symbol is referenced so consumers don't have to
# remember it.
os.environ.setdefault("SPARK_VERSION", "3.5")

import pydeequ
from pydeequ.checks import Check, CheckLevel
from pydeequ.verification import VerificationResult, VerificationSuite
from pyspark.sql import SparkSession

if TYPE_CHECKING:
    from datasets import Dataset

    from llm_uncertainty_monitoring.data_loader import SplitResult

__all__ = [
    "DataQualityReport",
    "DataQualityError",
    "validate_mmlu_dataset",
    "validate_split_result",
]


# ---------------------------------------------------------------------------
# Report / error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataQualityReport:
    """Outcome of running a set of PyDeequ checks against a dataset.

    ``failures`` is a list of human-readable strings, one per failed
    constraint, of the form ``"<constraint name>: <observed value or
    message>"``. The list is empty when every constraint passes.
    """

    passed: bool
    failures: list[str] = field(default_factory=list)


class DataQualityError(RuntimeError):
    """Raised when one or more data-quality checks fail.

    The originating :class:`DataQualityReport` is attached as
    :attr:`report` so callers can inspect the individual failures.
    """

    def __init__(self, report: DataQualityReport) -> None:
        super().__init__(
            f"data-quality validation failed with {len(report.failures)} "
            f"constraint failure(s): {report.failures!r}"
        )
        self.report = report


# ---------------------------------------------------------------------------
# Spark session bootstrap
# ---------------------------------------------------------------------------


_SPARK_SESSION: SparkSession | None = None


def _get_spark() -> SparkSession:
    """Return a process-wide SparkSession with the PyDeequ JARs configured.

    PyDeequ requires the Deequ Maven coordinates and the F2J repository
    to be present on the Spark classpath. We build the session once and
    cache it; subsequent calls reuse it via ``getOrCreate``.
    """
    global _SPARK_SESSION
    if _SPARK_SESSION is not None:
        return _SPARK_SESSION

    spark = (
        SparkSession.builder.appName("llm-uncertainty-monitoring-data-quality")
        .config("spark.jars.packages", pydeequ.deequ_maven_coord)
        .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
        .getOrCreate()
    )
    _SPARK_SESSION = spark
    return spark


# ---------------------------------------------------------------------------
# Single-dataset validation
# ---------------------------------------------------------------------------


def validate_mmlu_dataset(dataset: Dataset, num_choices: int = 4) -> DataQualityReport:
    """Run PyDeequ data-quality checks on a single MMLU-shaped split.

    Args:
        dataset: a HuggingFace ``datasets.Dataset`` carrying the columns
            ``index``, ``question``, ``choices``, ``answer``, and
            ``subject``. Produced by :mod:`data_loader`.
        num_choices: ``K``, the number of answer choices. Answers must
            lie in ``{"0", ..., str(K-1)}``.

    Returns:
        A :class:`DataQualityReport` summarising the outcome. The report
        is also returned on success.

    Raises:
        DataQualityError: when one or more constraints fail. The
            offending constraints are listed in ``err.report.failures``.
        ValueError: if ``num_choices`` is not a positive integer.
    """
    if num_choices < 1:
        raise ValueError(f"num_choices must be a positive integer, got {num_choices!r}")

    spark = _get_spark()

    # ``datasets.Dataset.to_pandas`` returns a fully-realized DataFrame.
    # POC scale (~thousands of rows) — fine to materialize in memory.
    pdf = dataset.to_pandas().copy()

    # Spark cannot reliably infer schema for list-typed columns from
    # pandas (``choices`` is ``list[str]``); we coerce the ``choices``
    # column to a string-encoded form for the completeness check, which
    # only cares about null vs. non-null values. ``answer`` is stored as
    # an integer index in MMLU but the contract checks the string form
    # ``["0", ..., str(K-1)]``.
    if "choices" in pdf.columns:
        pdf["choices"] = pdf["choices"].apply(
            lambda v: None if v is None else "|".join(map(str, v))
        )
    if "answer" in pdf.columns:
        pdf["answer"] = pdf["answer"].apply(lambda v: None if v is None else str(v))

    df = spark.createDataFrame(pdf)

    allowed_answers = [str(i) for i in range(num_choices)]
    check = (
        Check(spark, CheckLevel.Error, "mmlu_split_quality")
        .hasSize(lambda n: n > 0)
        .isComplete("question")
        .isComplete("choices")
        .isComplete("answer")
        .isComplete("subject")
        .isContainedIn("answer", allowed_answers)
        .isUnique("index")
    )

    result = VerificationSuite(spark).onData(df).addCheck(check).run()
    failures = _collect_failures(spark, result)
    report = DataQualityReport(passed=not failures, failures=failures)
    if not report.passed:
        raise DataQualityError(report)
    return report


# ---------------------------------------------------------------------------
# Cross-split validation
# ---------------------------------------------------------------------------


def validate_split_result(
    splits: SplitResult, num_choices: int = 4
) -> dict[str, DataQualityReport]:
    """Validate every split in a :class:`SplitResult` and assert pairwise
    disjointness of their ``index`` sets at the Spark level.

    Args:
        splits: the output of :func:`data_loader.split_remaining` plus
            :func:`data_loader.extract_difficult`, packaged as a
            :class:`SplitResult`.
        num_choices: ``K``, forwarded to :func:`validate_mmlu_dataset`.

    Returns:
        A dict mapping ``"difficult"``, ``"calibration"``, ``"test"`` to
        the per-split :class:`DataQualityReport`.

    Raises:
        DataQualityError: if any per-split check fails, or if any pair of
            splits shares an ``index`` value. The aggregated failures
            from every split are surfaced in ``err.report.failures``.
    """
    spark = _get_spark()

    # Validate each split independently. We catch the per-split
    # ``DataQualityError`` so we can report all failures in one go
    # rather than stopping at the first split.
    reports: dict[str, DataQualityReport] = {}
    aggregated_failures: list[str] = []
    split_datasets = {
        "difficult": splits.difficult,
        "calibration": splits.calibration,
        "test": splits.test,
    }
    for name, ds in split_datasets.items():
        try:
            reports[name] = validate_mmlu_dataset(ds, num_choices=num_choices)
        except DataQualityError as err:
            reports[name] = err.report
            aggregated_failures.extend(f"{name}/{f}" for f in err.report.failures)

    # Cross-split: pairwise disjoint indices.
    pdfs = {name: ds.to_pandas()[["index"]] for name, ds in split_datasets.items()}
    spark_dfs = {name: spark.createDataFrame(pdf) for name, pdf in pdfs.items()}
    pairs = [
        ("difficult", "calibration"),
        ("difficult", "test"),
        ("calibration", "test"),
    ]
    for left, right in pairs:
        overlap = spark_dfs[left].select("index").intersect(spark_dfs[right].select("index"))
        n_overlap = overlap.count()
        if n_overlap > 0:
            aggregated_failures.append(
                f"split_disjointness({left},{right}): {n_overlap} overlapping index value(s)"
            )

    if aggregated_failures:
        raise DataQualityError(DataQualityReport(passed=False, failures=aggregated_failures))
    return reports


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _collect_failures(spark: SparkSession, result: Any) -> list[str]:
    """Convert a PyDeequ ``VerificationResult`` into a list of failure
    strings of the form ``"<constraint>: <message>"``.

    PyDeequ exposes failures via ``checkResultsAsDataFrame``; each row
    has ``constraint``, ``constraint_status``, and ``constraint_message``
    columns. We surface only the rows whose status is not ``Success``.
    """
    df = VerificationResult.checkResultsAsDataFrame(spark, result)
    rows = df.collect()
    failures: list[str] = []
    for row in rows:
        status = row["constraint_status"]
        if status == "Success":
            continue
        constraint = row["constraint"]
        message = row["constraint_message"] or status
        failures.append(f"{constraint}: {message}")
    return failures
