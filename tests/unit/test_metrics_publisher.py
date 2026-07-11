# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit and property tests for ``llm_uncertainty_monitoring.metrics_publisher``.

Covers:

- the batch aggregates emitted by ``publish_batch`` match the
  directly-computed means, and the JSON artifact uploaded to S3 carries
  the same aggregates dict;
- per-prediction emission isolates the caller from CloudWatch failures;
- every emitted ``MetricDatum`` carries the configured dimensions;
- ``PutMetricData`` is called with batches of at most 20 entries.
"""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from llm_uncertainty_monitoring.metrics_publisher import (
    PUT_METRIC_DATA_MAX_BATCH,
    MetricsPublisher,
    PredictionRecord,
)

DIMENSIONS = {
    "EndpointName": "ep-llama-3-8b",
    "ModelId": "meta-textgeneration-llama-3-8b",
    "TargetCoverage": "0.90",
}
NAMESPACE = "LLMUncertainty"
S3_URI = "s3://DOC-EXAMPLE-BUCKET/path/to/batch.json"


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


@st.composite
def prediction_record(draw: st.DrawFn) -> PredictionRecord:
    """Build a ``PredictionRecord`` with random monitoring signals."""
    k = draw(st.integers(min_value=2, max_value=6))
    set_size = draw(st.integers(min_value=1, max_value=k))
    is_certain = set_size == 1
    routing_decision = "auto" if is_certain else draw(st.sampled_from(["auto", "human_review"]))
    has_ground_truth = draw(st.booleans())
    if has_ground_truth:
        ground_truth: int | None = draw(st.integers(min_value=0, max_value=k - 1))
        covered: int | None = draw(st.integers(min_value=0, max_value=1))
        top1_correct: int | None = draw(st.integers(min_value=0, max_value=1))
    else:
        ground_truth = None
        covered = None
        top1_correct = None

    probabilities = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=k,
            max_size=k,
        )
    )
    latency = draw(st.floats(min_value=0.0, max_value=2_000.0, allow_nan=False))

    return PredictionRecord(
        question="q?",
        choices=[f"c{i}" for i in range(k)],
        probabilities=probabilities,
        prediction_set=list(range(set_size)),
        top1_prediction=0,
        set_size=set_size,
        is_certain=is_certain,
        conformal_threshold=0.5,
        prediction_latency_ms=latency,
        routing_decision=routing_decision,
        ground_truth=ground_truth,
        covered=covered,
        top1_correct=top1_correct,
    )


# ---------------------------------------------------------------------------
# Property: batch aggregates equal direct-computed means
# ---------------------------------------------------------------------------


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(records=st.lists(prediction_record(), min_size=1, max_size=50))
def test_publish_batch_aggregates_equal_direct_means(records: list[PredictionRecord]) -> None:
    """Every aggregate emitted by ``publish_batch`` equals the
    directly-computed mean within ``1e-9``, and ``s3.put_object`` is
    called once with a JSON body whose ``aggregates`` dict matches."""
    cloudwatch = MagicMock(name="cloudwatch")
    s3 = MagicMock(name="s3")

    publisher = MetricsPublisher(
        namespace=NAMESPACE,
        dimensions=DIMENSIONS,
        cloudwatch_client=cloudwatch,
        s3_client=s3,
    )
    publisher.publish_batch(records, S3_URI, run_id="test-run")

    n = len(records)
    expected_aggregates: dict[str, float] = {
        "AveragePredictionSetSize": sum(r.set_size for r in records) / n,
        "AverageTopProbability": sum(
            float(max(r.probabilities)) if r.probabilities else 0.0 for r in records
        )
        / n,
        "AveragePredictionLatencyMs": sum(r.prediction_latency_ms for r in records) / n,
        "RoutedToHumanReview": sum(1 for r in records if r.routing_decision == "human_review") / n,
    }
    covered_values = [r.covered for r in records if r.covered is not None]
    if covered_values:
        expected_aggregates["Coverage"] = sum(covered_values) / len(covered_values)
    top1_values = [r.top1_correct for r in records if r.top1_correct is not None]
    if top1_values:
        expected_aggregates["Top1Correct"] = sum(top1_values) / len(top1_values)
    certain_with_gt = [r for r in records if r.is_certain and r.top1_correct is not None]
    if certain_with_gt:
        expected_aggregates["CertainAccuracy"] = sum(r.top1_correct for r in certain_with_gt) / len(
            certain_with_gt
        )
    uncertain_with_gt = [r for r in records if not r.is_certain and r.top1_correct is not None]
    if uncertain_with_gt:
        expected_aggregates["UncertainAccuracy"] = sum(
            r.top1_correct for r in uncertain_with_gt
        ) / len(uncertain_with_gt)

    # CloudWatch was called once with one MetricDatum per aggregate.
    assert cloudwatch.put_metric_data.call_count == 1
    cw_kwargs = cloudwatch.put_metric_data.call_args.kwargs
    assert cw_kwargs["Namespace"] == NAMESPACE
    emitted = {datum["MetricName"]: datum["Value"] for datum in cw_kwargs["MetricData"]}
    assert set(emitted) == set(expected_aggregates)
    for name, expected in expected_aggregates.items():
        assert math.isclose(emitted[name], expected, abs_tol=1e-9)

    # S3 PutObject was called once and the JSON aggregates dict matches.
    assert s3.put_object.call_count == 1
    put_kwargs = s3.put_object.call_args.kwargs
    assert put_kwargs["Bucket"] == "DOC-EXAMPLE-BUCKET"
    assert put_kwargs["Key"] == "path/to/batch.json"
    document = json.loads(put_kwargs["Body"].decode("utf-8"))
    assert document["run_id"] == "test-run"
    assert document["batch_size"] == n
    assert document["dimensions"] == DIMENSIONS
    assert set(document["aggregates"]) == set(expected_aggregates)
    for name, expected in expected_aggregates.items():
        assert math.isclose(document["aggregates"][name], expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Per-prediction unit tests
# ---------------------------------------------------------------------------


def _make_record(
    *,
    set_size: int = 1,
    is_certain: bool = True,
    routing_decision: str = "auto",
    ground_truth: int | None = None,
    covered: int | None = None,
    top1_correct: int | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        question="q?",
        choices=["a", "b", "c", "d"],
        probabilities=[0.7, 0.1, 0.1, 0.1],
        prediction_set=list(range(set_size)),
        top1_prediction=0,
        set_size=set_size,
        is_certain=is_certain,
        conformal_threshold=0.5,
        prediction_latency_ms=12.5,
        routing_decision=routing_decision,
        ground_truth=ground_truth,
        covered=covered,
        top1_correct=top1_correct,
    )


def _client_error() -> ClientError:
    return ClientError(
        error_response={
            "Error": {"Code": "Throttling", "Message": "rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        operation_name="PutMetricData",
    )


def test_publish_prediction_isolates_cloudwatch_failures() -> None:
    """A CloudWatch client that raises on every call must not propagate
    the error out of ``publish_prediction``/``flush``."""
    cloudwatch = MagicMock(name="cloudwatch")
    cloudwatch.put_metric_data.side_effect = _client_error()

    publisher = MetricsPublisher(
        namespace=NAMESPACE,
        dimensions=DIMENSIONS,
        cloudwatch_client=cloudwatch,
        s3_client=MagicMock(name="s3"),
    )

    # 25 predictions -> guaranteed at least one auto-flush plus a
    # trailing flush. Neither path may raise.
    for _ in range(25):
        publisher.publish_prediction(
            _make_record(ground_truth=0, covered=1, top1_correct=1, routing_decision="human_review")
        )
    publisher.flush()

    assert cloudwatch.put_metric_data.called


def test_every_emitted_datum_carries_configured_dimensions() -> None:
    """Every ``MetricDatum`` emitted (per-prediction or batch) carries
    the publisher's three dimensions in the AWS shape."""
    cloudwatch = MagicMock(name="cloudwatch")
    s3 = MagicMock(name="s3")

    publisher = MetricsPublisher(
        namespace=NAMESPACE,
        dimensions=DIMENSIONS,
        cloudwatch_client=cloudwatch,
        s3_client=s3,
    )

    publisher.publish_prediction(
        _make_record(
            set_size=2,
            is_certain=False,
            routing_decision="human_review",
            ground_truth=0,
            covered=1,
            top1_correct=1,
        )
    )
    publisher.flush()

    # Aggregate emission shares the same datum-builder so we cover both
    # code paths with one assertion loop.
    publisher.publish_batch([_make_record()], S3_URI)

    expected_dimensions = [
        {"Name": "EndpointName", "Value": "ep-llama-3-8b"},
        {"Name": "ModelId", "Value": "meta-textgeneration-llama-3-8b"},
        {"Name": "TargetCoverage", "Value": "0.90"},
    ]
    expected_set = {(d["Name"], d["Value"]) for d in expected_dimensions}

    all_data: list[dict[str, Any]] = []
    for call in cloudwatch.put_metric_data.call_args_list:
        all_data.extend(call.kwargs["MetricData"])

    assert all_data, "expected at least one MetricDatum to be emitted"
    for datum in all_data:
        # Skip undimensioned datums (SetSize alarm copy)
        if "Dimensions" not in datum:
            continue
        actual_set = {(d["Name"], d["Value"]) for d in datum["Dimensions"]}
        assert actual_set == expected_set


def test_put_metric_data_batches_respect_20_metric_limit() -> None:
    """Publishing 100 predictions emits CloudWatch in chunks of at most
    20 ``MetricDatum`` per call."""
    cloudwatch = MagicMock(name="cloudwatch")

    publisher = MetricsPublisher(
        namespace=NAMESPACE,
        dimensions=DIMENSIONS,
        cloudwatch_client=cloudwatch,
        s3_client=MagicMock(name="s3"),
    )

    # 5 metrics per record (4 dimensioned + 1 undimensioned SetSize) * 100 = 500
    # MetricDatum -> at most ceil(500 / 20) = 25 PutMetricData calls.
    n = 100
    metrics_per_record = 5
    for _ in range(n):
        publisher.publish_prediction(_make_record())
    publisher.flush()

    expected_total = n * metrics_per_record
    expected_max_calls = math.ceil(expected_total / PUT_METRIC_DATA_MAX_BATCH)

    assert cloudwatch.put_metric_data.call_count <= expected_max_calls
    for call in cloudwatch.put_metric_data.call_args_list:
        assert len(call.kwargs["MetricData"]) <= PUT_METRIC_DATA_MAX_BATCH

    total_emitted = sum(
        len(call.kwargs["MetricData"]) for call in cloudwatch.put_metric_data.call_args_list
    )
    assert total_emitted == expected_total
