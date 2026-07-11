# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CloudWatch metric publication and batch JSON archival.

The :class:`MetricsPublisher` is used from two execution contexts:

- the inference Lambda calls :meth:`MetricsPublisher.publish_prediction`
  once per request and :meth:`MetricsPublisher.flush` before returning,
  so a fresh datum is visible in CloudWatch within the invocation;
- the periodic batch driver calls :meth:`MetricsPublisher.publish_batch`
  once per labeled run to emit the aggregate metrics and persist a full
  JSON snapshot (records + aggregates) to S3 for downstream tooling
  (``uncertainty_explainer``, ad-hoc analysis).

Two design rules drive the implementation:

1. CloudWatch failures are isolated from the caller. Every
   ``PutMetricData`` and ``PutObject`` call is wrapped in a
   ``try/except`` that logs and never re-raises, so a degraded
   CloudWatch or S3 cannot break the inference Lambda's response.
2. ``PutMetricData`` accepts at most 20 ``MetricDatum`` per call, so the
   per-prediction publisher buffers up to 20 entries and flushes
   automatically when full.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

__all__ = [
    "PUT_METRIC_DATA_MAX_BATCH",
    "PredictionRecord",
    "MetricsPublisher",
]


logger = logging.getLogger(__name__)


PUT_METRIC_DATA_MAX_BATCH: int = 20
"""Hard limit imposed by the CloudWatch ``PutMetricData`` API."""


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction along with the signals needed for monitoring.

    Per-prediction emission populates ``set_size``, ``is_certain``, the
    top probability (``max(probabilities)``), and
    ``prediction_latency_ms``. When ``ground_truth`` is present, the
    accuracy signals ``covered`` (whether the truth fell inside the
    prediction set) and ``top1_correct`` (whether the argmax matched the
    truth) are also emitted. ``routing_decision`` is either ``"auto"``
    or ``"human_review"`` and drives the ``RoutedToHumanReview`` flag.
    """

    question: str
    choices: list[str]
    probabilities: list[float]
    prediction_set: list[int]
    top1_prediction: int
    set_size: int
    is_certain: bool
    conformal_threshold: float
    prediction_latency_ms: float
    routing_decision: str
    ground_truth: int | None
    covered: int | None
    top1_correct: int | None


class MetricsPublisher:
    """Publishes per-prediction metrics to CloudWatch and batch
    aggregates plus the full record set to CloudWatch + S3.

    A single publisher instance owns a small in-memory buffer of
    ``MetricDatum`` dicts. The buffer is flushed automatically whenever
    it reaches :data:`PUT_METRIC_DATA_MAX_BATCH` and explicitly via
    :meth:`flush`. The CloudWatch and S3 clients are constructed lazily
    on first use so test code can inject ``MagicMock`` instances without
    triggering a real AWS session.
    """

    def __init__(
        self,
        namespace: str,
        dimensions: dict[str, str],
        cloudwatch_client: Any | None = None,
        s3_client: Any | None = None,
    ) -> None:
        self.namespace = namespace
        self.dimensions = dict(dimensions)
        self._cloudwatch_client = cloudwatch_client
        self._s3_client = s3_client
        self._buffer: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-prediction
    # ------------------------------------------------------------------

    def publish_prediction(self, record: PredictionRecord) -> None:
        """Buffer the per-prediction metrics for ``record``.

        Always emits ``SetSize``, ``IsCertain`` (0/1), ``TopProbability``
        (``max(record.probabilities)``), and ``PredictionLatencyMs``.
        When ``record.ground_truth is not None``, also emits ``Covered``
        and ``Top1Correct`` (each 0/1). When
        ``record.routing_decision == "human_review"``, also emits
        ``RoutedToHumanReview`` with value ``1``.

        The buffer is flushed automatically as soon as it reaches the
        :data:`PUT_METRIC_DATA_MAX_BATCH` limit.
        """
        top_probability = float(max(record.probabilities)) if record.probabilities else 0.0

        self._enqueue("SetSize", float(record.set_size), "Count")
        self._enqueue("IsCertain", 1.0 if record.is_certain else 0.0, "Count")
        self._enqueue("TopProbability", top_probability, "None")
        self._enqueue("PredictionLatencyMs", float(record.prediction_latency_ms), "Milliseconds")

        if record.ground_truth is not None:
            if record.covered is not None:
                self._enqueue("Covered", float(record.covered), "Count")
            if record.top1_correct is not None:
                self._enqueue("Top1Correct", float(record.top1_correct), "Count")

        if record.routing_decision == "human_review":
            self._enqueue("RoutedToHumanReview", 1.0, "Count")

        # Auto-flush every time we hit the CloudWatch batch limit. Any
        # leftover (less than 20) is flushed when the caller calls
        # ``flush`` explicitly (e.g., the Lambda before returning).
        while len(self._buffer) >= PUT_METRIC_DATA_MAX_BATCH:
            self._flush_chunk(self._buffer[:PUT_METRIC_DATA_MAX_BATCH])
            del self._buffer[:PUT_METRIC_DATA_MAX_BATCH]

    def flush(self) -> None:
        """Send any buffered metrics to CloudWatch and clear the buffer.

        Safe to call when the buffer is empty. CloudWatch failures are
        logged and swallowed; the buffer is cleared regardless of
        success so a transient outage does not cause unbounded growth.
        """
        if not self._buffer:
            return
        # Send in chunks of at most 20 to be defensive even if upstream
        # callers populated the buffer directly.
        chunks = [
            self._buffer[i : i + PUT_METRIC_DATA_MAX_BATCH]
            for i in range(0, len(self._buffer), PUT_METRIC_DATA_MAX_BATCH)
        ]
        self._buffer = []
        for chunk in chunks:
            self._flush_chunk(chunk)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def publish_batch(
        self,
        records: list[PredictionRecord],
        s3_uri: str,
        run_id: str | None = None,
    ) -> None:
        """Emit batch aggregates to CloudWatch and persist a JSON
        artifact to S3.

        Computes six aggregates over ``records``:

        - ``AveragePredictionSetSize`` — mean of ``set_size``.
        - ``AverageTopProbability`` — mean of ``max(probabilities)``.
        - ``AveragePredictionLatencyMs`` — mean of
          ``prediction_latency_ms``.
        - ``Coverage`` — mean of ``covered`` over records where it is
          not ``None`` (skipped when no record has a ground truth).
        - ``Top1Correct`` — mean of ``top1_correct`` over records where
          it is not ``None`` (skipped when no record has a ground
          truth).
        - ``RoutedToHumanReviewFraction`` — fraction of records whose
          ``routing_decision`` is ``"human_review"``.

        All aggregates are published in a single ``PutMetricData`` call
        with the publisher's dimensions. The full record set plus the
        aggregates are then serialized to JSON and uploaded to
        ``s3_uri``. Both the CloudWatch and S3 calls are wrapped in
        ``try/except`` so a transient outage on either path does not
        propagate.
        """
        if not records:
            raise ValueError("publish_batch requires a non-empty list of records")

        resolved_run_id = (
            run_id
            if run_id is not None
            else f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )

        aggregates = self._compute_aggregates(records)

        # Emit aggregates as a single PutMetricData call. The dimension
        # contract for each datum is the publisher's dimensions.
        metric_data = [self._build_datum(name, value, "None") for name, value in aggregates.items()]
        if metric_data:
            self._flush_chunk(metric_data)

        document = {
            "run_id": resolved_run_id,
            "batch_size": len(records),
            "dimensions": self.dimensions,
            "aggregates": aggregates,
            "records": [asdict(record) for record in records],
        }

        bucket, key = _parse_s3_uri(s3_uri)
        body = json.dumps(document).encode("utf-8")
        try:
            self._s3.put_object(Bucket=bucket, Key=key, Body=body)
        except ClientError as err:
            logger.error(
                "S3 PutObject failed for batch artifact s3://%s/%s: %s",
                bucket,
                key,
                err.response.get("Error", {}).get("Message", str(err)),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_aggregates(records: list[PredictionRecord]) -> dict[str, float]:
        """Compute the six batch aggregates documented above.

        ``Coverage`` and ``Top1Correct`` are omitted when no record
        carries a ground truth (otherwise the mean would be undefined).
        """
        n = len(records)
        aggregates: dict[str, float] = {}

        aggregates["AveragePredictionSetSize"] = sum(r.set_size for r in records) / n
        aggregates["AverageTopProbability"] = (
            sum(float(max(r.probabilities)) if r.probabilities else 0.0 for r in records) / n
        )
        aggregates["AveragePredictionLatencyMs"] = sum(r.prediction_latency_ms for r in records) / n

        covered_values = [r.covered for r in records if r.covered is not None]
        if covered_values:
            aggregates["Coverage"] = sum(covered_values) / len(covered_values)

        top1_values = [r.top1_correct for r in records if r.top1_correct is not None]
        if top1_values:
            aggregates["Top1Correct"] = sum(top1_values) / len(top1_values)

        aggregates["RoutedToHumanReview"] = (
            sum(1 for r in records if r.routing_decision == "human_review") / n
        )

        # Certain vs uncertain accuracy (only when ground truth is available)
        certain_with_gt = [r for r in records if r.is_certain and r.top1_correct is not None]
        uncertain_with_gt = [r for r in records if not r.is_certain and r.top1_correct is not None]
        if certain_with_gt:
            aggregates["CertainAccuracy"] = sum(
                r.top1_correct for r in certain_with_gt if r.top1_correct is not None
            ) / len(certain_with_gt)
        if uncertain_with_gt:
            aggregates["UncertainAccuracy"] = sum(
                r.top1_correct for r in uncertain_with_gt if r.top1_correct is not None
            ) / len(uncertain_with_gt)

        return aggregates

    def _enqueue(self, name: str, value: float, unit: str) -> None:
        """Append a ``MetricDatum`` to the per-prediction buffer.

        For ``SetSize``, also publishes an undimensioned copy so that
        CloudWatch alarms can match without knowing the endpoint name.
        """
        self._buffer.append(self._build_datum(name, value, unit))
        # Undimensioned copy for alarm matching (SetSize only)
        if name == "SetSize":
            self._buffer.append(
                {
                    "MetricName": name,
                    "Value": float(value),
                    "Unit": unit,
                }
            )

    def _build_datum(self, name: str, value: float, unit: str) -> dict[str, Any]:
        """Build a single ``MetricDatum`` dict carrying the publisher's
        dimensions in the AWS-expected ``[{"Name": ..., "Value": ...}]``
        shape."""
        return {
            "MetricName": name,
            "Value": float(value),
            "Unit": unit,
            "Dimensions": [
                {"Name": dim_name, "Value": dim_value}
                for dim_name, dim_value in self.dimensions.items()
            ],
        }

    def _flush_chunk(self, chunk: list[dict[str, Any]]) -> None:
        """Send a single ``PutMetricData`` call.

        Any :class:`botocore.exceptions.ClientError` is logged and
        swallowed so an upstream caller (the inference Lambda) is never
        broken by a CloudWatch outage.
        """
        if not chunk:
            return
        try:
            self._cloudwatch.put_metric_data(Namespace=self.namespace, MetricData=chunk)
        except ClientError as err:
            metric_names = sorted({datum["MetricName"] for datum in chunk})
            logger.error(
                "CloudWatch PutMetricData failed for namespace=%s metrics=%s: %s",
                self.namespace,
                metric_names,
                err.response.get("Error", {}).get("Message", str(err)),
            )

    @property
    def _cloudwatch(self) -> Any:
        if self._cloudwatch_client is None:
            self._cloudwatch_client = boto3.client("cloudwatch")
        return self._cloudwatch_client

    @property
    def _s3(self) -> Any:
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Raises ``ValueError`` for any URI that is not S3-shaped or that has
    an empty bucket/key.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"expected an s3:// URI, got {s3_uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must include bucket and key, got {s3_uri!r}")
    return bucket, key
