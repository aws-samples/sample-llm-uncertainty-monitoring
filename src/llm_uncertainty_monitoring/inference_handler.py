# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""AWS Lambda entry point for conformal-prediction-based inference.

The handler is invoked once per inference request. It is designed to be
cheap on both cold and warm invocations:

- **Cold start**: a single DynamoDB read (via
  :func:`config.dynamo_store.load_config`) populates module-level globals
  with the calibrated threshold, the SageMaker predictor, and the
  metrics publisher. No further AWS metadata calls happen on the warm
  path.
- **Warm invocation**: validates the event, scores the K candidate
  answer tokens via the prefill scoring contract, builds the LAC
  prediction set, emits per-prediction CloudWatch metrics, and (when the
  prediction is uncertain) dispatches a payload to the configured
  human-review SQS queue.

The handler always returns ``{"statusCode": int, "body": dict}``. Error
mapping (per Design §Lambda):

============= ============================================================
``status``    Cause
============= ============================================================
``200``       success
``400``       malformed event (non-empty string ``question``, ``list[str]``
              ``choices`` of length >= 2, optional integer ``ground_truth``
              within ``[0, len(choices))``)
``502``       two consecutive SQS ``send_message`` failures while routing
              an uncertain prediction
``503``       missing required configuration attribute (caught from
              :func:`config.dynamo_store.load_config`) or missing
              ``human_review_destination_*`` when routing is needed
``500``       any other unexpected exception (logged with traceback)
============= ============================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
from botocore.exceptions import ClientError

from llm_uncertainty_monitoring.config import dynamo_store
from llm_uncertainty_monitoring.conformal import LACScore
from llm_uncertainty_monitoring.metrics_publisher import MetricsPublisher, PredictionRecord
from llm_uncertainty_monitoring.probability_extractor import build_prompt_and_probs

__all__ = ["lambda_handler"]


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)


_RESPONSE_BODY_FIELDS: tuple[str, ...] = (
    "prediction_set",
    "probabilities",
    "top1_prediction",
    "set_size",
    "is_certain",
    "conformal_threshold",
    "routing_decision",
)
"""Exact set of keys returned in the success-path response body."""


# ---------------------------------------------------------------------------
# Module-level globals populated lazily during cold start.
# ---------------------------------------------------------------------------

_BOOTSTRAPPED: bool = False
_CONFIG: dict[str, str] | None = None
_PREDICTOR: Any = None
_METRICS_PUBLISHER: MetricsPublisher | None = None
_SQS_CLIENT: Any = None
_NUM_CHOICES: int | None = None
_Q_HAT: float | None = None
_TARGET_COVERAGE: float | None = None


class _BadRequestError(Exception):
    """Raised by :func:`_validate_event` for malformed input.

    Carries the human-readable ``message`` plus the ``field`` name that
    triggered the failure, so the 400 response body can name the
    offending attribute.
    """

    def __init__(self, message: str, field: str) -> None:
        super().__init__(message)
        self.message = message
        self.field = field


# ---------------------------------------------------------------------------
# Bootstrap / cache reset
# ---------------------------------------------------------------------------


def _reset_for_testing() -> None:
    """Reset every module-level global so a fresh cold start can be
    simulated under test.

    Private — never call from production code. The companion
    :func:`config.dynamo_store.reset_cache` clears the cached config
    item so the next ``load_config`` re-reads from the (mocked) table.
    """
    global _BOOTSTRAPPED, _CONFIG, _PREDICTOR, _METRICS_PUBLISHER
    global _SQS_CLIENT, _NUM_CHOICES, _Q_HAT, _TARGET_COVERAGE
    _BOOTSTRAPPED = False
    _CONFIG = None
    _PREDICTOR = None
    _METRICS_PUBLISHER = None
    _SQS_CLIENT = None
    _NUM_CHOICES = None
    _Q_HAT = None
    _TARGET_COVERAGE = None
    dynamo_store.reset_cache()


def _ensure_bootstrapped() -> None:
    """Populate module-level globals on the first invocation of a cold
    Lambda execution environment.

    Subsequent calls are no-ops thanks to the ``_BOOTSTRAPPED`` flag, so
    warm invocations make zero AWS metadata calls. ``ValueError`` raised
    by :func:`config.dynamo_store.load_config` (missing item or missing
    required attribute) propagates to the caller, which maps it to a
    503 response.
    """
    global _BOOTSTRAPPED, _CONFIG, _PREDICTOR, _METRICS_PUBLISHER
    global _NUM_CHOICES, _Q_HAT, _TARGET_COVERAGE

    if _BOOTSTRAPPED:
        return

    config_id = os.environ.get("CONFIG_ID", "default")
    table_name = os.environ.get("CONFIG_TABLE", dynamo_store.DEFAULT_CONFIG_TABLE)
    config = dynamo_store.load_config(config_id=config_id, table_name=table_name)

    # Use Boto3Predictor (no sagemaker SDK dependency — boto3 is pre-installed in Lambda)
    from llm_uncertainty_monitoring.probability_extractor import Boto3Predictor

    predictor = Boto3Predictor(endpoint_name=config["endpoint_name"])

    metrics_publisher = MetricsPublisher(
        namespace=config["cloudwatch_namespace"],
        dimensions={
            "EndpointName": config["endpoint_name"],
            "ModelId": config["model_id"],
            "TargetCoverage": config["target_coverage"],
        },
    )

    _CONFIG = config
    _PREDICTOR = predictor
    _METRICS_PUBLISHER = metrics_publisher
    _NUM_CHOICES = int(config["num_choices"])
    _Q_HAT = float(config["conformal_threshold"])
    _TARGET_COVERAGE = float(config["target_coverage"])
    _BOOTSTRAPPED = True


def _get_sqs_client() -> Any:
    """Return the lazily-constructed SQS client.

    Built only when the first uncertain prediction needs routing, so
    Lambdas serving exclusively certain traffic never pay the boto3
    client-construction cost.
    """
    global _SQS_CLIENT
    if _SQS_CLIENT is None:
        _SQS_CLIENT = boto3.client("sqs")
    return _SQS_CLIENT


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------


def _validate_event(event: Any) -> tuple[str, list[str], int | None]:
    """Validate ``event`` against the input contract.

    Returns ``(question, choices, ground_truth)`` on success. Raises
    :class:`_BadRequestError` carrying a ``field`` name on any contract
    violation; the caller turns the exception into a 400 response.
    """
    if not isinstance(event, dict):
        raise _BadRequestError("event must be a JSON object", field="event")

    question = event.get("question")
    if not isinstance(question, str) or not question.strip():
        raise _BadRequestError("question must be a non-empty string", field="question")

    choices = event.get("choices")
    if not isinstance(choices, list):
        raise _BadRequestError("choices must be a list of strings", field="choices")
    if len(choices) < 2:
        raise _BadRequestError(
            f"choices must contain at least 2 entries, got {len(choices)}",
            field="choices",
        )
    if not all(isinstance(choice, str) for choice in choices):
        raise _BadRequestError("every choices entry must be a string", field="choices")

    raw_ground_truth = event.get("ground_truth")
    ground_truth: int | None
    if raw_ground_truth is None:
        ground_truth = None
    else:
        # ``bool`` is a subclass of ``int``; reject it explicitly.
        if isinstance(raw_ground_truth, bool) or not isinstance(raw_ground_truth, int):
            raise _BadRequestError("ground_truth must be an integer", field="ground_truth")
        if not (0 <= raw_ground_truth < len(choices)):
            raise _BadRequestError(
                f"ground_truth must be in [0, {len(choices)}), got {raw_ground_truth}",
                field="ground_truth",
            )
        ground_truth = raw_ground_truth

    return question, list(choices), ground_truth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sqs_url_from_arn(arn: str) -> str:
    """Convert an ``arn:aws:sqs:{region}:{account}:{queue}`` ARN into the
    canonical HTTPS queue URL.

    ``ValueError`` is raised for malformed ARNs; this propagates to the
    top-level handler as a 500.
    """
    parts = arn.split(":")
    if len(parts) != 6 or parts[0] != "arn" or parts[1] != "aws" or parts[2] != "sqs":
        raise ValueError(f"unexpected SQS ARN format: {arn!r}")
    region, account, queue = parts[3], parts[4], parts[5]
    if not region or not account or not queue:
        raise ValueError(f"unexpected SQS ARN format: {arn!r}")
    return f"https://sqs.{region}.amazonaws.com/{account}/{queue}"


def _route_to_human_review(
    *,
    question: str,
    choices: list[str],
    probabilities: list[float],
    prediction_set: list[int],
    top1_prediction: int,
    ground_truth: int | None,
    conformal_threshold: float,
    routing_decision: str,
) -> dict[str, Any] | None:
    """Dispatch the human-review payload to SQS, with one retry.

    Returns ``None`` on success and a ready-to-return error response
    dict on failure. The response uses status 503 when configuration is
    incomplete (missing ARN or destination type) and 502 when both SQS
    ``send_message`` calls fail.
    """
    assert _CONFIG is not None  # _ensure_bootstrapped guarantees this

    arn = _CONFIG.get("human_review_destination_arn") or os.environ.get(
        "HUMAN_REVIEW_DESTINATION_ARN"
    )
    if not arn:
        return {
            "statusCode": 503,
            "body": {
                "error": (
                    "human_review_destination_arn missing from configuration; "
                    "cannot route uncertain predictions"
                ),
                "field": "human_review_destination_arn",
            },
        }
    destination_type = _CONFIG.get("human_review_destination_type")
    if not destination_type:
        return {
            "statusCode": 503,
            "body": {
                "error": (
                    "human_review_destination_type missing from configuration; "
                    "cannot route uncertain predictions"
                ),
                "field": "human_review_destination_type",
            },
        }

    queue_url = _sqs_url_from_arn(arn)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload = {
        "run_id": run_id,
        "question": question,
        "choices": choices,
        "probabilities": probabilities,
        "prediction_set": prediction_set,
        "top1_prediction": top1_prediction,
        "ground_truth": ground_truth,
        "conformal_threshold": conformal_threshold,
        "routing_decision": routing_decision,
        "endpoint_name": _CONFIG["endpoint_name"],
    }
    message_body = json.dumps(payload)

    sqs_client = _get_sqs_client()
    last_error: str | None = None
    for attempt in range(2):
        try:
            sqs_client.send_message(QueueUrl=queue_url, MessageBody=message_body)
            return None
        except ClientError as err:
            last_error = err.response.get("Error", {}).get("Message", str(err))
            logger.error(
                "SQS send_message attempt %d/2 failed for queue %s: %s",
                attempt + 1,
                queue_url,
                last_error,
            )

    return {
        "statusCode": 502,
        "body": {
            "error": last_error or "SQS dispatch failed",
            "routing_decision": routing_decision,
            "human_review_destination_arn": arn,
        },
    }


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Score one inference event and return a conformal prediction set.

    See module docstring for the response and error contracts.
    """
    try:
        try:
            _ensure_bootstrapped()
        except ValueError as err:
            logger.error("bootstrap failed: %s", err)
            return {"statusCode": 503, "body": {"error": str(err)}}

        try:
            question, choices, ground_truth = _validate_event(event)
        except _BadRequestError as err:
            return {
                "statusCode": 400,
                "body": {"error": err.message, "field": err.field},
            }

        # Bootstrap guarantees every cached global is populated; the
        # explicit asserts narrow the optional types for mypy.
        assert _PREDICTOR is not None
        assert _METRICS_PUBLISHER is not None
        assert _NUM_CHOICES is not None
        assert _Q_HAT is not None

        num_choices = _NUM_CHOICES
        q_hat = _Q_HAT
        answer_tokens = tuple(str(i) for i in range(num_choices))

        start = time.perf_counter()
        _, probs = build_prompt_and_probs(
            _PREDICTOR, question, choices, answer_tokens=answer_tokens
        )
        probabilities = [float(p) for p in probs]
        prediction_set = LACScore().build_prediction_set(probabilities, q_hat)
        prediction_latency_ms = (time.perf_counter() - start) * 1000.0

        top1_prediction = int(np.argmax(probs))
        set_size = len(prediction_set)
        is_certain = set_size == 1
        routing_decision = "auto" if is_certain else "human_review"

        if ground_truth is None:
            covered: int | None = None
            top1_correct: int | None = None
        else:
            covered = int(ground_truth in prediction_set)
            top1_correct = int(ground_truth == top1_prediction)

        body: dict[str, Any] = {
            "prediction_set": prediction_set,
            "probabilities": probabilities,
            "top1_prediction": top1_prediction,
            "set_size": set_size,
            "is_certain": is_certain,
            "conformal_threshold": q_hat,
            "routing_decision": routing_decision,
        }

        record = PredictionRecord(
            question=question,
            choices=choices,
            probabilities=probabilities,
            prediction_set=prediction_set,
            top1_prediction=top1_prediction,
            set_size=set_size,
            is_certain=is_certain,
            conformal_threshold=q_hat,
            prediction_latency_ms=prediction_latency_ms,
            routing_decision=routing_decision,
            ground_truth=ground_truth,
            covered=covered,
            top1_correct=top1_correct,
        )
        _METRICS_PUBLISHER.publish_prediction(record)
        _METRICS_PUBLISHER.flush()

        if routing_decision == "human_review":
            error_response = _route_to_human_review(
                question=question,
                choices=choices,
                probabilities=probabilities,
                prediction_set=prediction_set,
                top1_prediction=top1_prediction,
                ground_truth=ground_truth,
                conformal_threshold=q_hat,
                routing_decision=routing_decision,
            )
            if error_response is not None:
                return error_response

        return {"statusCode": 200, "body": body}

    except Exception as err:  # noqa: BLE001 — defensive top-level catch (Req 6.6 + 500 mapping)
        logger.error(
            "unexpected exception in lambda_handler: %s\n%s",
            err,
            traceback.format_exc(),
        )
        return {"statusCode": 500, "body": {"error": str(err)}}
