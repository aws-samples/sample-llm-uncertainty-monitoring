# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""AWS Lambda handler for the human-review pipeline.

This Lambda is triggered by an SQS event source mapping
(``MessageBatchSize = 1``). For each SQS record it:

1. **Archives** the full inference payload to S3 under
   ``s3://{HUMAN_REVIEW_BUCKET}/human_review/pending/{run_id}-{message_id}.json``
   so that the complete probability vector, prediction set, and routing
   metadata are durably stored before any notification is sent.

2. **Publishes a one-paragraph summary** to the configured SNS topic so
   that subscribed reviewers receive a concise alert with the question,
   top-2 choices/probabilities, prediction set, routing decision, and a
   direct link to the archived payload in S3.

On any exception during S3 put **or** SNS publish the error is logged
and re-raised so that SQS redrives the message (DLQ handling is
described in ``docs/production-upgrades.md``).

Environment variables
---------------------
``HUMAN_REVIEW_BUCKET``
    S3 bucket for archiving review payloads.
``SNS_TOPIC_ARN``
    ARN of the SNS topic that receives reviewer notifications.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3

__all__ = ["lambda_handler"]


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Lazy clients — constructed once per Lambda execution environment.
# ---------------------------------------------------------------------------

_S3_CLIENT: Any = None
_SNS_CLIENT: Any = None


def _get_s3_client() -> Any:
    """Return the lazily-constructed S3 client."""
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _get_sns_client() -> Any:
    """Return the lazily-constructed SNS client."""
    global _SNS_CLIENT
    if _SNS_CLIENT is None:
        _SNS_CLIENT = boto3.client("sns")
    return _SNS_CLIENT


def _reset_for_testing() -> None:
    """Reset module-level globals for unit tests."""
    global _S3_CLIENT, _SNS_CLIENT
    _S3_CLIENT = None
    _SNS_CLIENT = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summary(payload: dict[str, Any], s3_uri: str) -> str:
    """Build a one-paragraph human-readable summary for the SNS message.

    Includes the question text, top-2 choices with their probabilities,
    the prediction set, the routing decision, and the S3 archive URI.
    """
    question = payload.get("question", "(no question)")
    choices = payload.get("choices", [])
    probabilities = payload.get("probabilities", [])
    prediction_set = payload.get("prediction_set", [])
    routing_decision = payload.get("routing_decision", "unknown")

    # Build top-2 choices by probability
    if probabilities and choices:
        indexed_probs = list(enumerate(probabilities))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        top2 = indexed_probs[:2]
        top2_lines = "; ".join(
            f"[{idx}] {choices[idx] if idx < len(choices) else '?'} (p={prob:.4f})"
            for idx, prob in top2
        )
    else:
        top2_lines = "(no probability data)"

    return (
        f"Uncertain prediction routed for human review.\n"
        f"Question: {question}\n"
        f"Top-2 choices: {top2_lines}\n"
        f"Prediction set: {prediction_set}\n"
        f"Routing decision: {routing_decision}\n"
        f"Full payload archived at: {s3_uri}"
    )


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process a batch of SQS records (batch size = 1 by design).

    For each record:
    1. Archive the full payload to S3.
    2. Publish a summary notification to SNS.

    On failure, the exception propagates so SQS can redrive the message.
    """
    bucket = os.environ.get("HUMAN_REVIEW_BUCKET", "")
    sns_topic_arn = os.environ.get("SNS_TOPIC_ARN", "")

    if not bucket:
        raise RuntimeError("HUMAN_REVIEW_BUCKET environment variable is not set")
    if not sns_topic_arn:
        raise RuntimeError("SNS_TOPIC_ARN environment variable is not set")

    records = event.get("Records", [])
    processed = 0

    for sqs_record in records:
        message_id = sqs_record.get("messageId", "unknown")
        body_raw = sqs_record.get("body", "{}")

        # Parse the JSON body sent by the inference handler.
        payload: dict[str, Any] = json.loads(body_raw)
        run_id = payload.get("run_id", "no-run-id")

        # --- Step 1: Archive to S3 -------------------------------------------
        s3_key = f"human_review/pending/{run_id}-{message_id}.json"
        s3_uri = f"s3://{bucket}/{s3_key}"

        logger.info("Archiving review payload to %s", s3_uri)
        _get_s3_client().put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=body_raw,
            ContentType="application/json",
        )

        # --- Step 2: Publish summary to SNS ----------------------------------
        summary = _build_summary(payload, s3_uri)

        logger.info("Publishing review notification to %s", sns_topic_arn)
        _get_sns_client().publish(
            TopicArn=sns_topic_arn,
            Subject="LLM Uncertainty: Prediction routed for human review",
            Message=summary,
        )

        processed += 1

    return {"statusCode": 200, "body": {"processed": processed}}
