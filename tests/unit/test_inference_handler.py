# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for ``llm_uncertainty_monitoring.inference_handler``.

Five tests cover the contract the demo cares about:

1. Happy path, certain prediction — returns 200 with ``routing_decision == "auto"``.
2. Happy path, uncertain prediction — returns 200 with
   ``routing_decision == "human_review"`` and dispatches a message to SQS.
3. Malformed input — returns 400 naming the offending field.
4. Missing required configuration attribute — returns 503 naming the attribute.
5. Two consecutive SQS failures while routing an uncertain prediction —
   returns 502 and stops after the second attempt (no infinite retries).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from botocore.exceptions import ClientError

from llm_uncertainty_monitoring import inference_handler

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_CONFIG: dict[str, str] = {
    "endpoint_name": "ep-llama-3-8b",
    "model_id": "meta-textgeneration-llama-3-8b",
    "num_choices": "4",
    "cloudwatch_namespace": "LLMUncertainty",
    "conformal_threshold": "0.30",
    "target_coverage": "0.90",
    "human_review_destination_arn": "arn:aws:sqs:us-east-1:123456789012:Human_Review_Queue",
    "human_review_destination_type": "sqs",
}

_EVENT: dict[str, Any] = {
    "question": "What is the primary purpose of market segmentation?",
    "choices": [
        "Reduce production cost",
        "Identify target customer groups",
        "Eliminate competition",
        "Increase advertising spend",
    ],
}


def _make_predictor(logprobs: dict[str, float]) -> MagicMock:
    """Return a ``MagicMock`` shaped like a SageMaker ``Predictor``.

    Dispatches on the trailing answer token in ``payload["inputs"]`` and
    returns a JumpStart-style response carrying the configured
    ``logprob`` for that token.
    """
    predictor = MagicMock(name="Predictor")

    def _predict(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        token = payload["inputs"].rsplit(" ", 1)[-1]
        return {
            "details": {
                "prefill": [
                    {"id": 1, "text": "ignored", "logprob": -10.0},
                    {"id": 2, "text": token, "logprob": logprobs[token]},
                ],
            },
            "generated_text": token,
        }

    predictor.predict.side_effect = _predict
    return predictor


@contextmanager
def _stubbed_handler(config: dict[str, str], logprobs: dict[str, float]) -> Iterator[MagicMock]:
    """Patch every external dependency of the inference Lambda.

    Yields the SQS client mock so tests can assert on dispatch behaviour.
    Resets module-level cold-start globals so each test starts clean.
    """
    predictor = _make_predictor(logprobs)
    sqs_client = MagicMock(name="sqs")
    with (
        patch.object(inference_handler.dynamo_store, "load_config", return_value=config),
        patch(
            "llm_uncertainty_monitoring.probability_extractor.Boto3Predictor",
            return_value=predictor,
        ),
        patch.object(inference_handler, "MetricsPublisher"),
        patch.object(inference_handler.boto3, "client", return_value=sqs_client),
    ):
        inference_handler._reset_for_testing()
        yield sqs_client


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_certain_prediction_returns_auto() -> None:
    """A peaked probability vector clears the LAC threshold for a single
    label, yielding a one-element prediction set and ``routing_decision == "auto"``."""
    # Token "1" dominates -> softmax peaked at index 1.
    logprobs = {"0": -10.0, "1": 0.0, "2": -10.0, "3": -10.0}

    with _stubbed_handler(dict(_CONFIG), logprobs) as sqs_client:
        response = inference_handler.lambda_handler(_EVENT, context=None)

    assert response["statusCode"] == 200
    body = response["body"]
    assert body["routing_decision"] == "auto"
    assert body["is_certain"] is True
    assert body["set_size"] == 1
    assert body["prediction_set"] == [1]
    assert body["top1_prediction"] == int(np.argmax(body["probabilities"]))
    sqs_client.send_message.assert_not_called()


def test_uncertain_prediction_routes_to_sqs() -> None:
    """A near-uniform probability vector with a high threshold yields a
    multi-element prediction set, ``routing_decision == "human_review"``,
    and a single SQS dispatch."""
    config = dict(_CONFIG)
    config["conformal_threshold"] = "0.99"  # forces multi-element prediction set
    logprobs = {"0": -0.1, "1": -0.2, "2": -0.3, "3": -0.4}

    with _stubbed_handler(config, logprobs) as sqs_client:
        response = inference_handler.lambda_handler(_EVENT, context=None)

    assert response["statusCode"] == 200
    body = response["body"]
    assert body["routing_decision"] == "human_review"
    assert body["is_certain"] is False
    assert body["set_size"] >= 2
    sqs_client.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# Error status codes
# ---------------------------------------------------------------------------


def test_400_when_event_is_malformed() -> None:
    """A request missing ``question`` is rejected with a 400 naming the field."""
    logprobs = {"0": -1.0, "1": -1.0, "2": -1.0, "3": -1.0}
    with _stubbed_handler(dict(_CONFIG), logprobs):
        response = inference_handler.lambda_handler({"choices": _EVENT["choices"]}, context=None)
    assert response["statusCode"] == 400
    assert response["body"]["field"] == "question"


def test_503_when_required_config_attribute_missing() -> None:
    """``dynamo_store.load_config`` raising ``ValueError`` for a missing
    required attribute maps to a 503 response that names the attribute."""
    error_message = (
        "required configuration attribute 'conformal_threshold' missing "
        "from item config_id='default' in table 'LLMUncertaintyConfig'"
    )
    with patch.object(
        inference_handler.dynamo_store,
        "load_config",
        side_effect=ValueError(error_message),
    ):
        inference_handler._reset_for_testing()
        response = inference_handler.lambda_handler(_EVENT, context=None)
    assert response["statusCode"] == 503
    assert "conformal_threshold" in response["body"]["error"]


def test_502_when_sqs_send_message_fails_twice() -> None:
    """Two consecutive ``send_message`` failures yield a 502 carrying the
    routing context, and ``send_message`` is called exactly twice
    (single retry, no infinite loop)."""
    config = dict(_CONFIG)
    config["conformal_threshold"] = "0.99"  # forces multi-element prediction set
    logprobs = {"0": -0.1, "1": -0.2, "2": -0.3, "3": -0.4}

    with _stubbed_handler(config, logprobs) as sqs_client:
        sqs_client.send_message.side_effect = ClientError(
            error_response={
                "Error": {"Code": "InternalError", "Message": "transient SQS failure"},
            },
            operation_name="SendMessage",
        )
        response = inference_handler.lambda_handler(_EVENT, context=None)

    assert response["statusCode"] == 502
    body = response["body"]
    assert body["routing_decision"] == "human_review"
    assert body["human_review_destination_arn"] == _CONFIG["human_review_destination_arn"]
    assert "transient SQS failure" in body["error"]
    assert sqs_client.send_message.call_count == 2
