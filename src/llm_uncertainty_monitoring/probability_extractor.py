# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Prefill-based softmax scoring against a SageMaker text-generation
endpoint.

Given a multiple-choice question and ``K`` candidate answer tokens, this
module submits ``K`` requests of the form ``prompt + " " + t_k`` to the
endpoint with ``max_new_tokens = 1``, ``details = True``, and
``decoder_input_details = True``, then reads the log-probability of the
last prefill token as the score for that answer. A softmax over those
``K`` log-probabilities yields the answer-probability vector consumed by
the rest of the pipeline (calibration, inference, batch evaluation).

The module is K-agnostic — ``K`` is always ``len(answer_tokens)`` — and
contains no model-specific logic. The default prompt template can be
overridden via the ``prompt_builder`` argument of
:func:`build_prompt_and_probs`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from llm_uncertainty_monitoring.conformal import SUM_TOLERANCE

__all__ = [
    "DEFAULT_ANSWER_TOKENS",
    "PromptBuilder",
    "default_prompt_builder",
    "get_answer_probs",
    "build_prompt_and_probs",
    "Boto3Predictor",
]


DEFAULT_ANSWER_TOKENS: tuple[str, ...] = ("0", "1", "2", "3")
"""Default K=4 answer tokens for MMLU-style multiple-choice questions
(kept as a default, not hard-coded into the logic)."""


PromptBuilder = Callable[[str, Sequence[str]], str]
"""Callable that renders ``(question, choices)`` into a single prompt
string. See :func:`default_prompt_builder` for the reference template."""


class PredictorLike(Protocol):
    """Protocol for anything that can predict — sagemaker.Predictor or Boto3Predictor."""

    def predict(self, data: Any) -> Any: ...


class Boto3Predictor:
    """Lightweight predictor that uses boto3 sagemaker-runtime directly.

    Drop-in replacement for ``sagemaker.Predictor`` in Lambda environments
    where the full sagemaker SDK is not available. Only requires boto3
    (pre-installed in the Lambda runtime).
    """

    def __init__(self, endpoint_name: str, region_name: str | None = None) -> None:
        import boto3

        self._client = boto3.client("sagemaker-runtime", region_name=region_name)
        self._endpoint_name = endpoint_name

    def predict(self, data: Any) -> dict[str, Any]:
        """Invoke the endpoint and return the parsed JSON response."""
        response = self._client.invoke_endpoint(
            EndpointName=self._endpoint_name,
            ContentType="application/json",
            Body=json.dumps(data).encode("utf-8"),
        )
        return dict(json.loads(response["Body"].read()))


def default_prompt_builder(question: str, choices: Sequence[str]) -> str:
    """Numbered-choice prompt.

    The produced template is::

        Question: <question>
        Choices:
        0. <choices[0]>
        1. <choices[1]>
        ...
        Answer:

    """
    numbered_choices = "\n".join(f"{i}. {choice}" for i, choice in enumerate(choices))
    return f"Question: {question}\nChoices:\n{numbered_choices}\nAnswer:"


def get_answer_probs(
    predictor: PredictorLike,
    prompt: str,
    answer_tokens: Sequence[str] = DEFAULT_ANSWER_TOKENS,
) -> NDArray[np.float64]:
    """Submit one prefill request per ``answer_token`` and return the
    softmax probability vector over the ``K`` tokens.

    For each token ``t_k`` the payload is::

        {
            "inputs": prompt + " " + t_k,
            "parameters": {
                "max_new_tokens": 1,
                "details": True,
                "decoder_input_details": True,
            },
        }
    decoder_input_details: True — returns logprobs for the prompt tokens (the prefill pass).

    The score for ``t_k`` is taken from
    ``response["details"]["prefill"][-1]["logprob"]``.

    Args:
        predictor: any object with a ``predict(payload)`` method that
            returns a parsed JSON dict. Accepts ``sagemaker.Predictor``
            or :class:`Boto3Predictor` (for Lambda environments).
        prompt: the rendered question prompt (no trailing space).
        answer_tokens: ``K`` candidate answer tokens. ``K = len(...)`` is
            inferred — there is no ``K = 4``-specific branch.

    Returns:
        A length-``K`` ``np.ndarray`` of probabilities summing to ``1``
        within :data:`~llm_uncertainty_monitoring.conformal.SUM_TOLERANCE`.

    Raises:
        ValueError: if ``answer_tokens`` is empty, or if any response is
            missing a non-empty ``details.prefill`` array. The
            message includes the offending token and a ``repr`` of the
            request payload.
    """
    if not answer_tokens:
        raise ValueError("answer_tokens must contain at least one token, got empty sequence")

    logprobs: list[float] = []
    for token in answer_tokens:
        payload = {
            "inputs": prompt + " " + token,
            "parameters": {
                "max_new_tokens": 1,
                "details": True,
                "decoder_input_details": True,
            },
        }
        response = predictor.predict(payload)
        prefill = _extract_prefill(response)
        if not prefill:
            raise ValueError(
                "endpoint response is missing a non-empty 'details.prefill' array "
                f"for answer_token={token!r}; payload={payload!r}"
            )
        last_logprob = prefill[-1].get("logprob")
        logprobs.append(float(cast(Any, last_logprob)))

    probs = cast(NDArray[np.float64], softmax(np.asarray(logprobs, dtype=np.float64)))

    total = float(probs.sum())
    assert (
        abs(total - 1.0) <= SUM_TOLERANCE
    ), f"softmax output must sum to 1 within {SUM_TOLERANCE}, got sum={total!r}"
    return probs


def build_prompt_and_probs(
    predictor: PredictorLike,
    question: str,
    choices: Sequence[str],
    answer_tokens: Sequence[str] = DEFAULT_ANSWER_TOKENS,
    prompt_builder: PromptBuilder = default_prompt_builder,
) -> tuple[str, NDArray[np.float64]]:
    """Render the prompt with ``prompt_builder`` and score it with
    :func:`get_answer_probs`.

    Returns:
        A ``(prompt, probabilities)`` tuple. The prompt is returned so
        callers can persist it for reproducibility (e.g., the calibration
        diagnostics manifest).
    """
    prompt = prompt_builder(question, choices)
    probs = get_answer_probs(predictor, prompt, answer_tokens)
    return prompt, probs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_prefill(response: object) -> list[dict[str, object]]:
    """Return ``response["details"]["prefill"]`` as a list, tolerating
    both pre-parsed dict responses and missing keys.

    Returns an empty list when ``details`` or ``prefill`` is absent or is
    not a non-empty sequence; callers raise :class:`ValueError` in that
    case so the offending token and payload are reported uniformly.
    """
    if not isinstance(response, dict):
        return []
    details = response.get("details")
    if not isinstance(details, dict):
        return []
    prefill = details.get("prefill")
    if not isinstance(prefill, list) or not prefill:
        return []
    return cast(list[dict[str, object]], prefill)
