# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for ``llm_uncertainty_monitoring.probability_extractor``.

We do not hit the network here: the SageMaker ``Predictor`` is replaced
by a ``MagicMock`` whose ``predict`` method returns a pre-canned dict
shaped like the JumpStart text-generation response. The tests cover:

- the softmax output sums to 1 within tolerance,
- ``ValueError`` is raised when ``details.prefill`` is missing or empty,
  and the message includes the offending token plus a ``repr`` of the
  payload (Req 3.4),
- the K-agnostic behaviour: both ``K = 4`` and ``K = 2`` are accepted
  without any code path branching on the value of ``K`` (Req 3.6).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_uncertainty_monitoring.conformal import SUM_TOLERANCE
from llm_uncertainty_monitoring.probability_extractor import (
    DEFAULT_ANSWER_TOKENS,
    build_prompt_and_probs,
    default_prompt_builder,
    get_answer_probs,
)


def _make_predictor(logprobs_by_token: dict[str, float]) -> MagicMock:
    """Build a mock ``Predictor`` whose ``predict`` returns a JumpStart-
    shaped response with the requested ``logprob`` for the trailing
    prefill token.

    The mock inspects ``payload["inputs"]`` to recover which answer token
    was appended (last whitespace-separated chunk), so a single mock
    serves every token in a single ``get_answer_probs`` call.
    """
    predictor = MagicMock()

    def _predict(payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        inputs = payload["inputs"]
        token = inputs.rsplit(" ", 1)[-1]
        logprob = logprobs_by_token[token]
        return {
            "details": {
                "prefill": [
                    {"id": 1, "text": "ignored", "logprob": -10.0},
                    {"id": 2, "text": token, "logprob": logprob},
                ],
                "tokens": [],
            },
            "generated_text": token,
        }

    predictor.predict.side_effect = _predict
    return predictor


# ---------------------------------------------------------------------------
# default_prompt_builder
# ---------------------------------------------------------------------------


def test_default_prompt_builder_renders_numbered_choices() -> None:
    prompt = default_prompt_builder("What is 1 + 1?", ["0", "1", "2", "3"])
    assert prompt == (
        "Question: What is 1 + 1?\n" "Choices:\n" "0. 0\n" "1. 1\n" "2. 2\n" "3. 3\n" "Answer:"
    )


# ---------------------------------------------------------------------------
# get_answer_probs — softmax invariants and K-agnosticism
# ---------------------------------------------------------------------------


def test_get_answer_probs_returns_softmax_summing_to_one_for_k4() -> None:
    """K = 4: probabilities are a length-4 ``np.ndarray`` summing to 1
    within tolerance, and ordered by the supplied logprobs."""
    logprobs = {"0": -1.0, "1": -2.0, "2": -3.0, "3": -0.5}
    predictor = _make_predictor(logprobs)

    probs = get_answer_probs(predictor, prompt="Q?\nAnswer:")

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (4,)
    assert abs(float(probs.sum()) - 1.0) <= SUM_TOLERANCE

    # Largest logprob (token "3", -0.5) yields the largest probability.
    assert int(np.argmax(probs)) == 3
    # K predictor calls were made, one per answer token.
    assert predictor.predict.call_count == len(DEFAULT_ANSWER_TOKENS)


def test_get_answer_probs_is_k_agnostic_for_k2() -> None:
    """K = 2: same invariants hold with a custom 2-token vocabulary."""
    answer_tokens = ("yes", "no")
    logprobs = {"yes": -0.2, "no": -1.5}
    predictor = _make_predictor(logprobs)

    probs = get_answer_probs(
        predictor, prompt="Is the sky blue?\nAnswer:", answer_tokens=answer_tokens
    )

    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) <= SUM_TOLERANCE
    # "yes" has the larger logprob -> larger probability.
    assert probs[0] > probs[1]
    assert predictor.predict.call_count == 2


def test_get_answer_probs_rejects_empty_answer_tokens() -> None:
    predictor = MagicMock()
    with pytest.raises(ValueError, match="answer_tokens"):
        get_answer_probs(predictor, prompt="Q?\nAnswer:", answer_tokens=[])
    predictor.predict.assert_not_called()


# ---------------------------------------------------------------------------
# get_answer_probs — error paths
# ---------------------------------------------------------------------------


def test_get_answer_probs_raises_on_missing_prefill() -> None:
    """``details.prefill`` missing entirely -> ValueError naming the
    offending token and including a ``repr`` of the payload."""
    predictor = MagicMock()
    predictor.predict.return_value = {"details": {}, "generated_text": "0"}

    with pytest.raises(ValueError) as exc:
        get_answer_probs(predictor, prompt="Q?\nAnswer:", answer_tokens=("0", "1"))

    msg = str(exc.value)
    assert "answer_token='0'" in msg
    assert "payload=" in msg
    assert "'inputs': 'Q?\\nAnswer: 0'" in msg


def test_get_answer_probs_raises_on_empty_prefill() -> None:
    """``details.prefill`` present but empty -> same ValueError shape."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "details": {"prefill": [], "tokens": []},
        "generated_text": "0",
    }

    with pytest.raises(ValueError) as exc:
        get_answer_probs(predictor, prompt="Q?\nAnswer:", answer_tokens=("0",))

    msg = str(exc.value)
    assert "answer_token='0'" in msg
    assert "payload=" in msg


# ---------------------------------------------------------------------------
# build_prompt_and_probs
# ---------------------------------------------------------------------------


def test_build_prompt_and_probs_uses_prompt_builder_and_returns_tuple() -> None:
    logprobs = {"A": -0.1, "B": -2.0}
    predictor = _make_predictor(logprobs)

    captured: dict[str, object] = {}

    def custom_builder(question: str, choices) -> str:  # noqa: ANN001 — test helper
        captured["question"] = question
        captured["choices"] = list(choices)
        return f"<<{question}|{'/'.join(choices)}>>"

    prompt, probs = build_prompt_and_probs(
        predictor,
        question="Pick one",
        choices=["alpha", "beta"],
        answer_tokens=("A", "B"),
        prompt_builder=custom_builder,
    )

    assert prompt == "<<Pick one|alpha/beta>>"
    assert captured == {"question": "Pick one", "choices": ["alpha", "beta"]}
    assert probs.shape == (2,)
    assert abs(float(probs.sum()) - 1.0) <= SUM_TOLERANCE
