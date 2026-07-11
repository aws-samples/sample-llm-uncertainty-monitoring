# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Calibration aggregation logic.

Computes softmax probability vectors from logprobs, derives LAC scores,
and computes the conformal threshold ``q_hat``.

Used by ``scripts/run_calibration.py`` for the real-time calibration path.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.special import softmax

from llm_uncertainty_monitoring.conformal import compute_threshold

__all__ = [
    "compute_calibration",
    "read_all_probs_lines",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# S3 reading
# ---------------------------------------------------------------------------


def read_all_probs_lines(s3_client: Any, bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Read and parse all JSONL lines from objects under ``prefix``."""
    paginator = s3_client.get_paginator("list_objects_v2")
    lines: list[dict[str, Any]] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue  # skip "folder" markers
            resp = s3_client.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read().decode("utf-8")
            for line in body.strip().split("\n"):
                if line.strip():
                    lines.append(json.loads(line))

    return lines


# ---------------------------------------------------------------------------
# Logprob extraction
# ---------------------------------------------------------------------------


def _extract_logprob(record: dict[str, Any]) -> float:
    """Extract the prefill logprob from a Batch Transform output record.

    Batch Transform with ``join_source="Input"`` produces records where
    the model response is nested under ``SageMakerOutput``. The TGI
    response has ``details.prefill[-1].logprob``.
    """
    output = record.get("SageMakerOutput", record)
    if isinstance(output, str):
        output = json.loads(output)

    details = output.get("details", {})
    prefill = details.get("prefill", [])
    if not prefill:
        raise ValueError(
            f"missing details.prefill in record index={record.get('index')}, "
            f"k={record.get('k')}"
        )
    return float(prefill[-1]["logprob"])


# ---------------------------------------------------------------------------
# Calibration computation
# ---------------------------------------------------------------------------


def compute_calibration(
    lines: list[dict[str, Any]],
    target_coverage: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Group lines by index, compute LAC scores, and return q_hat + diagnostics.

    Args:
        lines: parsed JSONL records from Batch Transform output, each
            containing ``index``, ``k``, ``answer``, and the TGI response.
        target_coverage: target marginal coverage (e.g. 0.9).

    Returns:
        (q_hat, per_sample_diagnostics) where each diagnostic entry has
        ``index``, ``lac_score``, ``probs``, ``answer``.

    Raises:
        ValueError: if the calibration set is too small for the requested
            coverage, or if any record is missing prefill logprobs.
    """
    # Group by index
    by_index: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    for line in lines:
        index = int(line["index"])
        k = int(line["k"])
        answer = int(line["answer"])
        logprob = _extract_logprob(line)
        by_index[index].append((k, logprob, answer))

    # Compute per-sample LAC scores
    diagnostics: list[dict[str, Any]] = []
    lac_scores: list[float] = []

    for index in sorted(by_index.keys()):
        entries = sorted(by_index[index], key=lambda x: x[0])  # sort by k
        logprobs = np.array([entry[1] for entry in entries])
        answer = entries[0][2]  # same answer for all K entries of this index

        probs = softmax(logprobs).tolist()
        lac_score = 1.0 - probs[answer]
        lac_scores.append(lac_score)

        diagnostics.append(
            {
                "index": index,
                "lac_score": lac_score,
                "probs": probs,
                "answer": answer,
            }
        )

    # Compute threshold
    q_hat = compute_threshold(lac_scores, target_coverage)
    return q_hat, diagnostics
