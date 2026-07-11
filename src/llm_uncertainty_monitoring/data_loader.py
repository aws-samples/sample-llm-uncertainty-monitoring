# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""MMLU loading, difficult-question extraction, calibration/test splitting,
and S3 persistence.

This module is built around the HuggingFace ``datasets.Dataset`` abstraction.
``load_mmlu`` is the named, MMLU-specific entry point that the blog and demo
notebook use, but the downstream helpers (``extract_difficult``,
``split_remaining``, ``persist_splits_to_s3``) are generic: they work on any
``Dataset`` whose rows look like a multiple-choice classification example.

Expected per-row schema for the generic helpers:

- ``index`` (int): unique identifier preserved across splits.
- ``question`` (str): the question text.
- ``choices`` (list[str]): the K answer choices.
- ``answer`` (int): 0-based index of the correct choice.
- *(optional)* ``subject`` or any other categorical column used as
  ``group_column`` for difficult-question extraction.

Adopters who want to bring a different dataset can replace the call to
``load_mmlu`` with any other ``Dataset`` source, for example::

    from datasets import load_dataset

    ds = load_dataset("json", data_files="my_data.jsonl", split="train")
    ds = ds.add_column("index", list(range(len(ds))))

…and feed it into the same downstream pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

import boto3
from datasets import Dataset, concatenate_datasets, load_dataset

__all__ = [
    "SplitResult",
    "load_mmlu",
    "extract_difficult",
    "split_remaining",
    "persist_splits_to_s3",
]


@dataclass(frozen=True)
class SplitResult:
    """Three pairwise-disjoint datasets produced by the splitting pipeline.

    The ``index`` columns of the three datasets form a partition of the
    ``index`` column of the input (or a subset of it, when the caller
    intentionally drops some rows).
    """

    difficult: Dataset
    calibration: Dataset
    test: Dataset


# ---------------------------------------------------------------------------
# MMLU loader (the only dataset-specific function in this module)
# ---------------------------------------------------------------------------


def load_mmlu(subjects: Iterable[str]) -> Dataset:
    """Load the ``test`` split of the requested MMLU subjects from the
    ``cais/mmlu`` HuggingFace dataset and concatenate them.

    Adds a ``subject`` column with the source subject name and a global
    ``index`` column counted across the concatenated dataset.

    Args:
        subjects: One or more MMLU subject names, e.g.
            ``["econometrics", "marketing", "management", "sociology"]``.

    Raises:
        ValueError: if ``subjects`` is empty, or if any subject name is not
            available in ``cais/mmlu``. The offending name is included in
            the error message.
    """
    subjects = list(subjects)
    if not subjects:
        raise ValueError("subjects must contain at least one MMLU subject")

    parts: list[Dataset] = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except (ValueError, FileNotFoundError) as exc:
            raise ValueError(
                f"unknown MMLU subject {subject!r} (cais/mmlu config not found)"
            ) from exc
        if "subject" not in ds.column_names:
            ds = ds.add_column("subject", [subject] * len(ds))
        parts.append(ds)

    concatenated = concatenate_datasets(parts)
    if "index" in concatenated.column_names:
        concatenated = concatenated.remove_columns("index")
    return concatenated.add_column("index", list(range(len(concatenated))))


# ---------------------------------------------------------------------------
# Generic helpers — work on any classification-shaped Dataset
# ---------------------------------------------------------------------------


def extract_difficult(
    dataset: Dataset,
    probs_by_index: dict[int, list[float]],
    n_per_group: int,
    group_column: str | None = "subject",
) -> Dataset:
    """Select the ``n_per_group`` rows per group with the lowest maximum
    answer probability — those are the "difficult" rows the model is
    least confident about.

    The function uses an externally-computed probability dictionary so the
    data loader stays independent of any model or endpoint. The caller
    runs the model once over every row of ``dataset``, builds
    ``probs_by_index = {row.index: probability_vector}``, and feeds it in.

    Args:
        dataset: any ``Dataset`` carrying an ``index`` column.
        probs_by_index: mapping from row ``index`` to the model's
            probability vector for that row.
        n_per_group: number of difficult rows to select per group. Capped
            at the group size when smaller.
        group_column: column to group by (``"subject"`` for MMLU). Pass
            ``None`` to treat the whole dataset as one group.

    Raises:
        ValueError: if ``dataset`` lacks an ``index`` column,
            ``n_per_group`` is negative, ``probs_by_index`` is missing an
            entry for any row, or ``group_column`` is set but absent from
            ``dataset``.
    """
    if "index" not in dataset.column_names:
        raise ValueError("dataset must have an 'index' column")
    if n_per_group < 0:
        raise ValueError(f"n_per_group must be non-negative, got {n_per_group!r}")
    if group_column is not None and group_column not in dataset.column_names:
        raise ValueError(f"group_column {group_column!r} not in dataset columns")

    indices_in_dataset: list[int] = list(dataset["index"])
    missing = [i for i in indices_in_dataset if i not in probs_by_index]
    if missing:
        raise ValueError(
            f"probs_by_index missing entries for indices {missing[:5]!r}"
            + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
        )

    # Compute "difficulty" as 1 - max(probs); larger = more difficult.
    difficulty_by_index: dict[int, float] = {
        i: 1.0 - max(probs_by_index[i]) for i in indices_in_dataset
    }

    # Group rows by group_column (or a single group if None).
    groups: dict[object, list[int]] = {}
    if group_column is None:
        groups[None] = indices_in_dataset
    else:
        for row_index, row_group in zip(indices_in_dataset, dataset[group_column], strict=True):
            groups.setdefault(row_group, []).append(row_index)

    # For each group, take the top n_per_group most-difficult rows.
    selected_indices: set[int] = set()
    for group_indices in groups.values():
        group_indices_sorted = sorted(
            group_indices, key=lambda i: difficulty_by_index[i], reverse=True
        )
        selected_indices.update(group_indices_sorted[:n_per_group])

    return dataset.filter(lambda row: row["index"] in selected_indices)


def split_remaining(
    dataset: Dataset,
    difficult: Dataset,
    calibration_fraction: float,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Split the rows of ``dataset`` that are *not* in ``difficult`` into a
    (calibration, test) pair.

    Both returned datasets carry their original ``index`` column, so
    downstream code can verify the three sets are pairwise disjoint.

    Args:
        dataset: full source dataset with an ``index`` column.
        difficult: the difficult subset returned by :func:`extract_difficult`.
        calibration_fraction: fraction of the *remaining* (non-difficult)
            rows to assign to the calibration split. Must lie in
            ``[0.1, 0.9]``.
        seed: random seed for the shuffle. Default ``0`` so the split is
            reproducible across runs.

    Raises:
        ValueError: if either dataset lacks an ``index`` column, or
            ``calibration_fraction`` is outside ``[0.1, 0.9]``.
    """
    if "index" not in dataset.column_names:
        raise ValueError("dataset must have an 'index' column")
    if "index" not in difficult.column_names:
        raise ValueError("difficult must have an 'index' column")
    if not (0.1 <= calibration_fraction <= 0.9):
        raise ValueError(
            "calibration_fraction must be in [0.1, 0.9], "
            f"got calibration_fraction={calibration_fraction!r}"
        )

    difficult_indices = set(difficult["index"])
    remaining = dataset.filter(lambda row: row["index"] not in difficult_indices)

    if len(remaining) == 0:
        empty = remaining
        return empty, empty

    shuffled = remaining.shuffle(seed=seed)
    calibration_size = round(calibration_fraction * len(shuffled))
    calibration = (
        shuffled.select(range(calibration_size)) if calibration_size > 0 else shuffled.select([])
    )
    test = (
        shuffled.select(range(calibration_size, len(shuffled)))
        if calibration_size < len(shuffled)
        else shuffled.select([])
    )
    return calibration, test


# ---------------------------------------------------------------------------
# S3 persistence
# ---------------------------------------------------------------------------


def persist_splits_to_s3(
    splits: SplitResult,
    s3_uri_prefix: str,
    s3_client: object | None = None,
) -> dict[str, str]:
    """Write the three splits as JSONL files under ``s3_uri_prefix`` and
    return a dict of the resulting URIs.

    Args:
        splits: the three datasets to persist.
        s3_uri_prefix: e.g. ``"s3://DOC-EXAMPLE-BUCKET/mmlu/"``. A trailing slash
            is added if missing.
        s3_client: optional pre-configured boto3 S3 client (used by tests).
            If ``None``, a default client is created.

    Returns:
        A dict with keys ``"difficult"``, ``"calibration"``, ``"test"``
        and values pointing to the S3 URIs of the written files.

    Raises:
        ValueError: if ``s3_uri_prefix`` is not an ``s3://`` URI.
    """
    parsed = urlparse(s3_uri_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(
            f"s3_uri_prefix must be of the form 's3://DOC-EXAMPLE-BUCKET/prefix/', "
            f"got {s3_uri_prefix!r}"
        )
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    client = s3_client if s3_client is not None else boto3.client("s3")

    file_names = {
        "difficult": "difficult.jsonl",
        "calibration": "cal.jsonl",
        "test": "test.jsonl",
    }
    datasets_by_name = {
        "difficult": splits.difficult,
        "calibration": splits.calibration,
        "test": splits.test,
    }

    uris: dict[str, str] = {}
    for split_name, ds in datasets_by_name.items():
        body = "\n".join(json.dumps(row) for row in ds).encode("utf-8")
        key = prefix + file_names[split_name]
        client.put_object(Bucket=bucket, Key=key, Body=body)
        uris[split_name] = f"s3://{bucket}/{key}"

    return uris
