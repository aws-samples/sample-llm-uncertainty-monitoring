# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Data-quality gate for MMLU splits before calibration.

Downloads ``cal.jsonl``, ``test.jsonl``, and optionally ``difficult.jsonl``
from S3, runs PyDeequ checks via :func:`data_quality.validate_mmlu_dataset`
on each, and asserts pairwise-disjoint indices via
:func:`data_quality.validate_split_result`.

Exit codes:
- 0: all checks pass.
- 1: one or more checks fail (failures printed as JSON to stdout).

Run this as a manual gate before launching Batch Transform. The
calibration script re-checks ``cal.jsonl`` defensively before computing
``q_hat``.

Example
-------
::

    python scripts/validate_splits.py \\
        --cal-s3-uri s3://DOC-EXAMPLE-BUCKET/data/cal.jsonl \\
        --test-s3-uri s3://DOC-EXAMPLE-BUCKET/data/test.jsonl \\
        --difficult-s3-uri s3://DOC-EXAMPLE-BUCKET/data/difficult.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from datasets import Dataset

from llm_uncertainty_monitoring.data_loader import SplitResult
from llm_uncertainty_monitoring.data_quality import (
    DataQualityError,
    validate_split_result,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PyDeequ data-quality checks on MMLU splits stored in S3. "
            "Exits non-zero with a JSON dump of failures if any check fails."
        ),
    )
    parser.add_argument(
        "--cal-s3-uri",
        required=True,
        help="S3 URI of the calibration split (e.g. s3://DOC-EXAMPLE-BUCKET/data/cal.jsonl).",
    )
    parser.add_argument(
        "--test-s3-uri",
        required=True,
        help="S3 URI of the test split (e.g. s3://DOC-EXAMPLE-BUCKET/data/test.jsonl).",
    )
    parser.add_argument(
        "--difficult-s3-uri",
        default="",
        help="S3 URI of the difficult split (optional). If omitted, only cal and test are checked.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=4,
        help="Number of answer choices K (default: %(default)s).",
    )
    return parser


# ---------------------------------------------------------------------------
# S3 download helpers
# ---------------------------------------------------------------------------


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` into ``(bucket, key)``."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"malformed S3 URI: {uri!r}")
    return bucket, key


def _download_jsonl_as_dataset(s3_client: Any, s3_uri: str, tmp_dir: Path) -> Dataset:
    """Download a JSONL file from S3 and load it as a HuggingFace Dataset."""
    bucket, key = _parse_s3_uri(s3_uri)
    filename = key.rsplit("/", 1)[-1] if "/" in key else key
    local_path = tmp_dir / filename

    logger.info("downloading %s to %s", s3_uri, local_path)
    s3_client.download_file(bucket, key, str(local_path))

    ds = Dataset.from_json(str(local_path))
    logger.info("loaded %d rows from %s", len(ds), s3_uri)
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Download splits from S3, validate, and exit with appropriate code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    s3_client = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Download splits
        try:
            cal_ds = _download_jsonl_as_dataset(s3_client, args.cal_s3_uri, tmp_path)
            test_ds = _download_jsonl_as_dataset(s3_client, args.test_s3_uri, tmp_path)

            if args.difficult_s3_uri:
                difficult_ds = _download_jsonl_as_dataset(
                    s3_client, args.difficult_s3_uri, tmp_path
                )
            else:
                # Create an empty dataset with the expected schema so
                # validate_split_result can still check disjointness.
                difficult_ds = Dataset.from_dict(
                    {"index": [], "question": [], "choices": [], "answer": [], "subject": []}
                )
                logger.info("no --difficult-s3-uri provided; using empty difficult split")
        except Exception as err:  # noqa: BLE001
            logger.error("failed to download splits from S3: %s", err)
            return 1

        # Run validation
        splits = SplitResult(
            difficult=difficult_ds,
            calibration=cal_ds,
            test=test_ds,
        )

        try:
            reports = validate_split_result(splits, num_choices=args.num_choices)
        except DataQualityError as err:
            logger.error("data-quality validation FAILED")
            print(json.dumps({"passed": False, "failures": err.report.failures}, indent=2))
            return 1

    # Success
    logger.info("all data-quality checks passed")
    summary = {
        "passed": True,
        "splits": {
            name: {"passed": report.passed, "failures": report.failures}
            for name, report in reports.items()
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
