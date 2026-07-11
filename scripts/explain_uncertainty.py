# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CLI for uncertainty explainability word-cloud generation.

Downloads a batch JSON from S3 (produced by ``run_batch_evaluation.py``),
extracts the records, and renders a side-by-side comparison word cloud
showing terms disproportionately associated with certain vs uncertain
predictions using differential frequency ratios.

Output:
    - A comparison PNG (certain vs uncertain word clouds side by side)
    - Top-N differential terms printed to console
    - Optional upload to S3

Example
-------
::

    python scripts/explain_uncertainty.py \\
        --batch-json-s3-uri s3://DOC-EXAMPLE-BUCKET/batch/batch-20260601-120000.json \\
        --top-n 10 \\
        --output-local-path ./wordcloud_comparison.png \\
        --output-s3-uri s3://DOC-EXAMPLE-BUCKET/wordclouds/comparison.png
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

from llm_uncertainty_monitoring.uncertainty_explainer import compare_certain_uncertain

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a differential word cloud comparing certain vs uncertain "
            "predictions from a batch JSON."
        ),
    )
    parser.add_argument(
        "--batch-json-s3-uri",
        required=True,
        help="S3 URI of the batch JSON (e.g. s3://DOC-EXAMPLE-BUCKET/batch/batch-20260601-120000.json).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top differential terms to display per group (default: %(default)s).",
    )
    parser.add_argument(
        "--output-local-path",
        default="./wordcloud_comparison.png",
        help="Local path for the rendered comparison PNG (default: %(default)s).",
    )
    parser.add_argument(
        "--output-s3-uri",
        default="",
        help="Optional S3 URI to upload the PNG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Download batch JSON, render comparison word cloud, optionally upload to S3."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    s3_client = boto3.client("s3")

    # Download batch JSON
    parsed = urlparse(args.batch_json_s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    logger.info("downloading batch JSON from %s", args.batch_json_s3_uri)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    s3_client.download_file(bucket, key, str(tmp_path))

    with tmp_path.open("r", encoding="utf-8") as f:
        batch_data: dict[str, Any] = json.load(f)
    tmp_path.unlink(missing_ok=True)

    records = batch_data.get("records", [])
    logger.info("loaded %d records from batch JSON", len(records))

    if not records:
        logger.error("batch JSON contains no records")
        return 1

    # Render comparison word cloud
    try:
        result = compare_certain_uncertain(
            records=records,
            top_n=args.top_n,
            output_local_path=args.output_local_path,
            output_s3_uri=args.output_s3_uri or None,
        )
    except ValueError as err:
        logger.error("explainer failed: %s", err)
        return 1

    # Print differential terms
    print(f"\nComparison word cloud saved to: {result['output_path']}")
    print(f"Certain predictions: {result['n_certain']}")
    print(f"Uncertain predictions: {result['n_uncertain']}")
    print(f"\nTop {args.top_n} uncertain-associated terms (ratio = uncertain / total):")
    for term, ratio in result["top_uncertain_differential"]:
        print(f"  {term:20s} {ratio:.0%}")
    print(f"\nTop {args.top_n} certain-associated terms (ratio = certain / total):")
    for term, ratio in result["top_certain_differential"]:
        print(f"  {term:20s} {ratio:.0%}")

    if args.output_s3_uri:
        print(f"\nUploaded to: {args.output_s3_uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
