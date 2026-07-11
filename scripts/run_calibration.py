# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Calibrate the conformal threshold (q_hat) via real-time endpoint scoring.

Steps:
1. Download ``cal.jsonl`` from S3.
2. For each calibration row, call the deployed endpoint via
   ``build_prompt_and_probs`` to get the probability vector.
3. Compute the LAC nonconformity score for each sample.
4. Compute ``q_hat`` via ``compute_threshold``.
5. Write ``q_hat`` + calibration metadata to DynamoDB.
6. Upload ``diagnostics.json`` to S3.

This approach uses the same real-time endpoint as ``prepare_data.py``
and the inference Lambda. It avoids SageMaker Batch Transform which has
a known incompatibility with JumpStart gated models (Batch Transform
requires ``ModelDataUrl`` but gated models like Llama 3 require
``ModelDataSource`` with ``AcceptEula=True``).

Example
-------
::

    python scripts/run_calibration.py \\
        --endpoint-name meta-textgeneration-llama-3-8b-2026-06-09-16-04-09 \\
        --cal-s3-uri s3://DOC-EXAMPLE-BUCKET/data/cal.jsonl \\
        --output-s3-uri s3://DOC-EXAMPLE-BUCKET/calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    DEFAULT_CONFIG_TABLE,
    load_config,
    put_config,
)
from llm_uncertainty_monitoring.conformal import LACScore, compute_threshold
from llm_uncertainty_monitoring.model_deployer import ModelDeployer
from llm_uncertainty_monitoring.probability_extractor import build_prompt_and_probs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate q_hat by scoring the calibration set against the "
            "deployed endpoint via real-time invocations."
        ),
    )
    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="Name of the deployed SageMaker real-time endpoint.",
    )
    parser.add_argument(
        "--cal-s3-uri",
        required=True,
        help="S3 URI of the calibration split (e.g. s3://DOC-EXAMPLE-BUCKET/data/cal.jsonl).",
    )
    parser.add_argument(
        "--output-s3-uri",
        required=True,
        help="S3 prefix for calibration outputs (diagnostics.json).",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=4,
        help="Number of answer choices K (default: %(default)s).",
    )
    parser.add_argument(
        "--config-id",
        default=DEFAULT_CONFIG_ID,
        help="DynamoDB Config partition key (default: %(default)s).",
    )
    parser.add_argument(
        "--config-table",
        default=DEFAULT_CONFIG_TABLE,
        help="DynamoDB table name (default: %(default)s).",
    )
    return parser


# ---------------------------------------------------------------------------
# S3 helpers
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Score cal.jsonl against the endpoint, compute q_hat, write to DDB + S3."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    s3_client = boto3.client("s3")

    # --- 1. Read target_coverage from DDB -------------------------------------
    try:
        config = load_config(config_id=args.config_id, table_name=args.config_table, strict=False)
        target_coverage = float(config["target_coverage"])
    except (ValueError, KeyError) as err:
        logger.error("cannot read target_coverage from DDB: %s", err)
        return 1
    logger.info("target_coverage=%.4f", target_coverage)

    # --- 2. Download cal.jsonl ------------------------------------------------
    cal_bucket, cal_key = _parse_s3_uri(args.cal_s3_uri)
    logger.info("downloading %s", args.cal_s3_uri)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_cal = Path(tmp_dir) / "cal.jsonl"
        s3_client.download_file(cal_bucket, cal_key, str(local_cal))

        cal_rows: list[dict[str, Any]] = []
        with local_cal.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cal_rows.append(json.loads(line))

    logger.info("loaded %d calibration rows", len(cal_rows))
    if not cal_rows:
        logger.error("cal.jsonl is empty — nothing to calibrate")
        return 1

    # --- 3. Score each row against the endpoint -------------------------------
    predictor = ModelDeployer().attach(args.endpoint_name)
    answer_tokens = tuple(str(i) for i in range(args.num_choices))
    lac = LACScore()

    diagnostics: list[dict[str, Any]] = []
    lac_scores: list[float] = []

    try:
        from tqdm import tqdm

        iterator = tqdm(cal_rows, desc="calibrating", total=len(cal_rows))
    except ImportError:
        iterator = cal_rows

    for row in iterator:
        question = row["question"]
        choices = row["choices"]
        answer = int(row["answer"])

        _, probs = build_prompt_and_probs(predictor, question, choices, answer_tokens=answer_tokens)
        probs_list = [float(p) for p in probs]
        score = lac.score(probs_list, answer)
        lac_scores.append(score)

        diagnostics.append(
            {
                "index": row.get("index"),
                "lac_score": score,
                "probs": probs_list,
                "answer": answer,
            }
        )

    # --- 4. Compute q_hat -----------------------------------------------------
    try:
        q_hat = compute_threshold(lac_scores, target_coverage)
    except ValueError as err:
        logger.error("calibration computation failed: %s", err)
        return 1

    n_samples = len(lac_scores)
    alpha = 1.0 - target_coverage
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info("q_hat=%.6f, N=%d, alpha=%.4f", q_hat, n_samples, alpha)

    # --- 5. Write q_hat to DynamoDB -------------------------------------------
    output_bucket, output_prefix = _parse_s3_uri(args.output_s3_uri)
    calibration_s3_uri = f"s3://{output_bucket}/{output_prefix}/"

    put_config(
        values={
            "conformal_threshold": str(q_hat),
            "target_coverage": str(target_coverage),
            "calibration_set_size": str(n_samples),
            "calibrated_at": timestamp,
            "calibration_job_name": f"realtime-cal-{timestamp[:10]}",
            "calibration_s3_uri": calibration_s3_uri,
        },
        config_id=args.config_id,
        table_name=args.config_table,
    )
    logger.info("wrote q_hat=%.6f to DDB %s.config_id=%s", q_hat, args.config_table, args.config_id)

    # --- 6. Upload diagnostics.json to S3 -------------------------------------
    diag_key = f"{output_prefix}/diagnostics.json"
    diagnostics_payload = {
        "q_hat": q_hat,
        "target_coverage": target_coverage,
        "alpha": alpha,
        "N": n_samples,
        "timestamp": timestamp,
        "calibration_s3_uri": calibration_s3_uri,
        "samples": diagnostics,
    }
    s3_client.put_object(
        Bucket=output_bucket,
        Key=diag_key,
        Body=json.dumps(diagnostics_payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("wrote diagnostics to s3://%s/%s", output_bucket, diag_key)

    # --- Summary --------------------------------------------------------------
    print()
    print("=" * 72)
    print("Calibration complete!")
    print(f"  q_hat:           {q_hat:.6f}")
    print(f"  N (samples):     {n_samples}")
    print(f"  target_coverage: {target_coverage}")
    print(f"  Diagnostics:     s3://{output_bucket}/{diag_key}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
