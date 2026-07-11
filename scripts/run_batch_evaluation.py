# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Periodic batch evaluation: replay labeled data through the inference path.

Downloads ``test.jsonl`` (and optionally ``difficult.jsonl``) from S3,
invokes the inference Lambda (or calls the handler in-process) for each
row, collects ``PredictionRecord``s, and publishes batch aggregates
(``AveragePredictionSetSize``, ``Coverage``, ``Top1Correct``, etc.) to
CloudWatch. The full record set + aggregates are persisted as a batch
JSON to S3 for downstream consumption (XAI word cloud, dashboards).

When ``--difficult-s3-uri`` is provided, the difficult set is appended
as the final batch so the ``AverageSetSize`` spike shows up at the tail
of the time-series — demonstrating the drift-detection signal.

Can be invoked:
- Ad-hoc from the terminal or notebook.
- As a Lambda handler (``lambda_handler``) triggered by EventBridge
  (task 13.2 wraps this as a scheduled Lambda).

Example
-------
::

    python scripts/run_batch_evaluation.py \\
        --input-s3-uri s3://DOC-EXAMPLE-BUCKET/data/test.jsonl \\
        --difficult-s3-uri s3://DOC-EXAMPLE-BUCKET/data/difficult.jsonl \\
        --s3-output-bucket DOC-EXAMPLE-BUCKET \\
        --s3-output-prefix batch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    DEFAULT_CONFIG_TABLE,
    load_config,
)
from llm_uncertainty_monitoring.metrics_publisher import MetricsPublisher, PredictionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay labeled data through the inference Lambda and publish "
            "batch aggregate metrics to CloudWatch + S3."
        ),
    )
    parser.add_argument(
        "--input-s3-uri",
        required=True,
        help="S3 URI of the test split (e.g. s3://DOC-EXAMPLE-BUCKET/data/test.jsonl).",
    )
    parser.add_argument(
        "--difficult-s3-uri",
        default="",
        help="S3 URI of the difficult split (optional). Appended as the final batch.",
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
    parser.add_argument(
        "--s3-output-bucket",
        default="",
        help="S3 bucket for batch JSON output. If empty, derived from input URI bucket.",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="batch",
        help="S3 key prefix for batch JSON output (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows per batch for metrics publishing (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Number of rows to skip from the start of the input file (default: %(default)s).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of rows to process (0 = all remaining, default: %(default)s).",
    )
    parser.add_argument(
        "--lambda-function-name",
        default="",
        help=(
            "Lambda function name to invoke remotely. If empty, calls "
            "inference_handler.lambda_handler in-process (no network call)."
        ),
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
    return bucket, key


def _download_jsonl(s3_client: Any, s3_uri: str) -> list[dict[str, Any]]:
    """Download a JSONL file from S3 and return parsed rows."""
    bucket, key = _parse_s3_uri(s3_uri)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    s3_client.download_file(bucket, key, str(tmp_path))
    rows: list[dict[str, Any]] = []
    with tmp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    tmp_path.unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------------------
# Inference invocation
# ---------------------------------------------------------------------------


def _invoke_lambda(
    lambda_client: Any,
    function_name: str,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Invoke the inference Lambda remotely and return the response body."""
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(event).encode("utf-8"),
    )
    payload: dict[str, Any] = json.loads(response["Payload"].read())
    return payload


def _invoke_in_process(event: dict[str, Any]) -> dict[str, Any]:
    """Call inference_handler.lambda_handler directly (no network)."""
    from llm_uncertainty_monitoring.inference_handler import lambda_handler

    result: dict[str, Any] = lambda_handler(event, context=None)
    return result


def _process_row(
    row: dict[str, Any],
    invoke_fn: Any,
) -> PredictionRecord | None:
    """Build an event from a JSONL row, invoke inference, return a PredictionRecord."""
    event: dict[str, Any] = {
        "question": row["question"],
        "choices": row["choices"],
    }
    if "answer" in row:
        event["ground_truth"] = int(row["answer"])

    start = time.perf_counter()
    response = invoke_fn(event)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if response.get("statusCode") != 200:
        logger.warning(
            "inference returned status %s for index %s: %s",
            response.get("statusCode"),
            row.get("index"),
            response.get("body", {}).get("error", "unknown"),
        )
        return None

    body = response["body"]
    ground_truth = event.get("ground_truth")
    prediction_set = body["prediction_set"]
    top1 = body["top1_prediction"]

    covered: int | None = None
    top1_correct: int | None = None
    if ground_truth is not None:
        covered = int(ground_truth in prediction_set)
        top1_correct = int(ground_truth == top1)

    return PredictionRecord(
        question=row["question"],
        choices=row["choices"],
        probabilities=body["probabilities"],
        prediction_set=prediction_set,
        top1_prediction=top1,
        set_size=body["set_size"],
        is_certain=body["is_certain"],
        conformal_threshold=body["conformal_threshold"],
        prediction_latency_ms=latency_ms,
        routing_decision=body["routing_decision"],
        ground_truth=ground_truth,
        covered=covered,
        top1_correct=top1_correct,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Download test data, replay through inference, publish batch metrics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    s3_client = boto3.client("s3")

    # --- 1. Read config for metrics publisher dimensions ----------------------
    try:
        config = load_config(config_id=args.config_id, table_name=args.config_table)
    except ValueError as err:
        logger.error("cannot read config from DDB: %s", err)
        return 1

    publisher = MetricsPublisher(
        namespace=config["cloudwatch_namespace"],
        dimensions={
            "EndpointName": config["endpoint_name"],
            "ModelId": config["model_id"],
            "TargetCoverage": config["target_coverage"],
        },
    )

    # --- 2. Download test data ------------------------------------------------
    logger.info("downloading test data from %s", args.input_s3_uri)
    rows = _download_jsonl(s3_client, args.input_s3_uri)
    logger.info("loaded %d test rows", len(rows))

    if args.difficult_s3_uri:
        logger.info("downloading difficult data from %s", args.difficult_s3_uri)
        difficult_rows = _download_jsonl(s3_client, args.difficult_s3_uri)
        logger.info("loaded %d difficult rows (appended as final batch)", len(difficult_rows))
    else:
        difficult_rows = []

    # --- 3. Set up invocation function ----------------------------------------
    lambda_client = None
    if args.lambda_function_name:
        lambda_client = boto3.client("lambda")
        invoke_fn = lambda event: _invoke_lambda(  # noqa: E731
            lambda_client, args.lambda_function_name, event
        )
        logger.info("invoking Lambda: %s", args.lambda_function_name)
    else:
        invoke_fn = _invoke_in_process
        logger.info("invoking inference_handler in-process (no Lambda call)")

    # --- 4. Process rows in batches -------------------------------------------
    all_rows = rows + difficult_rows

    # Apply skip/max filtering
    if args.skip_rows > 0:
        all_rows = all_rows[args.skip_rows :]
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]

    all_records: list[PredictionRecord] = []
    batch_size = args.batch_size
    total = len(all_rows)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_rows = all_rows[batch_start:batch_end]

        logger.info(
            "processing batch %d–%d of %d",
            batch_start + 1,
            batch_end,
            total,
        )

        batch_records: list[PredictionRecord] = []
        for row in batch_rows:
            record = _process_row(row, invoke_fn)
            if record is not None:
                batch_records.append(record)

        all_records.extend(batch_records)

    logger.info("processed %d/%d rows successfully", len(all_records), total)

    if not all_records:
        logger.error("no successful predictions — nothing to publish")
        return 1

    # --- 5. Publish batch aggregates ------------------------------------------
    output_bucket = args.s3_output_bucket
    if not output_bucket:
        output_bucket, _ = _parse_s3_uri(args.input_s3_uri)

    run_id = datetime.now(timezone.utc).strftime("batch-%Y%m%d-%H%M%S")
    s3_uri = f"s3://{output_bucket}/{args.s3_output_prefix}/{run_id}.json"

    logger.info("publishing batch aggregates to CloudWatch and %s", s3_uri)
    publisher.publish_batch(all_records, s3_uri=s3_uri, run_id=run_id)

    # --- 6. Summary -----------------------------------------------------------
    n_certain = sum(1 for r in all_records if r.is_certain)
    n_uncertain = len(all_records) - n_certain
    avg_set_size = float(np.mean([r.set_size for r in all_records]))

    print()
    print("=" * 72)
    print("Batch evaluation complete!")
    print(f"  run_id:              {run_id}")
    print(f"  total predictions:   {len(all_records)}")
    print(f"  certain:             {n_certain}")
    print(f"  uncertain:           {n_uncertain}")
    print(f"  avg set size:        {avg_set_size:.3f}")
    print(f"  batch JSON:          {s3_uri}")
    print("=" * 72)

    return 0


# ---------------------------------------------------------------------------
# Lambda handler (for EventBridge invocation via task 13.2)
# ---------------------------------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Entry point when invoked as a Lambda by EventBridge.

    Expects ``event`` to contain the CLI-equivalent arguments as keys:
    ``input_s3_uri``, ``difficult_s3_uri``, ``config_id``, etc.
    """
    argv = []
    if event.get("input_s3_uri"):
        argv.extend(["--input-s3-uri", event["input_s3_uri"]])
    if event.get("difficult_s3_uri"):
        argv.extend(["--difficult-s3-uri", event["difficult_s3_uri"]])
    if event.get("config_id"):
        argv.extend(["--config-id", event["config_id"]])
    if event.get("config_table"):
        argv.extend(["--config-table", event["config_table"]])
    if event.get("s3_output_bucket"):
        argv.extend(["--s3-output-bucket", event["s3_output_bucket"]])
    if event.get("s3_output_prefix"):
        argv.extend(["--s3-output-prefix", event["s3_output_prefix"]])
    if event.get("batch_size"):
        argv.extend(["--batch-size", str(event["batch_size"])])
    if event.get("lambda_function_name"):
        argv.extend(["--lambda-function-name", event["lambda_function_name"]])

    exit_code = main(argv)
    return {"statusCode": 200 if exit_code == 0 else 500, "exit_code": exit_code}


if __name__ == "__main__":
    sys.exit(main())
