# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""End-to-end smoke test for the LLM Uncertainty Monitoring system.

Runs the full pipeline against a live endpoint and validates results
against baseline values from the demo notebook.

Prerequisites (README "Deployment steps"):
- Steps 1-5 — base infrastructure: env vars, dependencies (``pip install``),
            deploy bucket, packaged Lambda, and the CloudFormation stack
            (DynamoDB table, data + human-review buckets, Lambdas, SQS, SNS,
            EventBridge). Everything below depends on this foundation.
- Step 6  — `deploy_model.py`: live SageMaker endpoint + DynamoDB config
            seeded with ``endpoint_name`` / ``model_id``.
- Step 7  — `prepare_data.py`: ``data/{cal,test,difficult}.jsonl`` in the
            data bucket.
- Step 9  — `run_calibration.py`: ``conformal_threshold`` (q_hat) in
            DynamoDB + ``calibration/diagnostics.json`` in S3.
- Step 11 — `run_batch_evaluation.py`: at least one ``batch/*.json`` object
            AND ``LLMUncertainty`` CloudWatch metrics. The CloudWatch check
            uses a 24h window, so this batch run must be recent (< 24h),
            otherwise that assertion fails even on a healthy setup.

Not required: step 6b (routing config — this test reads the inference
``routing_decision`` field but does not verify SQS delivery) and step 8
(`validate_splits.py`, an optional data-quality gate).

Exit codes:
- 0: all assertions pass (SMOKE OK)
- 1: one or more assertions failed

Example
-------
::

    python scripts/smoke_end_to_end.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    DEFAULT_CONFIG_TABLE,
    load_config,
    reset_cache,
)
from llm_uncertainty_monitoring.inference_handler import (
    _reset_for_testing,
    lambda_handler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_baselines() -> dict[str, Any]:
    """Load baseline values from smoke_baselines.json."""
    path = Path(__file__).parent / "smoke_baselines.json"
    with path.open("r", encoding="utf-8") as f:
        baselines: dict[str, Any] = json.load(f)
    return baselines


def _assert(condition: bool, message: str) -> None:
    """Assert with a descriptive message. Raises AssertionError on failure."""
    if not condition:
        raise AssertionError(f"SMOKE FAILED: {message}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def main() -> int:
    """Run end-to-end smoke assertions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    baselines = _load_baselines()
    tolerances = baselines["tolerances"]

    # --- 1. Verify DDB config is populated ------------------------------------
    logger.info("1. Checking DynamoDB config...")
    reset_cache()
    try:
        config = load_config(config_id=DEFAULT_CONFIG_ID, table_name=DEFAULT_CONFIG_TABLE)
    except ValueError as err:
        print(f"SMOKE FAILED: DDB config not found: {err}")
        return 1

    q_hat = float(config["conformal_threshold"])
    target_coverage = float(config["target_coverage"])
    endpoint_name = config["endpoint_name"]

    logger.info(
        "  q_hat=%.4f, target_coverage=%.2f, endpoint=%s",
        q_hat,
        target_coverage,
        endpoint_name,
    )

    # Check q_hat is in expected range
    q_hat_diff = abs(q_hat - baselines["q_hat"])
    _assert(
        q_hat_diff < tolerances["q_hat"],
        f"q_hat={q_hat:.4f} differs from baseline {baselines['q_hat']:.4f} "
        f"by {q_hat_diff:.4f} (tolerance={tolerances['q_hat']})",
    )
    logger.info("  q_hat within tolerance ✓")

    # --- 2. Test inference: certain + uncertain predictions -------------------
    logger.info("2. Testing inference handler...")
    _reset_for_testing()

    # Certain prediction
    certain_event = {
        "question": "What is GDP?",
        "choices": [
            "Total market value of all final goods and services produced",
            "Total money in circulation",
            "Government annual budget",
            "Stock market index",
        ],
    }
    result = lambda_handler(certain_event, context=None)
    _assert(result["statusCode"] == 200, f"certain prediction returned {result['statusCode']}")
    _assert(
        result["body"]["set_size"] == 1,
        f"certain prediction has set_size={result['body']['set_size']}, expected 1",
    )
    logger.info("  certain prediction: set_size=1, routing=auto ✓")

    # Uncertain prediction — use a difficult row from S3 (known to produce uncertainty)
    data_bucket = config.get("data_bucket_name")
    if not data_bucket:
        print(
            "SMOKE FAILED: 'data_bucket_name' missing from DDB config; "
            "re-run deploy_model.py so it seeds the CloudFormation output."
        )
        return 1

    import tempfile

    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name
    s3.download_file(data_bucket, "data/difficult.jsonl", tmp_path)
    with open(tmp_path, encoding="utf-8") as f:
        difficult_rows = [json.loads(line) for line in f if line.strip()]

    found_uncertain = False
    for row in difficult_rows[:10]:  # try up to 10 difficult rows
        uevent = {"question": row["question"], "choices": row["choices"]}
        result = lambda_handler(uevent, context=None)
        if result["statusCode"] == 200 and result["body"]["set_size"] > 1:
            found_uncertain = True
            logger.info(
                "  uncertain prediction: set_size=%d, routing=%s ✓",
                result["body"]["set_size"],
                result["body"]["routing_decision"],
            )
            break
    _assert(
        found_uncertain,
        "none of the difficult rows produced an uncertain prediction (set_size > 1)",
    )

    # --- 3. Check S3 artifacts exist ------------------------------------------
    logger.info("3. Checking S3 artifacts...")

    # Check calibration diagnostics
    try:
        s3.head_object(Bucket=data_bucket, Key="calibration/diagnostics.json")
        logger.info("  calibration/diagnostics.json exists ✓")
    except Exception as err:
        _assert(False, f"calibration/diagnostics.json not found: {err}")

    # Check at least one batch JSON exists
    response = s3.list_objects_v2(Bucket=data_bucket, Prefix="batch/", MaxKeys=1)
    _assert(
        response.get("KeyCount", 0) > 0,
        "no batch JSON found under s3://{data_bucket}/batch/",
    )
    logger.info("  batch/ JSON exists ✓")

    # Check data splits exist
    for split in ["cal.jsonl", "test.jsonl", "difficult.jsonl"]:
        try:
            s3.head_object(Bucket=data_bucket, Key=f"data/{split}")
            logger.info("  data/%s exists ✓", split)
        except Exception as err:
            _assert(False, f"data/{split} not found: {err}")

    # --- 4. Check CloudWatch metrics ------------------------------------------
    logger.info("4. Checking CloudWatch metrics...")
    cw = boto3.client("cloudwatch")
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=24)

    metrics_to_check = ["SetSize", "IsCertain", "PredictionLatencyMs"]
    for metric_name in metrics_to_check:
        stats = cw.get_metric_statistics(
            Namespace="LLMUncertainty",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint_name},
                {"Name": "ModelId", "Value": config.get("model_id", "")},
                {"Name": "TargetCoverage", "Value": config.get("target_coverage", "")},
            ],
            StartTime=start_time,
            EndTime=now,
            Period=3600,
            Statistics=["Sum", "SampleCount"],
        )
        has_data = len(stats.get("Datapoints", [])) > 0
        _assert(has_data, f"CloudWatch metric {metric_name} has no data in last 24h")
        logger.info("  %s: %d datapoints ✓", metric_name, len(stats["Datapoints"]))

    # --- 5. Summary -----------------------------------------------------------
    print()
    print("=" * 72)
    print("SMOKE OK")
    print(f"  q_hat:       {q_hat:.4f} (baseline: {baselines['q_hat']:.4f})")
    print(f"  endpoint:    {endpoint_name}")
    print(f"  data_bucket: {data_bucket}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as err:
        print(str(err))
        sys.exit(1)
