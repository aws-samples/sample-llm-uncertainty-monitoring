# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CLI wrapper around :class:`ModelDeployer` for one-shot endpoint provisioning.

Steps:
1. Deploy a SageMaker JumpStart text-generation model to a real-time endpoint.
2. Probe the endpoint with ``max_new_tokens=1, details=True,
   decoder_input_details=True`` and assert ``details.prefill`` is non-empty —
   catches early any model that does not expose prefill log-probabilities,
   which the downstream ``probability_extractor`` relies on.
3. Seed the DynamoDB ``Config`` item under ``--config-id`` so the calibration
   job and inference Lambda can discover the endpoint without hard-coded values.

Prints the endpoint name on the final stdout line. Exits non-zero on any
failure with a useful error message; deliberately does **not** delete the
endpoint on probe failure (deploys are slow and expensive).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import boto3

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    DEFAULT_CONFIG_TABLE,
    ensure_table,
    put_config,
)
from llm_uncertainty_monitoring.model_deployer import (
    DeployedEndpoint,
    EndpointCreationError,
    ModelDeployer,
)

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a JumpStart text-generation model to a SageMaker real-time "
            "endpoint, probe it for prefill log-probabilities, and seed the "
            "initial DynamoDB Config item."
        ),
    )
    parser.add_argument(
        "--model-id",
        default="meta-textgeneration-llama-3-8b",
        help="JumpStart model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--model-version",
        default="*",
        help='JumpStart model version. JumpStart accepts "*" for "latest" (default: %(default)s).',
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.2xlarge",
        help="SageMaker hosting instance type (default: %(default)s).",
    )
    parser.add_argument(
        "--initial-instance-count",
        type=int,
        default=1,
        help="Number of instances to launch behind the endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--accept-eula",
        action="store_true",
        help="Forward EULA acceptance to JumpStart for gated models (e.g. Llama 3).",
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
        "--cloudwatch-namespace",
        default="LLMUncertainty",
        help="CloudWatch namespace seeded into the Config item (default: %(default)s).",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=4,
        help="Number of answer choices K (default: %(default)s).",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help="Target marginal coverage for conformal prediction (default: %(default)s).",
    )
    parser.add_argument(
        "--stack-name",
        default="llm-uncertainty-monitoring",
        help=(
            "CloudFormation stack name. If the stack exists, its "
            "DataBucketName / HumanReviewBucketName outputs are read and "
            "seeded into the DynamoDB Config item so downstream scripts do "
            "not need to hardcode bucket names (default: %(default)s)."
        ),
    )
    return parser


class ProbeError(RuntimeError):
    """Raised when the probe request does not return prefill log-probabilities."""


def _probe_endpoint(deployed: DeployedEndpoint) -> None:
    """Send a single probe request and assert ``details.prefill`` is non-empty."""
    payload: dict[str, Any] = {
        "inputs": "Hello",
        "parameters": {
            "max_new_tokens": 1,
            "details": True,
            "decoder_input_details": True,
        },
    }
    logger.info("probing endpoint %s for prefill log-probabilities", deployed.endpoint_name)
    response = deployed.predictor.predict(payload)

    if not isinstance(response, dict):
        raise ProbeError(
            f"endpoint {deployed.endpoint_name!r} returned a non-dict response of type "
            f"{type(response).__name__}; aborting"
        )
    details = response.get("details")
    if not isinstance(details, dict):
        raise ProbeError(
            f"endpoint {deployed.endpoint_name!r} response is missing a 'details' object; aborting"
        )
    prefill = details.get("prefill")
    if not isinstance(prefill, list) or not prefill:
        raise ProbeError(
            f"endpoint {deployed.endpoint_name!r} does not return prefill logprobs "
            "(missing or empty 'details.prefill'); aborting"
        )
    logger.info(
        "probe succeeded — endpoint %s emitted %d prefill tokens",
        deployed.endpoint_name,
        len(prefill),
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, deploy, probe, seed DDB. Return an exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    config_id = args.config_id.strip()
    table_name = args.config_table.strip()
    if not config_id or not table_name:
        logger.error("--config-id and --config-table must be non-empty")
        return 1

    deployer = ModelDeployer()

    logger.info(
        "deploying JumpStart model %s (version=%s) to %s x%d",
        args.model_id,
        args.model_version,
        args.instance_type,
        args.initial_instance_count,
    )
    try:
        deployed = deployer.deploy(
            model_id=args.model_id,
            model_version=args.model_version,
            instance_type=args.instance_type,
            initial_instance_count=args.initial_instance_count,
            accept_eula=args.accept_eula,
        )
    except EndpointCreationError as err:
        logger.error("deployment failed: %s", err)
        return 1

    logger.info("endpoint %s is in service", deployed.endpoint_name)

    try:
        _probe_endpoint(deployed)
    except ProbeError as err:
        logger.error("%s", err)
        logger.error(
            "leaving endpoint %s in place for manual inspection; delete it with "
            "ModelDeployer.delete_endpoint when no longer needed",
            deployed.endpoint_name,
        )
        return 1

    config_values: dict[str, str] = {
        "endpoint_name": deployed.endpoint_name,
        "model_id": args.model_id,
        "num_choices": str(args.num_choices),
        "cloudwatch_namespace": args.cloudwatch_namespace,
        "nonconformity_score": "LAC",
        "target_coverage": str(args.target_coverage),
    }

    # Best-effort: pull the auto-generated bucket names from the stack outputs
    # so downstream scripts can look them up in DynamoDB instead of hardcoding.
    # If the stack does not exist yet (e.g. user hasn't deployed the template),
    # log a warning and continue — the caller can pass --data-bucket / etc. to
    # individual scripts.
    stack_name = args.stack_name.strip()
    if stack_name:
        try:
            cfn = boto3.client("cloudformation")
            response = cfn.describe_stacks(StackName=stack_name)
            outputs = {
                item["OutputKey"]: item["OutputValue"]
                for item in response["Stacks"][0].get("Outputs", [])
            }
            for output_key, config_key in (
                ("DataBucketName", "data_bucket_name"),
                ("HumanReviewBucketName", "human_review_bucket_name"),
            ):
                if output_key in outputs:
                    config_values[config_key] = outputs[output_key]
            logger.info(
                "read %d bucket-name outputs from stack %s",
                sum(
                    1
                    for k in ("data_bucket_name", "human_review_bucket_name")
                    if k in config_values
                ),
                stack_name,
            )
        except Exception as err:  # noqa: BLE001 — non-fatal if stack absent
            logger.warning(
                "could not read CloudFormation outputs from stack %s: %s. "
                "Downstream scripts will need bucket names supplied explicitly.",
                stack_name,
                err,
            )

    try:
        ensure_table(table_name)
        put_config(config_values, config_id=config_id, table_name=table_name)
    except Exception as err:  # noqa: BLE001 — surface any boto/credential error verbatim
        logger.error(
            "failed to seed DynamoDB %s.config_id=%s: %s",
            table_name,
            config_id,
            err,
        )
        return 1

    logger.info(
        "seeded %d attributes into DynamoDB %s.config_id=%s",
        len(config_values),
        table_name,
        config_id,
    )
    print(deployed.endpoint_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
