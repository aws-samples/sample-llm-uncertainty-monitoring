# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Package Lambda deployment ZIP and upload to S3.

Installs runtime dependencies (requirements-lambda.txt) into a temporary
directory, overlays the src/ package, and produces a self-contained ZIP.
Uploads the ZIP to the S3 deploy bucket for CloudFormation to reference.

Usage::

    python scripts/package_lambda.py \
        --s3-bucket DOC-EXAMPLE-BUCKET \
        --s3-key lambda-deployments/llm-uncertainty-lambda.zip

The bucket must already exist (created by CloudFormation or manually).
After uploading, update the Lambda function code via::

    aws cloudformation deploy --template-file template.yaml \
        --parameter-overrides LambdaCodeS3Bucket=<bucket> LambdaCodeS3Key=<key> ...
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Lambda ZIP and upload to S3.",
    )
    parser.add_argument(
        "--s3-bucket",
        required=True,
        help="S3 bucket for the deployment ZIP (e.g. DOC-EXAMPLE-BUCKET).",
    )
    parser.add_argument(
        "--s3-key",
        default="lambda-deployments/llm-uncertainty-lambda.zip",
        help="S3 key for the ZIP (default: %(default)s).",
    )
    parser.add_argument(
        "--output-local-path",
        default="",
        help="Optional local path to also write the ZIP (for inspection).",
    )
    return parser


def _build_deployment_zip(project_root: Path) -> bytes:
    """Package application code and runtime dependencies into an in-memory ZIP."""
    src_dir = project_root / "src"
    scripts_dir = project_root / "scripts"
    requirements_file = project_root / "requirements-lambda.txt"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        deps_dir = tmp_path / "deps"
        deps_dir.mkdir()

        logger.info("installing runtime dependencies into temporary directory")
        # Install binary packages (numpy, scipy) targeting Lambda platform
        # (Python 3.10, Linux x86_64)
        subprocess.run(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "--target",
                str(deps_dir),
                "--quiet",
                "--platform",
                "manylinux2014_x86_64",
                "--implementation",
                "cp",
                "--python-version",
                "3.10",
                "--only-binary=:all:",
                "-r",
                str(requirements_file),
            ],
            check=True,
            capture_output=True,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add third-party dependencies
            for root, _dirs, files in os.walk(deps_dir):
                for fname in files:
                    full_path = Path(root) / fname
                    arcname = str(full_path.relative_to(deps_dir))
                    zf.write(full_path, arcname)

            # Add src/llm_uncertainty_monitoring package
            for root, _dirs, files in os.walk(src_dir):
                for fname in files:
                    if fname.endswith(".py"):
                        full_path = Path(root) / fname
                        arcname = str(full_path.relative_to(src_dir))
                        zf.write(full_path, arcname)

            # Add scripts/ package (for batch eval lambda_handler)
            for root, _dirs, files in os.walk(scripts_dir):
                for fname in files:
                    if fname.endswith(".py"):
                        full_path = Path(root) / fname
                        arcname = "scripts/" + fname
                        zf.write(full_path, arcname)

        return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    """Build ZIP and upload to S3."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    zip_bytes = _build_deployment_zip(project_root)
    size_mb = len(zip_bytes) / (1024 * 1024)
    logger.info("deployment ZIP: %.1f MB", size_mb)

    # Optionally write locally
    if args.output_local_path:
        Path(args.output_local_path).write_bytes(zip_bytes)
        logger.info("wrote ZIP to %s", args.output_local_path)

    # Upload to S3
    s3_client = boto3.client("s3")
    s3_client.put_object(
        Bucket=args.s3_bucket,
        Key=args.s3_key,
        Body=zip_bytes,
    )
    logger.info(
        "uploaded %.1f MB ZIP to s3://%s/%s",
        size_mb,
        args.s3_bucket,
        args.s3_key,
    )

    print()
    print("=" * 72)
    print("Lambda deployment package ready!")
    print(f"  S3 location: s3://{args.s3_bucket}/{args.s3_key}")
    print(f"  Size:        {size_mb:.1f} MB")
    print()
    print("Next: deploy or update the CloudFormation stack:")
    print("  aws cloudformation deploy --template-file template.yaml \\")
    print("    --stack-name llm-uncertainty-monitoring \\")
    print(f"    --parameter-overrides LambdaCodeS3Bucket={args.s3_bucket} \\")
    print(f"      LambdaCodeS3Key={args.s3_key} \\")
    print("    --capabilities CAPABILITY_NAMED_IAM")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
