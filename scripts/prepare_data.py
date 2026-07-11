# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""CLI driver for the offline data pipeline.

Wires together the three building blocks already implemented in the
package — ``data_loader`` (MMLU loading + split helpers),
``probability_extractor`` (prefill-based softmax scoring), and
``model_deployer`` (attaching to an already-running endpoint) — into one
reproducible pass:

1. Load the requested MMLU subjects, with a global ``index`` column.
2. Attach to the deployed SageMaker endpoint and, for every row, render
   the K-token prompt and call the endpoint to get an answer-probability
   vector. The resulting ``probs_by_index`` mapping is memoized to a
   local JSON file alongside a manifest tying the cache to a specific
   endpoint, model, prompt template, and subject list.
3. On re-run with a matching manifest the endpoint pass is skipped and
   the cached probabilities are reused.
4. Extract the ``difficult-per-subject`` hardest rows per subject, split
   the remainder into calibration and test, and persist the three
   JSONL splits to S3.

The script is the entry point ``llmum-prepare-data`` (see
``pyproject.toml``). The endpoint pass dominates wall-clock time
(``len(dataset) * num_choices`` real-time invocations); a ``tqdm``
progress bar is used when available, with a plain ``print`` fallback so
the dependency stays optional.

Example
-------
::

    python scripts/prepare_data.py \\
        --endpoint-name meta-textgeneration-llama-3-8b-2026-05-22-12-00-00 \\
        --s3-uri-prefix s3://DOC-EXAMPLE-BUCKET/data \\
        --probs-cache-path ./probs_by_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import boto3
from botocore.exceptions import ClientError
from datasets import Dataset

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    DEFAULT_CONFIG_TABLE,
    load_config,
)
from llm_uncertainty_monitoring.data_loader import (
    SplitResult,
    extract_difficult,
    load_mmlu,
    persist_splits_to_s3,
    split_remaining,
)
from llm_uncertainty_monitoring.model_deployer import ModelDeployer
from llm_uncertainty_monitoring.probability_extractor import (
    DEFAULT_ANSWER_TOKENS,
    build_prompt_and_probs,
    default_prompt_builder,
)

logger = logging.getLogger(__name__)


DEFAULT_SUBJECTS: tuple[str, ...] = (
    "econometrics",
    "marketing",
    "management",
    "sociology",
)
DEFAULT_DIFFICULT_PER_SUBJECT = 10
DEFAULT_CALIBRATION_FRACTION = 0.5
DEFAULT_SEED = 0
DEFAULT_PROBS_CACHE_PATH = "./probs_by_index.json"


# ---------------------------------------------------------------------------
# S3 bucket helper
# ---------------------------------------------------------------------------


def _ensure_bucket(s3_uri_prefix: str) -> None:
    """Create the S3 bucket if it does not exist.

    Parses the bucket name from the ``s3://bucket/prefix`` URI and
    creates it in the session's default region. Idempotent — silently
    succeeds if the bucket already exists and is owned by the caller.
    """
    if not s3_uri_prefix.startswith("s3://"):
        return
    bucket = s3_uri_prefix[len("s3://") :].split("/", 1)[0]
    if not bucket:
        return

    s3_client = boto3.client("s3")
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as err:
        error_code = err.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            region = boto3.session.Session().region_name or "us-east-1"
            create_kwargs: dict[str, Any] = {"Bucket": bucket}
            if region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": region,
                }
            s3_client.create_bucket(**create_kwargs)
            logger.info("created S3 bucket: %s", bucket)
        else:
            raise


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the CLI.

    Defaults match the POC values referenced in the design document — the
    four MMLU subjects, ``difficult_per_subject=10``,
    ``calibration_fraction=0.5``, and ``seed=0``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline data pipeline: load MMLU, score every row against the "
            "deployed endpoint, memoize the probability vectors locally, then write "
            "difficult/calibration/test JSONL splits to S3."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(DEFAULT_SUBJECTS),
        help="MMLU subjects to load (default: %(default)s).",
    )
    parser.add_argument(
        "--difficult-per-subject",
        type=int,
        default=DEFAULT_DIFFICULT_PER_SUBJECT,
        help="Number of hardest rows to keep per subject (default: %(default)s).",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help=(
            "Fraction of the non-difficult rows assigned to the calibration split "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the calibration/test shuffle (default: %(default)s).",
    )
    parser.add_argument(
        "--endpoint-name",
        required=True,
        help="Name of the already-deployed SageMaker real-time endpoint to query.",
    )
    parser.add_argument(
        "--config-id",
        default=DEFAULT_CONFIG_ID,
        help=(
            "DynamoDB Config partition-key value to read model_id from "
            "(default: %(default)s). Must match what deploy_model.py wrote."
        ),
    )
    parser.add_argument(
        "--config-table",
        default=DEFAULT_CONFIG_TABLE,
        help="DynamoDB table holding the Config item (default: %(default)s).",
    )
    parser.add_argument(
        "--s3-uri-prefix",
        default="",
        help=(
            "S3 prefix under which difficult.jsonl, cal.jsonl, test.jsonl will be "
            "written. If empty, derives from "
            "the 'data_bucket_name' attribute in the DynamoDB config "
            "(seeded by deploy_model.py from CloudFormation stack outputs). "
            "Example: s3://DOC-EXAMPLE-BUCKET/data."
        ),
    )
    parser.add_argument(
        "--probs-cache-path",
        default=DEFAULT_PROBS_CACHE_PATH,
        help="Local JSON file used to memoize probs_by_index (default: %(default)s).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore the local cache and re-query the endpoint for every row.",
    )
    return parser


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _compute_prompt_template_hash() -> str:
    """Return a 16-char SHA-256 prefix of the default prompt template.

    Embedding this hash in the cache manifest invalidates the cache when
    the prompt builder changes, so a swapped template forces a fresh
    endpoint pass even if the endpoint and model id are unchanged.
    """
    template = default_prompt_builder("Q?", ["A", "B", "C", "D"])
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def _build_manifest(
    endpoint_name: str,
    model_id: str,
    prompt_template_hash: str,
    subjects: list[str],
) -> dict[str, Any]:
    """Build the manifest written alongside ``probs_by_index`` in the cache."""
    return {
        "endpoint_name": endpoint_name,
        "model_id": model_id,
        "prompt_template_hash": prompt_template_hash,
        "subjects": list(subjects),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_matches(
    cached_manifest: dict[str, Any],
    endpoint_name: str,
    model_id: str,
    prompt_template_hash: str,
    subjects: list[str],
) -> bool:
    """Return True iff the cached manifest is compatible with the
    current run.

    The cache is only reused when the endpoint, model id, prompt
    template, and subject list (in the same order — ``load_mmlu`` resets
    the global ``index`` column to ``range(len(...))`` so a different
    subject order yields different indices) all match.
    """
    return (
        cached_manifest.get("endpoint_name") == endpoint_name
        and cached_manifest.get("model_id") == model_id
        and cached_manifest.get("prompt_template_hash") == prompt_template_hash
        and list(cached_manifest.get("subjects", [])) == list(subjects)
    )


def _load_probs_cache(path: Path) -> tuple[dict[str, Any], dict[int, list[float]]] | None:
    """Read the cache file and return ``(manifest, probs_by_index)``.

    JSON object keys are strings; the ``probs_by_index`` map is converted
    back to ``int`` keys here so it can be passed straight to
    :func:`extract_difficult`. Returns ``None`` if the file is missing or
    malformed.
    """
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("could not read probs cache at %s: %s; ignoring it", path, err)
        return None

    manifest = payload.get("manifest")
    raw_probs = payload.get("probs_by_index")
    if not isinstance(manifest, dict) or not isinstance(raw_probs, dict):
        logger.warning(
            "probs cache at %s is missing 'manifest' or 'probs_by_index'; ignoring it", path
        )
        return None

    try:
        probs_by_index = {int(k): [float(x) for x in v] for k, v in raw_probs.items()}
    except (TypeError, ValueError) as err:
        logger.warning(
            "probs cache at %s has malformed entries: %s; ignoring it",
            path,
            err,
        )
        return None

    return manifest, probs_by_index


def _write_probs_cache(
    path: Path,
    manifest: dict[str, Any],
    probs_by_index: dict[int, list[float]],
) -> None:
    """Persist the manifest and probability map to ``path`` as JSON.

    Writes through a sibling ``.tmp`` file and replaces atomically to
    avoid leaving a half-written cache behind on crash.
    """
    payload = {
        "manifest": manifest,
        "probs_by_index": {str(k): v for k, v in probs_by_index.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Endpoint pass
# ---------------------------------------------------------------------------


def _iter_with_progress(rows: list[dict[str, Any]], description: str) -> Any:
    """Wrap an iterable in ``tqdm`` when available, else fall back to a
    plain carriage-return progress log.

    ``tqdm`` is a transitive dependency of ``datasets`` so it is usually
    importable in this environment; the fallback keeps the script
    runnable in stripped-down environments without adding a hard
    dependency.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover — defensive fallback path
        total = len(rows)

        def _generator() -> Any:
            for i, row in enumerate(rows, start=1):
                yield row
                if i == total or i % 25 == 0:
                    print(f"{description}: {i}/{total}", end="\r", flush=True)
            print()

        return _generator()
    return tqdm(rows, desc=description, total=len(rows))


def _query_probs_for_dataset(
    dataset: Dataset,
    endpoint_name: str,
) -> dict[int, list[float]]:
    """Score every row of ``dataset`` against the deployed endpoint and
    return the resulting ``probs_by_index`` mapping.

    Uses :meth:`ModelDeployer.attach` to bind a ``Predictor`` to the
    already-running endpoint — the script never deploys a new one.
    """
    predictor = ModelDeployer().attach(endpoint_name)
    rows = list(dataset)

    probs_by_index: dict[int, list[float]] = {}
    logger.info(
        "querying endpoint %s for %d rows × %d answer tokens",
        endpoint_name,
        len(rows),
        len(DEFAULT_ANSWER_TOKENS),
    )
    for row in _iter_with_progress(rows, "scoring rows"):
        index = int(row["index"])
        question = str(row["question"])
        choices = list(row["choices"])
        _, probs = build_prompt_and_probs(predictor, question, choices)
        probs_by_index[index] = [float(x) for x in probs.tolist()]
    return probs_by_index


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the full offline data pipeline.

    Returns ``0`` on success, ``1`` on any handled failure (the failure
    is logged with enough context to act on without re-running).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argument_parser().parse_args(argv)

    subjects = list(args.subjects)
    cache_path = Path(args.probs_cache_path)
    prompt_template_hash = _compute_prompt_template_hash()

    # Step 0 — read model_id from DDB so the cache manifest is consistent
    # with whatever deploy_model.py wrote. This avoids a stale --model-id
    # CLI flag pointing at a different model than the live endpoint.
    logger.info(
        "step 0/4: reading model_id from DDB (table=%s, config_id=%s)",
        args.config_table,
        args.config_id,
    )
    try:
        config = load_config(config_id=args.config_id, table_name=args.config_table, strict=False)
    except ValueError as err:
        logger.error("could not read config from DDB: %s", err)
        return 1
    model_id = config["model_id"]
    logger.info("model_id from DDB: %s", model_id)

    # Derive default S3 URI prefix if not provided.
    if not args.s3_uri_prefix:
        data_bucket = config.get("data_bucket_name")
        if not data_bucket:
            logger.error(
                "no --s3-uri-prefix supplied and 'data_bucket_name' is not "
                "in the DynamoDB config. Run deploy_model.py after the "
                "CloudFormation stack is created so it can seed the "
                "auto-generated bucket names from stack outputs, or pass "
                "--s3-uri-prefix explicitly."
            )
            return 1
        args.s3_uri_prefix = f"s3://{data_bucket}/data"
        logger.info("derived --s3-uri-prefix from DDB: %s", args.s3_uri_prefix)

    # Step 1 — load MMLU.
    logger.info("step 1/4: loading MMLU subjects %s", subjects)
    try:
        dataset = load_mmlu(subjects)
    except ValueError as err:
        logger.error("failed to load MMLU subjects: %s", err)
        return 1
    logger.info("loaded %d rows across %d subjects", len(dataset), len(subjects))

    # Step 2 — endpoint pass (or cache hit).
    cached = None if args.force_refresh else _load_probs_cache(cache_path)
    if cached is not None and _manifest_matches(
        cached[0],
        args.endpoint_name,
        model_id,
        prompt_template_hash,
        subjects,
    ):
        logger.info(
            "step 2/4: reusing cached probs_by_index from %s "
            "(endpoint=%s, model=%s, prompt_hash=%s)",
            cache_path,
            args.endpoint_name,
            model_id,
            prompt_template_hash,
        )
        probs_by_index = cached[1]
    else:
        if args.force_refresh:
            logger.info("step 2/4: --force-refresh set, ignoring any cached probs")
        elif cached is None:
            logger.info("step 2/4: no usable cache at %s, running endpoint pass", cache_path)
        else:
            logger.info(
                "step 2/4: cache manifest mismatch at %s "
                "(cached=%s, current endpoint=%s model=%s prompt_hash=%s); re-querying",
                cache_path,
                cached[0],
                args.endpoint_name,
                model_id,
                prompt_template_hash,
            )
        try:
            probs_by_index = _query_probs_for_dataset(dataset, args.endpoint_name)
        except Exception as err:  # noqa: BLE001 — surface any boto/sagemaker error verbatim
            logger.error("endpoint pass failed: %s", err)
            return 1

        manifest = _build_manifest(
            endpoint_name=args.endpoint_name,
            model_id=model_id,
            prompt_template_hash=prompt_template_hash,
            subjects=subjects,
        )
        try:
            _write_probs_cache(cache_path, manifest, probs_by_index)
        except OSError as err:
            logger.error("failed to write probs cache to %s: %s", cache_path, err)
            return 1
        logger.info("wrote %d probability vectors to %s", len(probs_by_index), cache_path)

    # Step 3 — extract the hardest N rows per subject.
    logger.info("step 3/4: selecting %d hardest rows per subject", args.difficult_per_subject)
    try:
        difficult = extract_difficult(
            dataset,
            probs_by_index,
            n_per_group=args.difficult_per_subject,
        )
    except ValueError as err:
        logger.error("difficult-row extraction failed: %s", err)
        return 1

    # Step 4 — calibration/test split + S3 persistence.
    logger.info(
        "step 4/4: splitting remainder into calibration/test "
        "(calibration_fraction=%.3f, seed=%d) and writing to %s",
        args.calibration_fraction,
        args.seed,
        args.s3_uri_prefix,
    )
    try:
        calibration, test = split_remaining(
            dataset,
            difficult,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
    except ValueError as err:
        logger.error("calibration/test split failed: %s", err)
        return 1

    splits = SplitResult(difficult=difficult, calibration=calibration, test=test)
    try:
        _ensure_bucket(args.s3_uri_prefix)
        s3_uris = persist_splits_to_s3(splits, args.s3_uri_prefix)
    except Exception as err:  # noqa: BLE001 — surface any S3/boto error verbatim
        logger.error("failed to write splits to %s: %s", args.s3_uri_prefix, err)
        return 1

    # Final summary.
    s3_uris_str = cast(dict[str, str], s3_uris)
    print()
    print("=== prepare_data summary ===")
    print(f"subjects:              {subjects}")
    print(f"input rows:            {len(dataset)}")
    print(f"difficult rows:        {len(difficult)}")
    print(f"calibration rows:      {len(calibration)}")
    print(f"test rows:             {len(test)}")
    print(f"probs cache path:      {cache_path}")
    print(f"difficult S3 URI:      {s3_uris_str['difficult']}")
    print(f"calibration S3 URI:    {s3_uris_str['calibration']}")
    print(f"test S3 URI:           {s3_uris_str['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
