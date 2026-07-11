# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""DynamoDB-backed configuration store.

Single table (default ``LLMUncertaintyConfig``), partition key
``config_id`` (default ``"default"``). One row per ``config_id`` holds
all attributes the pipeline needs at runtime (``q_hat``,
``endpoint_name``, ``target_coverage``, ...). ``load_config`` validates
required attributes are present, caches the result per
``(table, config_id)``; ``put_config`` upserts a subset and invalidates
the cache.
"""

from __future__ import annotations

from typing import Any

import boto3

__all__ = [
    "DEFAULT_CONFIG_TABLE",
    "DEFAULT_CONFIG_ID",
    "REQUIRED_ATTRIBUTES",
    "PARTITION_KEY",
    "ensure_table",
    "load_config",
    "put_config",
    "reset_cache",
]


DEFAULT_CONFIG_TABLE: str = "LLMUncertaintyConfig"
DEFAULT_CONFIG_ID: str = "default"
PARTITION_KEY: str = "config_id"

REQUIRED_ATTRIBUTES: tuple[str, ...] = (
    "conformal_threshold",
    "target_coverage",
    "endpoint_name",
    "model_id",
    "num_choices",
    "cloudwatch_namespace",
)
"""Attributes the inference Lambda needs at cold start."""


_CACHE: dict[tuple[str, str], dict[str, str]] = {}


def reset_cache() -> None:
    """Clear the in-process cache. Tests use this; production code does not."""
    _CACHE.clear()


def ensure_table(
    table_name: str = DEFAULT_CONFIG_TABLE,
    ddb_client: Any | None = None,
) -> None:
    """Create the table if absent. PAY_PER_REQUEST, single ``config_id`` String hash. Idempotent."""
    client = ddb_client if ddb_client is not None else boto3.client("dynamodb")
    try:
        client.describe_table(TableName=table_name)
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    client.create_table(
        TableName=table_name,
        AttributeDefinitions=[{"AttributeName": PARTITION_KEY, "AttributeType": "S"}],
        KeySchema=[{"AttributeName": PARTITION_KEY, "KeyType": "HASH"}],
        BillingMode="PAY_PER_REQUEST",
    )
    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=table_name, WaiterConfig={"Delay": 5, "MaxAttempts": 30})


def load_config(
    config_id: str = DEFAULT_CONFIG_ID,
    table_name: str = DEFAULT_CONFIG_TABLE,
    ddb_resource: Any | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Return all non-key attributes (stringified) for ``config_id``.

    Cached per ``(table_name, config_id)``. When *strict* is True
    (default), raises ``ValueError`` if any required attribute is absent.
    Set ``strict=False`` to allow partial configs (e.g. before
    calibration has written ``conformal_threshold``).
    """
    cache_key = (table_name, config_id)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    resource = ddb_resource if ddb_resource is not None else boto3.resource("dynamodb")
    table = resource.Table(table_name)

    item = table.get_item(Key={PARTITION_KEY: config_id}).get("Item")
    if item is None:
        raise ValueError(
            f"no configuration item found for config_id={config_id!r} in table {table_name!r}"
        )

    if strict:
        for attribute in REQUIRED_ATTRIBUTES:
            if attribute not in item:
                raise ValueError(
                    f"required configuration attribute {attribute!r} missing "
                    f"from item config_id={config_id!r} in table {table_name!r}"
                )

    config: dict[str, str] = {
        key: str(value) for key, value in item.items() if key != PARTITION_KEY
    }
    _CACHE[cache_key] = config
    return config


def put_config(
    values: dict[str, Any],
    config_id: str = DEFAULT_CONFIG_ID,
    table_name: str = DEFAULT_CONFIG_TABLE,
    ddb_resource: Any | None = None,
) -> None:
    """Upsert ``values`` into the row for ``config_id``.

    Uses ``UpdateExpression`` placeholder maps so reserved DDB keywords
    don't collide. Unspecified attributes are preserved. Invalidates the
    cache entry on success.
    """
    if not values:
        raise ValueError("put_config requires at least one attribute to write")

    resource = ddb_resource if ddb_resource is not None else boto3.resource("dynamodb")
    table = resource.Table(table_name)

    set_clauses: list[str] = []
    expression_names: dict[str, str] = {}
    expression_values: dict[str, str] = {}
    for index, (key, value) in enumerate(values.items()):
        name_placeholder = f"#k{index}"
        value_placeholder = f":v{index}"
        set_clauses.append(f"{name_placeholder} = {value_placeholder}")
        expression_names[name_placeholder] = key
        expression_values[value_placeholder] = str(value)

    table.update_item(
        Key={PARTITION_KEY: config_id},
        UpdateExpression="SET " + ", ".join(set_clauses),
        ExpressionAttributeNames=expression_names,
        ExpressionAttributeValues=expression_values,
    )

    _CACHE.pop((table_name, config_id), None)
