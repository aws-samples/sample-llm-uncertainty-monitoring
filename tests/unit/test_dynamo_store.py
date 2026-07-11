# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for ``llm_uncertainty_monitoring.config.dynamo_store``.

``boto3.resource("dynamodb")`` is replaced by a ``MagicMock`` so the
tests run offline. The module-level cache is reset between tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_uncertainty_monitoring.config.dynamo_store import (
    DEFAULT_CONFIG_ID,
    PARTITION_KEY,
    REQUIRED_ATTRIBUTES,
    load_config,
    put_config,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache_between_tests() -> None:
    reset_cache()


def _full_item() -> dict[str, Any]:
    """A DDB item satisfying every required attribute, with mixed value types."""
    return {
        PARTITION_KEY: DEFAULT_CONFIG_ID,
        "endpoint_name": "ep-llama-3-8b",
        "model_id": "meta-textgeneration-llama-3-8b",
        "num_choices": 4,
        "cloudwatch_namespace": "LLMUncertainty",
        "conformal_threshold": 0.42,
        "target_coverage": "0.9",
    }


def _make_resource(get_item_response: dict[str, Any]) -> tuple[MagicMock, MagicMock]:
    table = MagicMock(name="Table")
    table.get_item.return_value = get_item_response
    resource = MagicMock(name="Resource")
    resource.Table.return_value = table
    return resource, table


def test_load_config_returns_required_attributes_as_strings() -> None:
    resource, _ = _make_resource({"Item": _full_item()})

    config = load_config(ddb_resource=resource)

    assert PARTITION_KEY not in config
    for attribute in REQUIRED_ATTRIBUTES:
        assert attribute in config
    assert config["num_choices"] == "4"
    assert config["conformal_threshold"] == "0.42"
    assert all(isinstance(v, str) for v in config.values())


def test_load_config_raises_when_item_or_required_attribute_missing() -> None:
    # Missing Item → naming the config_id and table.
    resource, _ = _make_resource({})
    with pytest.raises(ValueError) as exc_info:
        load_config(config_id="missing-id", table_name="custom-table", ddb_resource=resource)
    assert "missing-id" in str(exc_info.value)
    assert "custom-table" in str(exc_info.value)

    # Missing required attribute → naming the attribute.
    item = _full_item()
    item.pop("conformal_threshold")
    resource, _ = _make_resource({"Item": item})
    with pytest.raises(ValueError) as exc_info:
        load_config(ddb_resource=resource)
    assert "conformal_threshold" in str(exc_info.value)


def test_load_config_caches_repeat_calls() -> None:
    resource, table = _make_resource({"Item": _full_item()})

    load_config(ddb_resource=resource)
    load_config(ddb_resource=resource)

    assert table.get_item.call_count == 1


def test_put_config_writes_via_update_expression_and_invalidates_cache() -> None:
    item = _full_item()
    resource, table = _make_resource({"Item": item})

    # Prime the cache.
    load_config(ddb_resource=resource)
    assert table.get_item.call_count == 1

    put_config(
        {"endpoint_name": "ep-replaced", "num_choices": 4, "conformal_threshold": 0.42},
        ddb_resource=resource,
    )

    # update_item shape: SET with three placeholder pairs, str-coerced values.
    kwargs = table.update_item.call_args.kwargs
    assert kwargs["Key"] == {PARTITION_KEY: DEFAULT_CONFIG_ID}
    assert kwargs["UpdateExpression"].startswith("SET ")
    assert kwargs["UpdateExpression"].count("=") == 3
    assert set(kwargs["ExpressionAttributeNames"].values()) == {
        "endpoint_name",
        "num_choices",
        "conformal_threshold",
    }
    assert all(isinstance(v, str) for v in kwargs["ExpressionAttributeValues"].values())

    # Cache invalidation: next load_config re-queries DDB.
    load_config(ddb_resource=resource)
    assert table.get_item.call_count == 2
