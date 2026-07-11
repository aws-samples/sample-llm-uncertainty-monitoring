# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Unit tests for ``llm_uncertainty_monitoring.model_deployer``.

POC-scope tests: one happy path per public method plus the error-wrapping
contract on ``deploy``. SageMaker calls are mocked at the module boundary.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from llm_uncertainty_monitoring.model_deployer import (
    DeployedEndpoint,
    EndpointCreationError,
    ModelDeployer,
)

MODULE = "llm_uncertainty_monitoring.model_deployer"


def test_deploy_returns_deployed_endpoint() -> None:
    """``deploy`` builds a JumpStartModel, calls ``.deploy(...)`` with the
    expected kwargs, and wraps the result in a ``DeployedEndpoint``."""
    fake_predictor = MagicMock(name="Predictor")
    fake_model = MagicMock(name="JumpStartModel-instance")
    fake_model.deploy.return_value = fake_predictor

    with (
        patch(f"{MODULE}.JumpStartModel", return_value=fake_model) as model_cls,
        patch(f"{MODULE}.name_from_base", return_value="my-endpoint-2024-01-01"),
    ):
        result = ModelDeployer().deploy(
            model_id="meta-textgeneration-llama-3-8b",
            model_version="2.0.0",
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
            accept_eula=True,
        )

    model_cls.assert_called_once_with(
        model_id="meta-textgeneration-llama-3-8b",
        model_version="2.0.0",
    )
    fake_model.deploy.assert_called_once_with(
        initial_instance_count=1,
        instance_type="ml.g5.2xlarge",
        endpoint_name="my-endpoint-2024-01-01",
        accept_eula=True,
    )
    assert isinstance(result, DeployedEndpoint)
    assert result.endpoint_name == "my-endpoint-2024-01-01"
    assert result.predictor is fake_predictor


def test_deploy_wraps_client_error() -> None:
    """A ``ClientError`` raised by ``.deploy()`` is re-raised as
    :class:`EndpointCreationError` carrying the endpoint name and the
    upstream ``Error.Message``."""
    fake_model = MagicMock()
    fake_model.deploy.side_effect = ClientError(
        error_response={
            "Error": {"Code": "ValidationException", "Message": "EULA acceptance required."},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        operation_name="CreateEndpoint",
    )

    with (
        patch(f"{MODULE}.JumpStartModel", return_value=fake_model),
        patch(f"{MODULE}.name_from_base", return_value="ep-failing"),
        pytest.raises(EndpointCreationError) as exc_info,
    ):
        ModelDeployer().deploy(
            model_id="m",
            model_version="*",
            instance_type="ml.g5.2xlarge",
            initial_instance_count=1,
        )

    err = exc_info.value
    assert err.endpoint_name == "ep-failing"
    assert "EULA acceptance required." in str(err)


def test_attach_returns_predictor() -> None:
    """``attach`` returns a ``Predictor`` bound to the supplied endpoint."""
    fake_predictor = MagicMock(name="Predictor")

    with patch(f"{MODULE}.Predictor", return_value=fake_predictor) as predictor_cls:
        result = ModelDeployer().attach("existing-endpoint")

    # Verify the call includes serializer and deserializer (for JSON payloads)
    call_kwargs = predictor_cls.call_args[1]
    assert call_kwargs["endpoint_name"] == "existing-endpoint"
    assert call_kwargs["sagemaker_session"] is None
    assert "serializer" in call_kwargs
    assert "deserializer" in call_kwargs
    assert result is fake_predictor


def test_delete_endpoint_tears_down_endpoint_config_and_model() -> None:
    """``delete_endpoint`` describes the endpoint and its config to
    discover dependent names, then deletes endpoint, endpoint config,
    and every model behind the production variants."""
    session = MagicMock(name="Session")
    session.sagemaker_client.describe_endpoint.return_value = {
        "EndpointConfigName": "ep-config",
    }
    session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [{"ModelName": "model-a"}],
    }

    ModelDeployer(sagemaker_session=session).delete_endpoint("ep-to-delete")

    session.delete_endpoint.assert_called_once_with("ep-to-delete")
    session.delete_endpoint_config.assert_called_once_with("ep-config")
    session.delete_model.assert_called_once_with("model-a")
