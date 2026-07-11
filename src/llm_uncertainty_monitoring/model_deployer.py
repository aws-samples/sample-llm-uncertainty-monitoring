# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""SageMaker JumpStart deployment, attachment, and teardown.

A thin, model-agnostic wrapper over
:class:`sagemaker.jumpstart.model.JumpStartModel`. The deployer exposes
three operations:

- :meth:`ModelDeployer.deploy` — create a real-time endpoint for a given
  ``(model_id, model_version)`` pair, optionally forwarding the EULA flag.
- :meth:`ModelDeployer.attach` — return a :class:`sagemaker.predictor.Predictor`
  bound to an already-running endpoint.
- :meth:`ModelDeployer.delete_endpoint` — tear down the endpoint, its
  endpoint configuration, and the underlying model.

The deployer never embeds a prompt template, tokenizer, or per-model
parameter default. The returned ``Predictor.predict(payload)`` surface is
the only callable surface; the prefill-scoring contract lives entirely in
``probability_extractor``.

On deploy failure, a :class:`botocore.exceptions.ClientError` raised by
SageMaker is re-raised as :class:`EndpointCreationError` carrying the
endpoint name and the upstream ``Error.Message`` so the caller can log a
single actionable message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from botocore.exceptions import ClientError
from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker.session import Session
from sagemaker.utils import name_from_base

__all__ = [
    "DeployedEndpoint",
    "EndpointCreationError",
    "ModelDeployer",
]


@dataclass(frozen=True)
class DeployedEndpoint:
    """A live SageMaker real-time endpoint and the predictor bound to it.

    ``predictor`` is the only callable surface exposed downstream; callers
    invoke it through ``predictor.predict(payload)``.
    """

    endpoint_name: str
    predictor: Predictor


class EndpointCreationError(RuntimeError):
    """Raised when a JumpStart deploy call fails with a ``ClientError``.

    Carries the endpoint name and the upstream
    ``ClientError.response["Error"]["Message"]`` so a single log line is
    enough to diagnose the failure without re-fetching the SageMaker
    describe-endpoint response.
    """

    def __init__(self, endpoint_name: str, upstream_message: str) -> None:
        self.endpoint_name = endpoint_name
        self.upstream_message = upstream_message
        super().__init__(
            f"failed to create SageMaker endpoint {endpoint_name!r}: {upstream_message}"
        )


class ModelDeployer:
    """Model-agnostic façade over ``JumpStartModel`` for deploy / attach /
    teardown.

    A single ``ModelDeployer`` instance can deploy or tear down many
    endpoints. Pass an explicit ``sagemaker_session`` when running under a
    non-default boto profile (e.g., from a Processing Job role); leave it
    as ``None`` to let the SageMaker SDK build the default session.
    """

    def __init__(self, sagemaker_session: Session | None = None) -> None:
        self._sagemaker_session = sagemaker_session

    # ------------------------------------------------------------------
    # deploy
    # ------------------------------------------------------------------

    def deploy(
        self,
        model_id: str,
        model_version: str,
        instance_type: str,
        initial_instance_count: int,
        accept_eula: bool = False,
    ) -> DeployedEndpoint:
        """Deploy a JumpStart text-generation model to a real-time endpoint.

        The endpoint name is generated up-front from ``model_id`` so it is
        available for the error path even when the underlying
        ``JumpStartModel.deploy`` call fails.

        Args:
            model_id: JumpStart model identifier
                (e.g. ``"meta-textgeneration-llama-3-8b"``).
            model_version: JumpStart model version (``"*"`` accepted by
                JumpStart for "latest").
            instance_type: e.g. ``"ml.g5.2xlarge"``.
            initial_instance_count: number of instances to launch.
            accept_eula: forwarded as-is to ``JumpStartModel.deploy`` for
                gated models that require EULA acknowledgement.

        Returns:
            A :class:`DeployedEndpoint` carrying the generated endpoint
            name and the predictor returned by ``deploy``.

        Raises:
            EndpointCreationError: if SageMaker raises a
                :class:`botocore.exceptions.ClientError` during endpoint
                creation. The error message embeds the endpoint name and
                the upstream ``Error.Message``.
        """
        endpoint_name = name_from_base(model_id)

        model_kwargs: dict[str, Any] = {
            "model_id": model_id,
            "model_version": model_version,
        }
        if self._sagemaker_session is not None:
            model_kwargs["sagemaker_session"] = self._sagemaker_session

        model = JumpStartModel(**model_kwargs)

        try:
            predictor = model.deploy(
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                accept_eula=accept_eula,
            )
        except ClientError as err:
            upstream_message = err.response.get("Error", {}).get("Message", str(err))
            raise EndpointCreationError(endpoint_name, upstream_message) from err

        return DeployedEndpoint(
            endpoint_name=endpoint_name,
            predictor=cast(Predictor, predictor),
        )

    # ------------------------------------------------------------------
    # attach
    # ------------------------------------------------------------------

    def attach(self, endpoint_name: str) -> Predictor:
        """Return a :class:`Predictor` bound to an already-running
        endpoint without re-deploying.

        Useful for re-runs of the offline calibration job and for
        ``scripts/prepare_data.py`` after the endpoint was created in a
        previous session.

        The predictor is configured with JSON serializer/deserializer to
        match the JumpStart TGI inference container's expected content
        types (``application/json`` in and out).
        """
        from sagemaker.deserializers import JSONDeserializer
        from sagemaker.serializers import JSONSerializer

        return Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self._sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )

    # ------------------------------------------------------------------
    # delete_endpoint
    # ------------------------------------------------------------------

    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete the endpoint, its endpoint configuration, and the
        underlying model(s).

        The endpoint configuration and model names are resolved by
        describing the endpoint and the endpoint configuration *before*
        deleting anything, since those describe calls fail once the
        resources are gone.
        """
        session = self._resolve_session()
        sagemaker_client = session.sagemaker_client

        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_desc["EndpointConfigName"]

        config_desc = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        model_names = [variant["ModelName"] for variant in config_desc["ProductionVariants"]]

        session.delete_endpoint(endpoint_name)
        session.delete_endpoint_config(endpoint_config_name)
        for model_name in model_names:
            session.delete_model(model_name)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _resolve_session(self) -> Session:
        """Return the configured SageMaker session, creating a default
        one lazily when none was injected at construction time."""
        if self._sagemaker_session is None:
            self._sagemaker_session = Session()
        return self._sagemaker_session
