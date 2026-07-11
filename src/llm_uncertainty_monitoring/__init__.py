# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""LLM Uncertainty Monitoring System.

Split conformal prediction with the LAC nonconformity score, deployed as an
AWS-native pipeline (SageMaker JumpStart endpoint, SageMaker Processing Job for
calibration, AWS Lambda for inference, SSM Parameter Store for configuration,
CloudWatch for metrics, SQS for human review routing).
"""

__version__ = "0.1.0"
