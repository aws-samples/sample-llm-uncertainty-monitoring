# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Calibration pipeline: scoring and q_hat computation.

The aggregation logic (``aggregator.compute_calibration``) is called by
``scripts/run_calibration.py`` after scoring the calibration set against
the live endpoint. It computes LAC scores, derives ``q_hat``, and returns
diagnostics.
"""
