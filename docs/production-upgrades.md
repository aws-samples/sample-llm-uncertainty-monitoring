# Production Upgrades

This document describes production-grade enhancements beyond the POC implementation.

## Infrastructure as Code

Replace the boto3 deploy scripts with CloudFormation, CDK, or Terraform:
- `bootstrap_iam.py` → IAM resources (`AWS::IAM::Role`, `AWS::IAM::Policy`)
- `deploy_inference_lambda.py` → `AWS::Lambda::Function`, `AWS::SQS::Queue`, `AWS::SNS::Topic`, `AWS::Lambda::EventSourceMapping`
- `schedule_batch_evaluation.py` → `AWS::Scheduler::Schedule`

Benefits: version-controlled infra, drift detection, rollback capability.

## Cognito-Authenticated Reviewer Dashboard

Replace the SNS notification flow with a web UI:

- **Frontend**: Amplify SPA (React) with Cognito User Pool authentication
- **Backend**: API Gateway + Lambda (list pending, submit feedback, mark resolved)
- **Storage**: DynamoDB table for review state (`pending → in_review → resolved`)
- **Workflow**: Reviewer sees question + probability vector + prediction set → selects correct answer → feedback lands in a `feedback/` S3 prefix for retraining

This closes the feedback loop: reviewer labels flow back into the calibration set for periodic recalibration.

## Amazon Augmented AI (A2I)

Alternative to the custom dashboard:

- Replace the Reviewer Lambda's SQS → SNS path with `start_human_loop()`
- A2I provides: managed worker UI, task assignment, timeout/retry, workforce management
- Cognito workforce authenticates reviewers
- Completed reviews write to S3 → ingest into feedback table
- Switchable via `human_review_destination_type = "a2i"` in DynamoDB config

## DLQ and Idempotency

- **Dead Letter Queue**: Add a DLQ to the SQS Human Review Queue (`maxReceiveCount = 3`). Failed messages go to the DLQ for manual inspection.
- **Idempotency**: Use a SHA-256 hash of `(question, choices, timestamp)` as a DynamoDB idempotency key to prevent duplicate processing (or use AWS Lambda Powertools idempotency utility).

## Event Replay from S3

- Buffer all Lambda invocation inputs/outputs to S3 via Kinesis Data Firehose
- Enables replay of historical traffic for regression testing after model updates
- Supports audit requirements in regulated environments

## Scalability

| Current (POC) | Production upgrade |
|---------------|-------------------|
| Single Lambda (1024 MB) | Lambda with provisioned concurrency (warm starts) |
| Sequential K endpoint calls | Parallelized prefill calls (async invocations) |
| One SageMaker endpoint | Endpoint autoscaling (min/max instance count) |
| Lambda-based inference | Fargate behind ALB (for sustained high-throughput) |

## Network and Security

- **VPC endpoints**: Place Lambda in a VPC with interface endpoints for SageMaker, DynamoDB, SQS, SNS, S3, CloudWatch (no internet egress)
- **Encryption**: SSE-KMS on S3 buckets, DynamoDB encryption at rest, SecureString for sensitive config values
- **Object Lock**: Enable S3 Object Lock on diagnostics and batch JSON (compliance mode) for audit immutability

## SageMaker Model Monitor

Not used in the POC (our monitoring signal is `SetSize` per request, computed in real time). Model Monitor adds value for:

- **Input feature drift**: Detect when the question distribution shifts (e.g., new vocabulary, different domains)
- **Data quality monitoring**: Automated statistical tests (KS test, chi-squared) on input features
- **Requires**: Custom analysis container to extract features from text inputs

Documented as a complement to (not replacement for) CloudWatch-native set-size monitoring.

## CloudWatch Alarms

| Alarm | Metric | Condition | Action |
|-------|--------|-----------|--------|
| Drift alert | `SetSize` (Average, 1h) | > threshold (e.g., 2.0) | SNS → on-call |
| Coverage drop | `Coverage` (batch) | < target_coverage - 0.05 | SNS → recalibrate |
| Queue depth | `ApproximateNumberOfMessagesVisible` | > 100 | SNS → scale reviewers |
| Latency p99 | `PredictionLatencyMs` (p99) | > 1000ms | SNS → investigate endpoint |

## Batch Transform for Non-Gated Models

The POC uses real-time endpoint scoring for calibration because JumpStart gated models (Llama 3) have an API incompatibility with Batch Transform (`ModelDataSource` vs `ModelDataUrl`). For non-gated models (FLAN-T5, open-weight models with direct S3 artifacts), Batch Transform remains viable and more cost-effective for large calibration sets.
