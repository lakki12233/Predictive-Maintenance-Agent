# Cloud Deployment Guide

**Optional deployment reference for production environments.**

> 💡 **Note:** For the take-home assignment, local Docker deployment is sufficient. This guide is for future production use.

---

## Quick Overview

The Predictive Maintenance Agent API can be deployed to any cloud platform that supports Docker containers:

- **AWS ECS Fargate** - Recommended for auto-scaling
- **GCP Cloud Run** - Simpler, pay-per-use
- **Azure Container Instances** - Quick setup
- **Kubernetes** - For existing K8s infrastructure

---

## Example: AWS ECS Fargate

### 1. Push Docker Image to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name predictive-agent-api

# Build and push
docker build -t predictive-agent-api -f docker/Dockerfile .
docker tag predictive-agent-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/predictive-agent-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/predictive-agent-api:latest
```

### 2. Create ECS Service

```bash
# Create cluster
aws ecs create-cluster --cluster-name predictive-agent-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service with 2 replicas
aws ecs create-service \
  --cluster predictive-agent-cluster \
  --service-name predictive-agent-api \
  --task-definition predictive-agent-api-task \
  --desired-count 2 \
  --launch-type FARGATE
```

### 3. Add Load Balancer (Optional)

Use AWS Application Load Balancer to distribute traffic across containers.

---

## Estimated Costs

| Component | Monthly Cost |
|-----------|-------------|
| ECS Fargate (2 tasks) | ~$50 |
| Load Balancer | ~$20 |
| ECR Storage | ~$1 |
| **Total** | **~$70/month** |

---

## Environment Variables

Set these for production:

```bash
RUST_MODEL_TYPE=mobilenet  # or clip
LOG_LEVEL=info
MAX_WORKERS=4
```

---

## Health Checks

The `/health` endpoint is ready for load balancer health checks:

```bash
curl http://localhost:8000/health
```

---

## Scaling

Configure auto-scaling based on CPU/memory:

- **Min replicas:** 2
- **Max replicas:** 10
- **Target CPU:** 70%

---


