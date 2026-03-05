# CAMRIE Docker Images

Docker images for running CAMRIE MRI simulations in the cloud.

## Images

### Fargate Image (`Dockerfile.fargate`)

Full-featured image for AWS Fargate or local Docker usage.

```bash
# Build
docker build -f Dockerfile.fargate -t camrie:fargate .

# Run simulation
docker run -v /path/to/data:/data camrie:fargate \
    camrie-sim --rho /data/rho.nii.gz --t1 /data/t1.nii.gz \
    --normal 0 0 1 --num-slices 5 --output /data/output

# Run Streamlit (expose port 8501)
docker run -p 8501:8501 \
    -v /path/to/data:/data \
    camrie:fargate \
    streamlit run /app/camrie/streamlit_app.py
```

### Lambda Image (`Dockerfile.lambda`)

Minimal image for AWS Lambda (requires Lambda container support).

```bash
# Build
docker build -f Dockerfile.lambda -t camrie:lambda .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag camrie:lambda 123456789.dkr.ecr.us-east-1.amazonaws.com/camrie:lambda
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/camrie:lambda
```

## AWS Fargate Deployment

### 1. Create ECR Repository

```bash
aws ecr create-repository --repository-name camrie
```

### 2. Build and Push

```bash
# Get login
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -f Dockerfile.fargate -t camrie .

# Tag and push
docker tag camrie:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/camrie:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/camrie:latest
```

### 3. Create Task Definition

```json
{
  "family": "camrie-simulation",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "camrie",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/camrie:latest",
      "essential": true,
      "command": ["camrie-sim", "--config", "/data/config.json"],
      "mountPoints": [
        {
          "sourceVolume": "efs-data",
          "containerPath": "/data"
        }
      ]
    }
  ],
  "volumes": [
    {
      "name": "efs-data",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678"
      }
    }
  ]
}
```

### 4. Run Task

```bash
aws ecs run-task \
    --cluster your-cluster \
    --task-definition camrie-simulation \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CAMRIE_OUTPUT_DIR` | Output directory | `/tmp/camrie_output` |
| `CAMRIE_B0` | Magnetic field strength | `3.0` |
| `CAMRIE_GPU` | Use GPU | `false` |
| `CAMRIE_THREADS` | CPU threads | `4` |

## Volume Mounts

| Path | Description |
|------|-------------|
| `/data` | Input/output data directory |
| `/config` | Configuration files |

## Health Check

The Fargate image includes a health check endpoint when running Streamlit:

```bash
curl http://localhost:8501/_stcore/health
```
