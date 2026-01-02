# Serverless Deployment

Deploy FastAgentic on serverless platforms for auto-scaling and pay-per-use pricing.

## Platform Considerations

| Platform | Best For | Limitations |
|----------|----------|-------------|
| AWS Lambda | Event-driven, short tasks | 15min timeout, cold starts |
| Google Cloud Run | HTTP workloads, containers | 60min timeout |
| Azure Container Apps | Mixed workloads | Complexity |

**Recommendation:** Cloud Run for most FastAgentic workloads due to container support and longer timeouts.

## Google Cloud Run

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Cloud Run uses PORT env var
CMD exec fastagentic run --host 0.0.0.0 --port $PORT
```

### Deploy

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT/my-agent

# Deploy
gcloud run deploy my-agent \
  --image gcr.io/PROJECT/my-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "FASTAGENTIC_DURABLE_STORE=redis://..." \
  --set-env-vars "FASTAGENTIC_OTEL_ENDPOINT=..." \
  --set-secrets "OPENAI_API_KEY=openai-key:latest" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --min-instances 1 \
  --max-instances 100
```

### cloud-run.yaml

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: my-agent
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/PROJECT/my-agent
          resources:
            limits:
              cpu: "2"
              memory: 2Gi
          env:
            - name: FASTAGENTIC_DURABLE_STORE
              valueFrom:
                secretKeyRef:
                  name: agent-secrets
                  key: database-url
```

## AWS Lambda (with Adapter)

### Requirements

FastAgentic on Lambda requires the Mangum adapter:

```bash
pip install mangum
```

### handler.py

```python
from mangum import Mangum
from fastagentic import App, agent_endpoint
from fastagentic.adapters.pydanticai import PydanticAIAdapter

app = App(
    title="My Agent",
    durable_store=os.environ["DURABLE_STORE"],
)

@agent_endpoint(path="/agent", runnable=PydanticAIAdapter(agent))
async def run_agent(input: AgentInput) -> AgentOutput:
    pass

# Lambda handler
handler = Mangum(app.asgi(), lifespan="off")
```

### serverless.yml

```yaml
service: my-agent

provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  timeout: 300
  memorySize: 2048
  environment:
    DURABLE_STORE: ${ssm:/my-agent/durable-store}
    OPENAI_API_KEY: ${ssm:/my-agent/openai-key~true}

functions:
  agent:
    handler: handler.handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements

custom:
  pythonRequirements:
    dockerizePip: true
```

### Limitations on Lambda

- **15-minute timeout**: Long-running agents may fail
- **Cold starts**: First request latency (use provisioned concurrency)
- **No WebSocket**: SSE works, WebSocket doesn't
- **Stateless**: Requires external durable store

## Azure Container Apps

### Deploy

```bash
az containerapp create \
  --name my-agent \
  --resource-group rg-agents \
  --environment my-env \
  --image myregistry.azurecr.io/my-agent:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 2 \
  --memory 4Gi \
  --min-replicas 1 \
  --max-replicas 10 \
  --env-vars \
    FASTAGENTIC_DURABLE_STORE=secretref:database-url \
    OPENAI_API_KEY=secretref:openai-key
```

## Durable Store for Serverless

Serverless requires external durable storage:

### Redis (Recommended for serverless)

```bash
# AWS ElastiCache, GCP Memorystore, Azure Cache
export FASTAGENTIC_DURABLE_STORE=redis://your-redis-host:6379
```

### PostgreSQL

```bash
# AWS RDS, GCP Cloud SQL, Azure Database
export FASTAGENTIC_DURABLE_STORE=postgresql://user:pass@host:5432/db
```

### DynamoDB (AWS-native)

```bash
export FASTAGENTIC_DURABLE_STORE=dynamodb://table-name?region=us-east-1
```

## Cold Start Optimization

### Minimize Dependencies

```dockerfile
# Use slim image
FROM python:3.11-slim

# Only install required adapters
RUN pip install fastagentic[pydanticai]  # Not [all]
```

### Provisioned Concurrency (Lambda)

```yaml
functions:
  agent:
    handler: handler.handler
    provisionedConcurrency: 5  # Keep 5 instances warm
```

### Minimum Instances (Cloud Run)

```bash
gcloud run deploy my-agent --min-instances 1
```

## Streaming on Serverless

SSE streaming works on most serverless platforms:

```python
@agent_endpoint(path="/agent", runnable=adapter, stream=True)
async def run_agent(input: AgentInput) -> AgentOutput:
    pass
```

**Platform support:**
- Cloud Run: Full SSE support
- Lambda + API Gateway: Limited (consider Lambda URLs)
- Azure Container Apps: Full support

## Cost Optimization

### Right-size Resources

| Workload | Cloud Run | Lambda |
|----------|-----------|--------|
| Light chat | 1 CPU, 512MB | 512MB |
| Standard agent | 2 CPU, 2GB | 2GB |
| Heavy workflow | 4 CPU, 4GB | 4GB+ |

### Use Spot/Preemptible (Cloud Run)

```bash
gcloud run deploy my-agent \
  --execution-environment gen2 \
  --cpu-boost
```

## Next Steps

- [Docker Deployment](docker.md) - For dedicated infrastructure
- [Kubernetes Deployment](kubernetes.md) - For complex deployments
- [Environment Variables](../configuration/environment-vars.md) - Configuration
