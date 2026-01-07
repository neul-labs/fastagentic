# FastAgentic Helm Charts

This directory contains Helm charts for deploying FastAgentic applications.

## Charts

| Chart | Description |
|-------|-------------|
| [runtime](./runtime) | FastAgentic runtime chart for deploying agentic applications |

## Usage

### Add the repository

```bash
helm repo add fastagentic https://github.com/neul-labs/fastagentic/tree/main/charts
helm repo update
```

### Install the runtime chart

```bash
helm install my-agent fastagentic/runtime \
  --set image.repository=myregistry/my-agent \
  --set image.tag=v1.0.0 \
  --set auth.oidcIssuer=https://auth.company.com \
  --set durability.backend=postgres \
  --set durability.connectionString=postgresql://user:pass@host:5432/db
```

### Using values.yaml

Create a `values.yaml` file:

```yaml
image:
  repository: myregistry/my-agent
  tag: v1.0.0

replicaCount: 3

auth:
  oidcIssuer: https://auth.company.com
  oidcAudience: my-agent

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: agent.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agent-tls
      hosts:
        - agent.company.com
```

Then install:

```bash
helm install my-agent fastagentic/runtime -f values.yaml
```

## Documentation

For full deployment documentation, see:
- [Kubernetes Deployment](../../docs/operations/deployment/kubernetes.md)
- [Operations Index](../../docs/operations/index.md)
