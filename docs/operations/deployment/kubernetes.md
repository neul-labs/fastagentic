# Kubernetes Deployment

Deploy FastAgentic applications on Kubernetes with production-grade configuration.

## Quick Start with Helm

```bash
# Add repository
helm repo add fastagentic https://charts.fastagentic.dev
helm repo update

# Install
helm install my-agent fastagentic/runtime \
  --set image.repository=myregistry/my-agent \
  --set image.tag=v1.0.0 \
  --set auth.oidcIssuer=https://auth.company.com \
  --set durability.backend=postgres \
  --set durability.connectionString=postgresql://... \
  --set telemetry.enabled=true
```

## Helm Values

### values.yaml

```yaml
# Image configuration
image:
  repository: myregistry/my-agent
  tag: latest
  pullPolicy: IfNotPresent

# Replicas
replicaCount: 3

# Authentication
auth:
  oidcIssuer: https://auth.company.com
  oidcAudience: my-agent
  # Reference existing secret for OIDC client credentials
  existingSecret: my-agent-oidc

# Durability
durability:
  backend: postgres  # or redis, s3
  connectionString: ""  # Set via existingSecret
  existingSecret: my-agent-db

# Telemetry
telemetry:
  enabled: true
  serviceName: my-agent
  exporterEndpoint: http://otel-collector.monitoring:4318

# Resources
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilization: 70
  targetMemoryUtilization: 80

# Ingress
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
  hosts:
    - host: agent.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agent-tls
      hosts:
        - agent.company.com

# Service
service:
  type: ClusterIP
  port: 80
  targetPort: 8000

# Pod configuration
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"

# Health checks
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /health/startup
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 30

# Pod disruption budget
podDisruptionBudget:
  enabled: true
  minAvailable: 2

# Node affinity
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app.kubernetes.io/name: my-agent
          topologyKey: kubernetes.io/hostname

# Environment variables
env:
  - name: FASTAGENTIC_LOG_FORMAT
    value: json
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: llm-credentials
        key: openai-api-key
```

## Raw Kubernetes Manifests

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-agent
  labels:
    app.kubernetes.io/name: my-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app.kubernetes.io/name: my-agent
  template:
    metadata:
      labels:
        app.kubernetes.io/name: my-agent
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      serviceAccountName: my-agent
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: agent
          image: myregistry/my-agent:v1.0.0
          ports:
            - containerPort: 8000
              name: http
          env:
            - name: FASTAGENTIC_HOST
              value: "0.0.0.0"
            - name: FASTAGENTIC_PORT
              value: "8000"
            - name: FASTAGENTIC_OIDC_ISSUER
              value: "https://auth.company.com"
            - name: FASTAGENTIC_DURABLE_STORE
              valueFrom:
                secretKeyRef:
                  name: my-agent-secrets
                  key: database-url
            - name: FASTAGENTIC_OTEL_ENDPOINT
              value: "http://otel-collector.monitoring:4318"
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-credentials
                  key: openai-api-key
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 2000m
              memory: 2Gi
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app.kubernetes.io/name: my-agent
                topologyKey: kubernetes.io/hostname
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-agent
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      name: http
  selector:
    app.kubernetes.io/name: my-agent
```

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-agent
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: my-agent
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: my-agent
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-agent
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - agent.company.com
      secretName: agent-tls
  rules:
    - host: agent.company.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-agent
                port:
                  number: 80
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-agent-secrets
type: Opaque
stringData:
  database-url: postgresql://user:password@postgres:5432/fastagentic
---
apiVersion: v1
kind: Secret
metadata:
  name: llm-credentials
type: Opaque
stringData:
  openai-api-key: sk-...
  anthropic-api-key: sk-ant-...
```

## High Availability

### Multi-Zone Deployment

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app.kubernetes.io/name: my-agent
        topologyKey: topology.kubernetes.io/zone
```

### Database High Availability

Use managed databases or deploy with replication:

```yaml
# PostgreSQL with CloudNativePG
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: fastagentic-db
spec:
  instances: 3
  storage:
    size: 10Gi
```

## Monitoring

### ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-agent
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: my-agent
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

### PrometheusRule

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: my-agent-alerts
spec:
  groups:
    - name: fastagentic
      rules:
        - alert: HighErrorRate
          expr: rate(fastagentic_runs_failed_total[5m]) / rate(fastagentic_runs_total[5m]) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: High agent error rate
```

## Secrets Management

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: my-agent-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault
    kind: ClusterSecretStore
  target:
    name: my-agent-secrets
  data:
    - secretKey: database-url
      remoteRef:
        key: fastagentic/database
        property: url
    - secretKey: openai-api-key
      remoteRef:
        key: fastagentic/llm
        property: openai
```

## Next Steps

- [Serverless Deployment](serverless.md) - AWS Lambda, Cloud Run
- [Environment Variables](../configuration/environment-vars.md) - Configuration reference
- [Alerting](../observability/alerting.md) - Alert rules
