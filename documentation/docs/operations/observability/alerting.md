# Alerting Guide

Configure alerts for FastAgentic applications to detect issues before they impact users.

## Alert Categories

| Category | Severity | Response Time |
|----------|----------|---------------|
| Availability | Critical | Immediate |
| Performance | Warning | 15 minutes |
| Cost | Warning | 1 hour |
| Capacity | Info | Next business day |

## Prometheus Alert Rules

### prometheus-rules.yaml

```yaml
groups:
  - name: fastagentic-availability
    rules:
      - alert: FastAgenticDown
        expr: up{job="fastagentic"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "FastAgentic instance down"
          description: "{{ $labels.instance }} has been down for more than 1 minute"
          runbook: "https://docs.example.com/runbooks/fastagentic-down"

      - alert: HighErrorRate
        expr: |
          rate(fastagentic_requests_total{status=~"5.."}[5m])
          / rate(fastagentic_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.path }}"
          description: "Error rate is {{ $value | humanizePercentage }}"
          runbook: "https://docs.example.com/runbooks/high-error-rate"

      - alert: DurableStoreDown
        expr: fastagentic_durable_store_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Durable store unavailable"
          description: "Checkpoint storage is down"
          runbook: "https://docs.example.com/runbooks/durable-store-down"

  - name: fastagentic-performance
    rules:
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(fastagentic_request_duration_seconds_bucket[5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.path }}"
          description: "P95 latency is {{ $value | humanizeDuration }}"

      - alert: SlowCheckpoints
        expr: |
          histogram_quantile(0.95, rate(fastagentic_checkpoint_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow checkpoint writes"
          description: "P95 checkpoint time is {{ $value | humanizeDuration }}"

      - alert: HighRunDuration
        expr: |
          histogram_quantile(0.95, rate(fastagentic_run_duration_seconds_bucket[5m])) > 300
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Agent runs taking too long"
          description: "P95 run duration is {{ $value | humanizeDuration }}"

  - name: fastagentic-cost
    rules:
      - alert: HighCostPerHour
        expr: sum(rate(fastagentic_cost_usd_total[1h])) * 3600 > 100
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High hourly cost"
          description: "Hourly cost is ${{ $value | printf \"%.2f\" }}"

      - alert: TenantCostExceeded
        expr: |
          sum(increase(fastagentic_cost_usd_total[24h])) by (tenant) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tenant {{ $labels.tenant }} exceeded daily budget"
          description: "Daily cost is ${{ $value | printf \"%.2f\" }}"

      - alert: UnusualTokenUsage
        expr: |
          sum(rate(fastagentic_tokens_total[1h]))
          > 2 * sum(rate(fastagentic_tokens_total[1h] offset 1d))
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Unusual token usage spike"
          description: "Token usage 2x higher than yesterday"

  - name: fastagentic-capacity
    rules:
      - alert: HighConcurrency
        expr: sum(fastagentic_runs_active) / count(fastagentic_runs_active) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High concurrent run utilization"
          description: "{{ $value | humanizePercentage }} capacity used"

      - alert: RateLimitHits
        expr: sum(rate(fastagentic_rate_limit_hits_total[5m])) > 10
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "Users hitting rate limits"
          description: "{{ $value }} rate limit hits per second"

      - alert: QuotaNearLimit
        expr: fastagentic_quota_usage_ratio > 0.9
        for: 5m
        labels:
          severity: info
        annotations:
          summary: "User {{ $labels.user }} near quota limit"
          description: "{{ $value | humanizePercentage }} of quota used"
```

## Alertmanager Configuration

### alertmanager.yml

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/...'

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-warnings'
    - match:
        severity: info
      receiver: 'slack-info'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
        severity: critical

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#agent-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'slack-info'
    slack_configs:
      - channel: '#agent-info'
```

## Runbook Templates

### High Error Rate

**Alert:** `HighErrorRate`

**Symptoms:**
- HTTP 5xx responses > 5%
- Users reporting failures

**Diagnosis:**
1. Check application logs: `fastagentic tail --level ERROR`
2. Verify LLM provider status
3. Check durable store connectivity
4. Review recent deployments

**Resolution:**
1. If LLM provider down: Enable fallback model
2. If durable store down: See DurableStoreDown runbook
3. If deployment issue: Rollback
4. If unknown: Escalate to on-call engineer

### Durable Store Down

**Alert:** `DurableStoreDown`

**Symptoms:**
- Checkpoints failing
- Runs cannot resume
- New runs may fail

**Diagnosis:**
1. Check store health: `fastagentic inspect --config`
2. Verify network connectivity
3. Check store logs (Redis/Postgres)

**Resolution:**
1. If network issue: Check security groups, DNS
2. If store crashed: Restart store service
3. If disk full: Expand storage
4. Failover to backup if available

### High Cost

**Alert:** `HighCostPerHour`

**Symptoms:**
- Cost exceeding budget
- Possible abuse or bug

**Diagnosis:**
1. Identify high-cost users: Query `fastagentic_cost_usd_total` by tenant
2. Check for unusual patterns
3. Review recent changes

**Resolution:**
1. If abuse: Apply rate limits or block user
2. If bug: Fix and deploy
3. If legitimate growth: Adjust alerts, add capacity

## Integration Examples

### PagerDuty

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - routing_key: '<routing-key>'
        severity: '{{ .GroupLabels.severity }}'
        description: '{{ .GroupLabels.alertname }}'
        details:
          summary: '{{ .CommonAnnotations.summary }}'
          description: '{{ .CommonAnnotations.description }}'
          runbook: '{{ .CommonAnnotations.runbook }}'
```

### Slack

```yaml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: |
          *Summary:* {{ .CommonAnnotations.summary }}
          *Description:* {{ .CommonAnnotations.description }}
          *Runbook:* {{ .CommonAnnotations.runbook }}
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ .CommonAnnotations.runbook }}'
```

### OpsGenie

```yaml
receivers:
  - name: 'opsgenie'
    opsgenie_configs:
      - api_key: '<api-key>'
        message: '{{ .GroupLabels.alertname }}'
        priority: '{{ if eq .GroupLabels.severity "critical" }}P1{{ else }}P3{{ end }}'
```

## Next Steps

- [Metrics Reference](metrics.md) - Available metrics
- [Tracing](tracing.md) - Distributed tracing
- [Troubleshooting](../runbook/troubleshooting.md) - Common issues
