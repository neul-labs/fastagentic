# Troubleshooting Guide

Common issues and solutions for FastAgentic deployments.

## Diagnostic Commands

```bash
# Check application health
curl http://localhost:8000/health/ready

# View configuration
fastagentic config show

# Tail logs
fastagentic tail --level ERROR

# Inspect runs
fastagentic inspect --runs --status failed

# Check connectivity
fastagentic diagnose
```

## Common Issues

### Application Won't Start

**Symptoms:**
- Container exits immediately
- "Address already in use" error
- Import errors

**Diagnosis:**
```bash
# Check logs
docker logs my-agent

# Verify port availability
lsof -i :8000

# Test imports
python -c "from fastagentic import App"
```

**Solutions:**

1. **Port conflict:**
   ```bash
   export FASTAGENTIC_PORT=8001
   ```

2. **Missing dependencies:**
   ```bash
   pip install fastagentic[pydanticai,langgraph]
   ```

3. **Configuration error:**
   ```bash
   fastagentic config validate
   ```

---

### Authentication Failures

**Symptoms:**
- 401 Unauthorized responses
- "Invalid token" errors
- JWKS fetch failures

**Diagnosis:**
```bash
# Check OIDC configuration
echo $FASTAGENTIC_OIDC_ISSUER

# Verify JWKS endpoint
curl $FASTAGENTIC_OIDC_ISSUER/.well-known/openid-configuration

# Test token validation
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/health/ready
```

**Solutions:**

1. **Wrong issuer:**
   ```bash
   export FASTAGENTIC_OIDC_ISSUER=https://correct-issuer.com
   ```

2. **Audience mismatch:**
   ```bash
   export FASTAGENTIC_OIDC_AUDIENCE=correct-audience
   ```

3. **Clock skew:**
   ```bash
   # Sync time on container
   ntpdate -s time.google.com
   ```

4. **Network issues reaching JWKS:**
   ```bash
   # Allow egress to OIDC provider
   # Check DNS resolution
   ```

---

### Durable Store Connection Failed

**Symptoms:**
- "Connection refused" to Redis/Postgres
- Checkpoints not saving
- Runs fail to resume

**Diagnosis:**
```bash
# Test Redis
redis-cli -h $REDIS_HOST ping

# Test Postgres
psql $DATABASE_URL -c "SELECT 1"

# Check FastAgentic connection
fastagentic diagnose --store
```

**Solutions:**

1. **Wrong connection string:**
   ```bash
   # Redis
   export FASTAGENTIC_DURABLE_STORE=redis://host:6379/0

   # Postgres
   export FASTAGENTIC_DURABLE_STORE=postgresql://user:pass@host:5432/db
   ```

2. **Network/firewall:**
   ```bash
   # Check connectivity
   nc -zv redis-host 6379
   ```

3. **Authentication:**
   ```bash
   # Redis with auth
   export FASTAGENTIC_DURABLE_STORE=redis://:password@host:6379
   ```

4. **TLS required:**
   ```bash
   # Redis TLS
   export FASTAGENTIC_DURABLE_STORE=rediss://host:6379

   # Postgres TLS
   export FASTAGENTIC_DURABLE_STORE=postgresql://...?sslmode=require
   ```

---

### LLM Provider Errors

**Symptoms:**
- 429 Rate limit errors
- 401 Invalid API key
- Timeout errors

**Diagnosis:**
```bash
# Check API key is set
echo $OPENAI_API_KEY | head -c 10

# Test API directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Solutions:**

1. **Invalid API key:**
   ```bash
   export OPENAI_API_KEY=sk-correct-key
   ```

2. **Rate limiting:**
   - Implement retry with backoff
   - Add rate limits on your side
   - Request higher limits from provider

3. **Timeouts:**
   ```bash
   # Increase timeout
   export FASTAGENTIC_LLM_TIMEOUT=120
   ```

---

### High Latency

**Symptoms:**
- P95 latency > 10s
- Slow checkpoint writes
- Streaming delays

**Diagnosis:**
```bash
# Check metrics
curl http://localhost:8000/metrics | grep duration

# Profile a request
fastagentic profile /chat --input '{"message":"test"}'

# Check store latency
fastagentic diagnose --store --latency
```

**Solutions:**

1. **Slow LLM responses:**
   - Use faster model (gpt-4o-mini vs gpt-4)
   - Reduce max_tokens
   - Use streaming

2. **Slow checkpoints:**
   - Use Redis instead of Postgres for hot path
   - Reduce checkpoint frequency
   - Check store latency/network

3. **High load:**
   - Scale horizontally
   - Add rate limiting
   - Check resource limits

---

### Streaming Not Working

**Symptoms:**
- SSE connection drops
- No events received
- Events arrive in batches

**Diagnosis:**
```bash
# Test SSE directly
curl -N http://localhost:8000/chat/stream \
  -H "Accept: text/event-stream" \
  -d '{"message":"test"}'

# Check proxy configuration
# Nginx/ALB may buffer SSE
```

**Solutions:**

1. **Proxy buffering:**
   ```nginx
   # Nginx
   proxy_buffering off;
   proxy_cache off;
   proxy_read_timeout 300s;
   ```

   ```yaml
   # AWS ALB
   # Use WebSocket or increase idle timeout
   ```

2. **Missing headers:**
   ```python
   # Ensure endpoint has stream=True
   @agent_endpoint(path="/chat", stream=True)
   ```

3. **Timeout:**
   ```bash
   export FASTAGENTIC_STREAM_TIMEOUT=300
   ```

---

### Out of Memory

**Symptoms:**
- OOMKilled in Kubernetes
- Container restarts
- Slow performance before crash

**Diagnosis:**
```bash
# Check memory usage
kubectl top pods

# Check for memory leaks
fastagentic profile --memory

# Review checkpoint sizes
fastagentic inspect --checkpoints --size
```

**Solutions:**

1. **Increase limits:**
   ```yaml
   resources:
     limits:
       memory: 4Gi
   ```

2. **Reduce checkpoint retention:**
   ```bash
   export FASTAGENTIC_MAX_CHECKPOINTS=50
   export FASTAGENTIC_CHECKPOINT_TTL=1h
   ```

3. **Optimize adapter:**
   - Clear conversation history periodically
   - Limit context window size

---

### MCP Not Working

**Symptoms:**
- `/mcp/schema` returns 404
- MCP clients can't connect
- Tools not registered

**Diagnosis:**
```bash
# Check MCP is enabled
echo $FASTAGENTIC_MCP_ENABLED

# View MCP schema
curl http://localhost:8000/mcp/schema

# Check stdio transport
fastagentic run --stdio
```

**Solutions:**

1. **MCP disabled:**
   ```bash
   export FASTAGENTIC_MCP_ENABLED=true
   ```

2. **Wrong path:**
   ```bash
   # Check configured path
   export FASTAGENTIC_MCP_PATH=/mcp
   ```

3. **Schema not generated:**
   - Ensure decorators are properly applied
   - Check for import errors

---

## Health Check Failures

### Liveness Probe Failing

```bash
# Check endpoint
curl http://localhost:8000/health/live

# Common causes:
# - Application crashed
# - Deadlock
# - Infinite loop
```

### Readiness Probe Failing

```bash
# Check endpoint
curl http://localhost:8000/health/ready

# Common causes:
# - Database not connected
# - Dependent service down
# - Still initializing
```

## Getting Help

1. **Check logs:** `fastagentic tail --level DEBUG`
2. **Run diagnostics:** `fastagentic diagnose`
3. **Search issues:** [GitHub Issues](https://github.com/fastagentic/fastagentic/issues)
4. **Community:** Discord/Slack channel

## Next Steps

- [Alerting](../observability/alerting.md) - Set up alerts
- [Metrics](../observability/metrics.md) - Monitor performance
- [Tracing](../observability/tracing.md) - Debug with traces
