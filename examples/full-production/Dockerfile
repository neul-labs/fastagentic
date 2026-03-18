# Production Dockerfile for FastAgentic
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN uv sync --no-dev

# Copy application
COPY . .

# Run with gunicorn for production
CMD ["uv", "run", "uvicorn", "app:app.fastapi", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

EXPOSE 8000
