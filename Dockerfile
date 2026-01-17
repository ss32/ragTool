# RAG Tool Dockerfile
# Build: docker build -t rag-tool .
# Run:   docker run -it --rm -v ~/.rag_tool:/root/.rag_tool -p 7860:7860 --add-host=host.docker.internal:host-gateway rag-tool gui --host 0.0.0.0

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY .claude/ ./.claude/

# Create data directory
RUN mkdir -p /root/.rag_tool

# Expose GUI port
EXPOSE 7860

# Default Ollama host (can be overridden with -e OLLAMA_HOST=...)
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Default command - show help
ENTRYPOINT ["python", "rag_tool.py"]
CMD ["--help"]
