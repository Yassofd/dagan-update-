# ========================
# STAGE 1 - Build / Install dependencies
# ========================
FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency file to leverage Docker caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# ========================
# STAGE 2 - Final runtime image
# ========================
FROM public.ecr.aws/docker/library/python:3.11-slim AS runtime

# Set working directory
WORKDIR /app

# Copy app source code
COPY . .

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Add venv binaries to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose FastAPI port
EXPOSE 8080

# Default command to run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
