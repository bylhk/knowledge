# Docker

Docker provides reproducible, isolated environments for training and serving ML models — ensuring consistency from local development to production.

---

## Dockerfile Structure

A well-structured Dockerfile follows a logical top-down order:

```dockerfile
# 1. Base image
FROM python:3.12-slim AS base

# 2. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 3. System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Application dependencies (cached layer)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application code (changes most frequently — last)
COPY src/ src/

# 6. Entrypoint
EXPOSE 8080
CMD ["python", "src/main.py"]
```

---

## Layer Caching

Docker caches each layer. Order instructions from least-frequently changed to most-frequently changed:

```
base image → system deps → pip install → application code
```

- Copy `requirements.txt` before copying source code — dependency installs are cached unless requirements change
- Avoid `COPY . .` early in the Dockerfile — any source code change invalidates all subsequent layers

---

## Multi-Stage Builds

Use multi-stage builds to keep final images small by separating build-time dependencies from runtime:

```dockerfile
# Build stage — includes compilers, build tools
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage — only what's needed to run
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
WORKDIR /app
COPY src/ src/
CMD ["python", "src/main.py"]
```

This avoids shipping compilers, headers, and build artefacts in the production image.

---

## Security

### Never bake credentials into image layers

Credentials in `ENV`, `ARG`, or `COPY` statements are permanently stored in the image layer history — even if deleted in a later layer.

```dockerfile
# ❌ Bad — token is stored in image history
ARG GIT_TOKEN
RUN pip install git+https://${GIT_TOKEN}@github.com/org/package.git

# ✅ Good — secret is mounted at build time, never persisted
RUN --mount=type=secret,id=git_token \
    GIT_TOKEN=$(cat /run/secrets/git_token) && \
    pip install git+https://${GIT_TOKEN}@github.com/org/package.git && \
    unset GIT_TOKEN
```

### Clear environment variables after use

```dockerfile
ENV GIT_TOKEN=
ENV GOOGLE_APPLICATION_CREDENTIALS=/dev/null
```

### Run as non-root

```dockerfile
RUN useradd --create-home appuser
USER appuser
```

### Scan images for vulnerabilities

Use tools like `trivy`, `grype`, or `docker scout` in CI to catch known CVEs before deployment.

---

## Image Size Optimisation

### Why minimise image size?

Every megabyte in a Docker image has a compounding cost across the entire lifecycle:

| Impact area | How large images hurt |
|-------------|----------------------|
| Build time | More layers to process, longer CI pipelines |
| Push / pull | Slower uploads to and downloads from the registry — especially across regions |
| Cold start | Containers take longer to start — critical for autoscaling and serverless (e.g. AWS Lambda, Cloud Run) |
| Storage cost | Registry storage is billed per GB; multiply by number of tags and environments |
| Network cost | Data transfer between registry and compute adds up at scale |
| Security surface | More packages = more potential CVEs to patch |
| Rollback speed | Smaller images mean faster rollbacks when a deployment fails |

For ML serving images, cold start latency directly affects autoscaling responsiveness — a 3GB image can add 30–60 seconds to scale-up time versus a 500MB image.

### How to minimise image size

| Technique | Impact | Example |
|-----------|--------|--------|
| Use `-slim` or `-alpine` base images | Reduces base from ~900MB to ~150MB | `python:3.12-slim` instead of `python:3.12` |
| Multi-stage builds | Removes compilers, headers, build tools from final image | See [Multi-Stage Builds](#multi-stage-builds) |
| `--no-cache-dir` on pip install | Avoids storing pip download cache in the layer | `pip install --no-cache-dir -r requirements.txt` |
| `rm -rf /var/lib/apt/lists/*` after apt-get | Removes package index cache (~30MB) | Combine with `apt-get install` in one `RUN` |
| `.dockerignore` file | Prevents unnecessary files from entering the build context | See below |
| Pin exact versions | Avoids pulling unexpected large transitive dependencies | `numpy==1.26.4` not `numpy>=1.22` |
| Combine `RUN` commands | Fewer layers, and intermediate files can be cleaned in the same layer | Chain with `&&` |
| Remove unnecessary system packages | Don't install `vim`, `curl`, `wget` in production images | Only install what the app needs |
| Use CPU-only ML libraries | GPU builds of PyTorch/TensorFlow add 2–5GB | `--index-url https://download.pytorch.org/whl/cpu` |
| Download models at runtime | Keep model weights out of the image | Pull from S3/GCS at container startup |

### Combining RUN commands

```dockerfile
# ❌ Bad — leaves apt cache in an earlier layer (still counts toward image size)
RUN apt-get update
RUN apt-get install -y build-essential
RUN rm -rf /var/lib/apt/lists/*

# ✅ Good — single layer, cache cleaned in the same step
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
```

### Checking image size

```bash
# View image size
docker images model-serve

# Inspect layer-by-layer size breakdown
docker history model-serve:latest

# Use dive for interactive layer analysis
dive model-serve:latest
```

### .dockerignore

```
.git/
__pycache__/
*.pyc
.env
notebooks/
data/
tests/
*.md
```

---

## ML-Specific Considerations

### CPU vs GPU images

Maintain separate Dockerfiles or build targets for CPU and GPU:

```dockerfile
# CPU
FROM python:3.12-slim AS cpu
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu
RUN pip install torch
```

Use CI variables or build args to select the target:

```bash
docker build --target gpu -t model-serve:gpu .
```

### Model artefacts

- Do **not** bake large model files into the image — download them at container startup from cloud storage (S3, GCS)
- This keeps images small and allows model updates without rebuilding
- If the model must be in the image (e.g. edge deployment), use a dedicated layer near the end so code changes don't invalidate it

```dockerfile
# Model layer — only changes when model is retrained
COPY artifacts/model.pkl /app/artifacts/

# Code layer — changes frequently
COPY src/ /app/src/
```

### Health checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/live || exit 1
```

---

## Docker Compose for Local Development

Use `docker-compose.yml` to replicate the production stack locally:

```yaml
services:
  api:
    build: .
    ports:
      - "8080:8080"
    env_file: env/dev.env
    volumes:
      - ./src:/app/src  # hot reload
    depends_on:
      - redis
      - db

  redis:
    image: redis:7-alpine

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: features
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
```

---

## Container Runtime Best Practices

| Practice | Why |
|----------|-----|
| `PYTHONUNBUFFERED=1` | Ensures logs appear immediately — critical for debugging |
| `PYTHONDONTWRITEBYTECODE=1` | Avoids `.pyc` files cluttering the container |
| One process per container | Simplifies scaling, logging, and failure isolation |
| Graceful shutdown handling | Catch `SIGTERM` to finish in-flight requests before exiting |
| Read-only filesystem where possible | Reduces attack surface |
| Resource limits (CPU/memory) | Prevents a single container from starving others |

---

## Image Tagging Strategy

- Tag images with the Git commit SHA for traceability: `model-serve:abc123f`
- Use semantic version tags for releases: `model-serve:1.2.0`
- Avoid relying solely on `latest` — it is mutable and ambiguous
- Tag with environment for promotion tracking: `model-serve:1.2.0-staging`
