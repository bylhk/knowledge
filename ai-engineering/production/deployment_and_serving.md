# Deployment & Serving

## Overview

Deploying LLM applications to production requires choosing inference infrastructure, building APIs, containerising services, managing model versions, and implementing traffic management (A/B testing, canary releases). This card covers patterns for serving both API-based (cloud) and self-hosted model deployments.

---

## Serving Architectures

| Architecture | Description | Best For |
|--------------|-------------|----------|
| API passthrough | Thin wrapper around cloud LLM APIs | Simple chatbots, MVPs |
| Self-hosted inference | Run open models on your infra | Privacy, cost control, customisation |
| Hybrid | Cloud APIs + local models for different tasks | Cost optimisation, fallback |
| Edge/on-device | Quantised models on user hardware | Offline, latency-sensitive |

---

## Pattern 1: FastAPI Inference Server

### Basic LLM API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

app = FastAPI(title="LLM Service", version="1.0.0")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class QueryRequest(BaseModel):
    query: str = Field(max_length=5000)
    temperature: float = Field(default=0.7, ge=0, le=1)
    max_tokens: int = Field(default=1000, le=4000)

class QueryResponse(BaseModel):
    answer: str
    model: str
    tokens_used: int

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = llm.invoke(
            request.query,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return QueryResponse(
            answer=response.content,
            model="gemini-2.0-flash",
            tokens_used=response.usage_metadata.get("total_tokens", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### With RAG Pipeline

```python
from fastapi import FastAPI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize expensive resources on startup."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    app.state.vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    app.state.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    yield
    # Cleanup

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
async def ask(query: str):
    # Retrieve
    docs = app.state.vectorstore.similarity_search(query, k=5)
    context = "\n".join(d.page_content for d in docs)

    # Generate
    response = app.state.llm.invoke(
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context:"
    )

    return {
        "answer": response.content,
        "sources": [d.metadata.get("source") for d in docs],
    }
```

---

## Pattern 2: vLLM Production Serving

High-throughput self-hosted inference with OpenAI-compatible API.

### Docker Deployment

```dockerfile
FROM vllm/vllm-openai:latest

# Model is downloaded at runtime or pre-baked
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"

CMD ["--model", "${MODEL_NAME}", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.9"]
```

```bash
docker run --gpus all -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --port 8000
```

### Client Usage (OpenAI-Compatible)

```python
from openai import OpenAI

# Connect to vLLM server — same API as OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    temperature=0.7,
    max_tokens=500,
)
print(response.choices[0].message.content)
```

---

## Pattern 3: Containerised Deployment

### Dockerfile for LLM App

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (App + Vector DB)

```yaml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - CHROMA_HOST=chromadb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - chromadb
      - redis

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  chroma_data:
```

---

## Pattern 4: Model Versioning & A/B Testing

### A/B Testing with Feature Flags

```python
import random
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    model_id: str
    weight: float  # Traffic percentage (0-1)

# A/B test configuration
MODEL_VARIANTS = [
    ModelConfig(name="control", model_id="gemini-2.0-flash-lite", weight=0.5),
    ModelConfig(name="treatment", model_id="gemini-2.0-flash", weight=0.5),
]

def select_model(user_id: str = None) -> ModelConfig:
    """Select model variant for A/B testing."""
    if user_id:
        # Deterministic assignment by user (consistent experience)
        hash_val = hash(user_id) % 100 / 100
    else:
        hash_val = random.random()

    cumulative = 0.0
    for variant in MODEL_VARIANTS:
        cumulative += variant.weight
        if hash_val < cumulative:
            return variant

    return MODEL_VARIANTS[-1]

@app.post("/query")
async def query(request: QueryRequest):
    variant = select_model(request.user_id)

    llm = ChatGoogleGenerativeAI(model=variant.model_id)
    response = llm.invoke(request.query)

    # Log for analysis
    log_experiment(
        variant=variant.name,
        query=request.query,
        response=response.content,
        latency_ms=elapsed,
        tokens=response.usage_metadata["total_tokens"],
    )

    return {"answer": response.content, "variant": variant.name}
```

### Canary Deployment

```python
class CanaryRouter:
    """Gradually shift traffic from old to new model."""

    def __init__(self, old_model: str, new_model: str, canary_pct: float = 0.05):
        self.old_model = old_model
        self.new_model = new_model
        self.canary_pct = canary_pct
        self.error_count = 0
        self.total_canary = 0

    def get_model(self) -> str:
        if random.random() < self.canary_pct:
            self.total_canary += 1
            return self.new_model
        return self.old_model

    def report_error(self, model: str):
        if model == self.new_model:
            self.error_count += 1
            # Auto-rollback if error rate too high
            if self.total_canary > 10 and self.error_count / self.total_canary > 0.1:
                self.canary_pct = 0.0
                print("ROLLBACK: Canary error rate exceeded 10%")

    def promote(self, new_pct: float):
        """Increase canary traffic (gradual rollout)."""
        self.canary_pct = min(new_pct, 1.0)
```

---

## Pattern 5: Auto-Scaling & Load Balancing

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
        - name: app
          image: registry.example.com/llm-service:v1.2.0
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
          env:
            - name: GOOGLE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-secrets
                  key: google-api-key
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## Pattern 6: CI/CD for LLM Apps

### GitLab CI Pipeline

```yaml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - uv sync
    - uv run pytest tests/ -v
    - uv run ruff check src/

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy-staging:
  stage: deploy
  script:
    - kubectl set image deployment/llm-service app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/llm-service --timeout=300s
  environment:
    name: staging

deploy-production:
  stage: deploy
  script:
    - kubectl set image deployment/llm-service app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - kubectl rollout status deployment/llm-service --timeout=300s
  environment:
    name: production
  when: manual  # Require manual approval for prod
```

---

## Pattern 7: Serverless Deployment

### AWS Lambda (Short Requests)

```python
# handler.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def handler(event, context):
    body = json.loads(event["body"])
    query = body["query"]

    response = llm.invoke(query)

    return {
        "statusCode": 200,
        "body": json.dumps({"answer": response.content}),
    }
```

### Google Cloud Run

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
gcloud run deploy llm-service \
    --source . \
    --port 8080 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --set-env-vars GOOGLE_API_KEY=$GOOGLE_API_KEY
```

---

## Pattern 8: Model Registry & Versioning

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelVersion:
    name: str
    version: str
    model_id: str
    deployed_at: datetime
    config: dict
    status: str  # "active", "canary", "deprecated"

class ModelRegistry:
    """Track deployed model versions."""

    def __init__(self):
        self.versions: dict[str, ModelVersion] = {}
        self.active_version: str | None = None

    def register(self, name: str, version: str, model_id: str, config: dict):
        key = f"{name}:{version}"
        self.versions[key] = ModelVersion(
            name=name,
            version=version,
            model_id=model_id,
            deployed_at=datetime.now(),
            config=config,
            status="registered",
        )
        return key

    def promote(self, key: str):
        """Promote a version to active."""
        if self.active_version:
            self.versions[self.active_version].status = "deprecated"
        self.versions[key].status = "active"
        self.active_version = key

    def get_active(self) -> ModelVersion:
        return self.versions[self.active_version]

    def rollback(self, to_key: str):
        """Rollback to a previous version."""
        self.promote(to_key)
```

---

## Comparison: Deployment Options

| Option | Cold Start | Scaling | Cost Model | GPU | Best For |
|--------|-----------|---------|------------|-----|----------|
| Cloud Run / Lambda | Yes | Auto | Per-request | No* | API wrappers, low traffic |
| ECS / GKE | No | Auto (HPA) | Per-instance | Yes | Production, steady traffic |
| vLLM on GPU | No | Manual | Per-GPU-hour | Yes | Self-hosted inference |
| Ollama | No | None | Free (local) | Optional | Development, testing |
| Vertex AI Endpoints | No | Auto | Per-node-hour | Yes | Managed GPU serving |
| Replicate / Modal | Minimal | Auto | Per-second | Yes | Burst GPU workloads |

## Health Checks & Monitoring

```python
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

startup_time = datetime.now()

@app.get("/health")
async def health():
    return {"status": "healthy", "uptime_seconds": (datetime.now() - startup_time).seconds}

@app.get("/ready")
async def readiness():
    """Check if all dependencies are available."""
    checks = {
        "vectorstore": check_vectorstore(),
        "llm_api": check_llm_connectivity(),
        "cache": check_redis(),
    }
    all_healthy = all(checks.values())
    return {"ready": all_healthy, "checks": checks}
```

## Best Practices

- Always have a `/health` endpoint — load balancers and orchestrators need it
- Use connection pooling for vector DBs and caches
- Set timeouts on all external calls (LLM APIs, vector search)
- Cache responses (semantic or exact) to reduce cost and latency
- Log every request with trace IDs for debugging
- Use async handlers (FastAPI + `async def`) for I/O-bound LLM calls
- Rate limit per user/API key to prevent abuse and cost overruns
- Separate model serving from application logic (microservice pattern)
- Pin model versions — don't auto-update in production without testing
- Monitor: latency (p50/p95), error rate, token usage, cost per request
- Have a rollback plan — model quality can degrade silently
- Use canary releases for any model or prompt change in production
