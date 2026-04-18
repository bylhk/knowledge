# Online Serving Endpoints

An online serving endpoint receives a request, runs inference, and returns a prediction — typically within milliseconds. Two decisions dominate the design: which communication protocol to use, and how to scale the deployment.

---

## Communication Protocols

### REST (HTTP/JSON)

The default choice. Human-readable, universally supported, easy to debug.

```
Client → HTTP POST /predict
         Content-Type: application/json
         Body: {"features": [[0.5, 0.3, 0.8]]}

Server → HTTP 200
         Body: {"scores": [0.82]}
```

```python
import httpx

response = httpx.post(
    "http://serving:8080/predict",
    json={"features": [[0.5, 0.3, 0.8]]},
    timeout=5.0,
)
scores = response.json()["scores"]
```

| Pros | Cons |
|------|------|
| Universal — any language, any client | JSON serialisation overhead — slow for large payloads |
| Human-readable — easy to debug with curl | HTTP/1.1 is request-response only — no streaming |
| Firewall-friendly | Higher latency than binary protocols |
| Native browser support | No schema enforcement by default |

**When to use:** internal microservices, external APIs, any client that cannot use gRPC, latency > 10ms is acceptable.

---

### gRPC (HTTP/2 + Protocol Buffers)

gRPC uses HTTP/2 for multiplexed connections and Protocol Buffers for binary serialisation. It is 5–10x faster than REST for the same payload and supports streaming.

**Define the service contract in a `.proto` file:**

```protobuf
// scoring.proto
syntax = "proto3";

package scoring;

service ScoringService {
  rpc Predict (PredictRequest) returns (PredictResponse);
  rpc PredictStream (stream PredictRequest) returns (stream PredictResponse);
}

message PredictRequest {
  repeated float features = 1;   // flat feature vector
  string request_id       = 2;
}

message PredictResponse {
  float  score      = 1;
  string request_id = 2;
}
```

```bash
# Generate Python stubs from the proto file
python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  scoring.proto
```

**Server:**

```python
import grpc
from concurrent import futures
import scoring_pb2
import scoring_pb2_grpc
import numpy as np
import joblib

class ScoringServicer(scoring_pb2_grpc.ScoringServiceServicer):
    """
    gRPC servicer for the churn scoring model.

    Notes
    -----
    Model is loaded once at init — reused across all requests.
    Binary serialisation via protobuf is 5–10x faster than JSON.
    """

    def __init__(self, model_path: str) -> None:
        self._model = joblib.load(model_path)

    def Predict(
        self,
        request: scoring_pb2.PredictRequest,
        context: grpc.ServicerContext,
    ) -> scoring_pb2.PredictResponse:
        features = np.array(request.features, dtype=np.float32).reshape(1, -1)
        score    = float(self._model.predict_proba(features)[0, 1])
        return scoring_pb2.PredictResponse(
            score=score,
            request_id=request.request_id,
        )


def serve(model_path: str, port: int = 50051) -> None:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length",    50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    scoring_pb2_grpc.add_ScoringServiceServicer_to_server(
        ScoringServicer(model_path), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
```

**Client:**

```python
import grpc
import scoring_pb2
import scoring_pb2_grpc

channel = grpc.insecure_channel("serving:50051")
stub    = scoring_pb2_grpc.ScoringServiceStub(channel)

response = stub.Predict(scoring_pb2.PredictRequest(
    features=[0.5, 0.3, 0.8],
    request_id="req_001",
))
print(response.score)
```

| Pros | Cons |
|------|------|
| 5–10x faster than REST — binary serialisation, HTTP/2 multiplexing | Not human-readable — harder to debug |
| Strongly typed — schema enforced by the `.proto` contract | Requires proto compilation step |
| Bidirectional streaming — client and server can stream simultaneously | Limited browser support (needs gRPC-Web proxy) |
| Connection multiplexing — many requests over one TCP connection | More setup than REST |
| Code generation for all major languages | |

**When to use:** internal service-to-service communication, high-throughput inference (> 1000 RPS), large payloads (embeddings, image tensors), latency-sensitive paths.

---

### WebSocket

WebSocket provides a persistent, full-duplex connection — the server can push results to the client without the client polling.

```python
from fastapi import FastAPI, WebSocket
import numpy as np
import json

app = FastAPI()

@app.websocket("/stream")
async def stream_predictions(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data     = await websocket.receive_json()
            features = np.array(data["features"], dtype=np.float32)
            score    = float(model.predict_proba(features.reshape(1, -1))[0, 1])
            await websocket.send_json({"score": score, "request_id": data["request_id"]})
    except Exception:
        await websocket.close()
```

**When to use:** real-time dashboards, live scoring feeds, interactive applications where the server pushes updates.

---

### Protocol Comparison

| | REST (HTTP/JSON) | gRPC (HTTP/2 + Protobuf) | WebSocket |
|---|-----------------|--------------------------|----------|
| Serialisation | JSON (text) | Protobuf (binary) | JSON or binary |
| Speed | Baseline | 5–10x faster | Similar to REST |
| Streaming | ❌ (HTTP/1.1) | ✅ Bidirectional | ✅ Full-duplex |
| Schema enforcement | Optional (Pydantic) | ✅ Proto contract | ❌ |
| Browser support | ✅ Native | ⚠️ Needs gRPC-Web | ✅ Native |
| Debuggability | ✅ curl, Postman | ⚠️ grpcurl, Postman | ⚠️ |
| Best for | External APIs, simple services | Internal high-throughput | Real-time push |

---

## Scalable Deployment

### Horizontal scaling — stateless replicas

ML serving containers must be stateless — no request state stored in memory between calls. This allows any number of identical replicas to run behind a load balancer.

```
Load Balancer
    ├── replica-1 (model loaded)
    ├── replica-2 (model loaded)
    └── replica-3 (model loaded)
```

Each replica loads the model independently at startup. Shared state (cache, feature store) lives outside the replica in Redis or a database.

### Autoscaling on Kubernetes

```yaml
# Horizontal Pod Autoscaler — scale on CPU or custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: scoring-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: scoring-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
    - type: Pods
      pods:
        metric:
          name: requests_per_second   # custom metric from Prometheus
        target:
          type: AverageValue
          averageValue: "100"         # scale up when > 100 RPS per pod
```

```yaml
# Deployment — rolling update, resource limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scoring-service
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # add 1 new pod before removing old
      maxUnavailable: 0    # never reduce below desired count during update
  template:
    spec:
      containers:
        - name: scoring
          image: gcr.io/my-project/scoring:v1.2.0
          resources:
            requests:
              cpu: "1"
              memory: "2Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /live
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
```

### Request batching — improve throughput without changing clients

For CPU-bound models, batching multiple concurrent requests into a single `predict_proba` call significantly improves throughput. Clients send individual requests; the server accumulates them into a batch.

```python
import asyncio
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class BatchAccumulator:
    """
    Accumulate concurrent requests into batches for efficient inference.

    Waits up to `max_wait_ms` or until `max_batch_size` requests arrive,
    then runs a single model call for the entire batch.
    """

    def __init__(self, model, max_batch_size: int = 64, max_wait_ms: float = 10.0) -> None:
        self._model         = model
        self._max_batch     = max_batch_size
        self._max_wait      = max_wait_ms / 1000.0
        self._queue: list   = []
        self._lock          = asyncio.Lock()
        self._event         = asyncio.Event()

    async def predict(self, features: np.ndarray) -> float:
        future = asyncio.get_event_loop().create_future()
        async with self._lock:
            self._queue.append((features, future))
            if len(self._queue) >= self._max_batch:
                self._event.set()

        await asyncio.wait_for(future, timeout=5.0)
        return future.result()

    async def run_loop(self) -> None:
        while True:
            await asyncio.wait_for(
                self._event.wait(), timeout=self._max_wait
            )
            async with self._lock:
                batch = self._queue[:self._max_batch]
                self._queue = self._queue[self._max_batch:]
                self._event.clear()

            if not batch:
                continue

            features_batch = np.stack([f for f, _ in batch])
            scores         = self._model.predict_proba(features_batch)[:, 1]

            for (_, future), score in zip(batch, scores):
                future.set_result(float(score))
```

### Caching — skip inference for repeated requests

```python
import redis
import hashlib
import json
import numpy as np

class CachedScorer:
    """
    Cache prediction results in Redis.
    Identical feature vectors return the cached score without running the model.
    """

    def __init__(self, model, redis_client: redis.Redis, ttl_seconds: int = 300) -> None:
        self._model  = model
        self._redis  = redis_client
        self._ttl    = ttl_seconds

    def _cache_key(self, features: np.ndarray) -> str:
        return "score:" + hashlib.md5(features.tobytes()).hexdigest()

    def predict(self, features: np.ndarray) -> float:
        key    = self._cache_key(features)
        cached = self._redis.get(key)
        if cached is not None:
            return float(cached)

        score = float(self._model.predict_proba(features.reshape(1, -1))[0, 1])
        self._redis.setex(key, self._ttl, str(score))
        return score
```

### Blue/green deployment — zero-downtime model updates

```
Load Balancer
    ├── blue  (model v11) ← 100% traffic
    └── green (model v12) ← 0% traffic (warming up)

After validation:
    ├── blue  (model v11) ← 0% traffic (kept for rollback)
    └── green (model v12) ← 100% traffic
```

```yaml
# Kubernetes service — switch traffic by updating the selector
apiVersion: v1
kind: Service
metadata:
  name: scoring-service
spec:
  selector:
    app: scoring
    version: green   # change from blue to green to switch traffic
  ports:
    - port: 8080
```

### Canary deployment — gradual traffic shift

```yaml
# Istio VirtualService — route 5% to canary
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: scoring-vs
spec:
  http:
    - route:
        - destination:
            host: scoring-stable
          weight: 95
        - destination:
            host: scoring-canary
          weight: 5
```

---

## Serving Tool Comparison

| Tool | Best for | Latency | Throughput | GPU support |
|------|---------|---------|-----------|-------------|
| **FastAPI** | Custom logic, Python models | Low | Medium | Via PyTorch/TF |
| **Triton Inference Server** | Multi-model, GPU, high throughput | Very low | Very high | Native |
| **TorchServe** | PyTorch models, production serving | Low | High | Native |
| **BentoML** | Package any model quickly | Low | Medium–High | Via PyTorch/TF |
| **Ray Serve** | Distributed, model composition | Low | Very high | Native |
| **Vertex AI Endpoint** | Managed, no infra, GCP | Low | Auto-scaling | Yes |
| **SageMaker Endpoint** | Managed, no infra, AWS | Low | Auto-scaling | Yes |

---

## FastAPI

The default choice for Python ML models with custom pre/postprocessing logic.

### Request and response validation with Pydantic v2

Define all request and response schemas as Pydantic `BaseModel` classes. FastAPI reads these to validate incoming JSON, generate OpenAPI docs, and serialise responses — all automatically.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np
import joblib
import os

app = FastAPI()


class PredictRequest(BaseModel):
    entity_id: str
    features:  list[float] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def must_be_finite(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) for x in v):
            raise ValueError("features must all be finite — no NaN or Inf")
        return v


class PredictResponse(BaseModel):
    entity_id:     str
    score:         float = Field(ge=0.0, le=1.0)
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
    code:   int


# Load model once at startup — never per request
class ModelServer:
    model         = None
    model_version = "unknown"

    @classmethod
    def load(cls, path: str, version: str) -> None:
        cls.model         = joblib.load(path)
        cls.model_version = version


@app.on_event("startup")
async def startup() -> None:
    ModelServer.load(
        path=os.getenv("MODEL_PATH", "/artifacts/model.pkl"),
        version=os.getenv("MODEL_VERSION", "unknown"),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(request: PredictRequest) -> PredictResponse:
    if ModelServer.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array(request.features, dtype=np.float32).reshape(1, -1)
    score    = float(ModelServer.model.predict_proba(features)[0, 1])

    return PredictResponse(
        entity_id=request.entity_id,
        score=score,
        model_version=ModelServer.model_version,
    )


@app.get("/live")
async def liveness() -> dict:
    return {"status": "ok"}
```

FastAPI automatically:
- Validates the incoming JSON against `PredictRequest` — returns HTTP 422 with a clear error if validation fails
- Serialises the return value using `PredictResponse` — no manual `jsonify` or `dict` construction
- Generates `/docs` (Swagger UI) and `/openapi.json` from the Pydantic schemas

### Deployment with Gunicorn + Uvicorn

```dockerfile
CMD ["gunicorn", "main:app",
     "--workers", "1",
     "--worker-class", "uvicorn.workers.UvicornWorker",
     "--bind", "0.0.0.0:8080",
     "--timeout", "120"]
```

---

## Triton Inference Server

NVIDIA Triton is the standard for high-throughput, multi-model GPU serving. It supports TensorRT, ONNX, PyTorch, TensorFlow, and Python backends in a single server.

### Model repository structure

```
model_repository/
├── churn_model/
│   ├── config.pbtxt          # model configuration
│   └── 1/
│       └── model.onnx        # model version 1
├── embedding_model/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── ensemble_pipeline/
    ├── config.pbtxt          # ensemble: embedding → churn_model
    └── 1/
        └── (no model file — ensemble is defined in config)
```

### config.pbtxt

```protobuf
name: "churn_model"
backend: "onnxruntime"
max_batch_size: 64

input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [ -1, 128 ]   # -1 = dynamic batch dimension
  }
]

output [
  {
    name: "scores"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 5000   # wait up to 5ms to form a batch
}

instance_group [
  { count: 2, kind: KIND_GPU }   # 2 model instances on GPU
]
```

### Dynamic batching

Triton automatically groups individual requests into batches before running inference — dramatically improving GPU utilisation without changing client code.

```
Client A sends 1 request  ─┐
Client B sends 1 request  ─┤→ Triton batches → GPU inference (batch=32) → responses
Client C sends 1 request  ─┘
...
```

### Python client

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

features = np.random.rand(1, 128).astype(np.float32)
inputs   = [httpclient.InferInput("features", features.shape, "FP32")]
inputs[0].set_data_from_numpy(features)

outputs   = [httpclient.InferRequestedOutput("scores")]
response  = client.infer("churn_model", inputs, outputs=outputs)
score     = response.as_numpy("scores")
```

---

## TorchServe

TorchServe is PyTorch's native model serving framework. It packages a model and its handler into a `.mar` archive and serves it via REST or gRPC.

### Package a model

```bash
# Create a model archive
torch-model-archiver \
  --model-name churn_model \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.pt \
  --handler handler.py \
  --export-path model_store/
```

### Custom handler

```python
from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np

class ChurnHandler(BaseHandler):
    """
    Custom TorchServe handler for the churn scoring model.

    Notes
    -----
    preprocess  → converts raw request JSON to a tensor
    inference   → runs the model forward pass
    postprocess → converts tensor output to a JSON-serialisable list
    """

    def preprocess(self, data: list) -> torch.Tensor:
        features = [d.get("body") or d.get("data") for d in data]
        arr      = np.array(features, dtype=np.float32)
        return torch.from_numpy(arr)

    def inference(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data: torch.Tensor) -> list:
        return data.numpy().tolist()
```

### Start the server

```bash
torchserve \
  --start \
  --model-store model_store/ \
  --models churn_model=churn_model.mar \
  --ts-config config.properties
```

---

## BentoML

BentoML packages any Python model (sklearn, XGBoost, PyTorch, ONNX) into a self-contained service called a Bento, deployable to any cloud or Kubernetes.

```python
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Save model to BentoML model store
bentoml.sklearn.save_model("churn_model", trained_model)

# Define the service
svc = bentoml.Service("churn_scoring", runners=[
    bentoml.sklearn.get("churn_model:latest").to_runner()
])

churn_runner = bentoml.sklearn.get("churn_model:latest").to_runner()

@svc.api(input=NumpyNdarray(dtype="float32", shape=(-1, 128)), output=NumpyNdarray())
async def predict(features: np.ndarray) -> np.ndarray:
    return await churn_runner.predict.async_run(features)
```

```bash
# Serve locally
bentoml serve service:svc --reload

# Build a Bento (self-contained deployable)
bentoml build

# Deploy to cloud
bentoml deploy churn_scoring:latest --platform aws-lambda
```

---

## Ray Serve

Ray Serve enables composing multiple models into a single serving pipeline with automatic scaling per component.

```python
import ray
from ray import serve
import numpy as np

ray.init()
serve.start()

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 0.5})
class EmbeddingModel:
    def __init__(self):
        self.model = load_embedding_model()

    async def __call__(self, text: str) -> np.ndarray:
        return self.model.encode(text)

@serve.deployment(num_replicas=4)
class ScoringModel:
    def __init__(self):
        self.model = load_scoring_model()

    async def __call__(self, embedding: np.ndarray) -> float:
        return float(self.model.predict_proba(embedding.reshape(1, -1))[0, 1])

@serve.deployment
class Pipeline:
    def __init__(self, embedder, scorer):
        self.embedder = embedder
        self.scorer   = scorer

    async def __call__(self, text: str) -> dict:
        embedding = await self.embedder.remote(text)
        score     = await self.scorer.remote(embedding)
        return {"score": score}

# Compose and deploy
pipeline = Pipeline.bind(EmbeddingModel.bind(), ScoringModel.bind())
serve.run(pipeline)
```

---

## Managed Endpoints

### Vertex AI Endpoint (GCP)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="europe-west2")

# Deploy a model from the registry to an endpoint
model    = aiplatform.Model("projects/my-project/locations/europe-west2/models/123")
endpoint = aiplatform.Endpoint.create(display_name="churn-endpoint")

endpoint.deploy(
    model=model,
    deployed_model_display_name="churn-v12",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=10,
    traffic_percentage=100,
)

# Predict
response = endpoint.predict(instances=[{"features": [0.1, 0.5, 0.3]}])
```

### SageMaker Endpoint (AWS)

```python
import boto3

sagemaker = boto3.client("sagemaker", region_name="eu-west-1")

# Create endpoint config
sagemaker.create_endpoint_config(
    EndpointConfigName="churn-config-v12",
    ProductionVariants=[{
        "VariantName":          "primary",
        "ModelName":            "churn-model-v12",
        "InstanceType":         "ml.m5.large",
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1.0,
    }],
)

# Create or update endpoint
sagemaker.create_endpoint(
    EndpointName="churn-endpoint",
    EndpointConfigName="churn-config-v12",
)
```

---

## Serving Patterns

### Synchronous (request-response)

```
Client → POST /predict → model → response (< 100ms)
```

Best for: interactive applications, real-time personalisation.

### Asynchronous (queue-based)

```
Client → enqueue request → worker pulls → model → write result → client polls
```

Best for: long-running inference (large models, video), decoupling producers from consumers.

### Batch prediction (pre-computed)

```
Scheduler → batch job → model → write to feature store → client reads
```

Best for: when predictions can be pre-computed, latency is not critical.

### Streaming prediction

```
Event stream → consumer → model → emit result → downstream consumer
```

Best for: real-time fraud detection, anomaly detection, live recommendations.

---

## Rules

- Load models at startup, never per request
- Always expose a `/live` health check endpoint — required for load balancer health probes
- Set request timeouts on every external call — never wait indefinitely
- Use dynamic batching (Triton) or async batching (Ray Serve) for GPU models — individual requests waste GPU capacity
- Export PyTorch models to ONNX for production serving — ONNX Runtime is faster than PyTorch for inference
- Version your model endpoints — never update in-place; use blue/green or canary deployment
