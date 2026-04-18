# Knowledge Base

A structured reference for MLOps, software engineering, and machine learning best practices.

---

## Folder Structure

```
knowledge/
├── cicd/                   CI/CD pipeline design and configuration
├── dashboard/              Observability, metrics, and dashboards
├── data/                   Data processing tools, SQL, dtypes, file formats, and config validation
├── docker/                 Docker and Kaniko best practices
├── documentation/          Code documentation, docstring standards, and GenAI tools
├── environment/            Local and cross-platform development environments
├── git/                    Git workflow, commits, and merge requests
├── language/               Language selection guide per module type
├── machine-learning/
│   ├── algorithm/          Algorithm reference by domain
│   └── concept/            ML pipeline best practices (11 files)
├── optimisation/           Python performance, profiling, and Cython
├── package/                Python package structure, Pydantic models
├── pipeline/               Data, training, serving, and streaming pipelines
└── semantic-build/         Semantic versioning and automated releases
```

---

## Index

### CI/CD

| File | Contents |
|------|---------|
| [cicd/readme.md](cicd/readme.md) | Pipeline stages (lint → build → train → evaluate → deploy → monitor), CI variables vs config files, environment promotion |
| [cicd/configuration.md](cicd/configuration.md) | Layered config strategy, path-based rules, ad-hoc CI variable shortcuts, image tagging, artefact management |

---

### Dashboard & Observability

| File | Contents |
|------|---------|
| [dashboard/readme.md](dashboard/readme.md) | Pre-aggregation pattern, delta load, multi-level aggregation with statistical tests (t-test, z-test, PSI), centralised pipeline logs, cloud cost labelling, model performance monitoring |

---

### Data

| File | Contents |
|------|---------|
| [data/readme.md](data/readme.md) | Tool selection: SQL → Spark → Cython → NumPy → Polars → Pandas → Python, decision framework, performance comparison |
| [data/sql.md](data/sql.md) | High-performance SQL: schema design, query efficiency, code reuse (table functions, scalar functions, dynamic SQL), anti-patterns |
| [data/data-types.md](data/data-types.md) | NumPy dtypes, float32 vs float64, zero-copy views, NumPy ↔ PyTorch memory sharing |
| [data/hive-partition.md](data/hive-partition.md) | Hive partitioning, Parquet vs Feather, cross-platform reads (BigQuery, Spark, PyArrow, Athena), partition pitfalls |

---

### Docker

| File | Contents |
|------|---------|
| [docker/readme.md](docker/readme.md) | Dockerfile structure, layer caching, multi-stage builds, security, image size optimisation, ML-specific patterns |
| [docker/kaniko.md](docker/kaniko.md) | Daemonless CI builds, secret handling, layer caching with `--cache-repo`, troubleshooting |

---

### Documentation

| File | Contents |
|------|---------|
| [documentation/readme.md](documentation/readme.md) | Module docstrings, NumPy-style function docstrings, inline comments, naming conventions, import organisation, Python 3.12 type hints |
| [documentation/web-generation.md](documentation/web-generation.md) | MkDocs + mkdocstrings setup, Sphinx + autodoc, docstring → HTML mapping, CI/CD publishing |
| [documentation/genai-assisted-development.md](documentation/genai-assisted-development.md) | Amazon Q Developer in the IDE, code review prompts, documentation generation prompts, prompt engineering skills |

---

### Environment

| File | Contents |
|------|---------|
| [environment/readme.md](environment/readme.md) | Cross-platform runner (`--local\|docker\|gcp\|aws`), venv / uv setup, Docker local development, environment variable management via shell exports and direnv |

---

### Git

| File | Contents |
|------|---------|
| [git/readme.md](git/readme.md) | `.gitignore` for ML repos, pre-commit hooks (black, isort, ruff, detect-secrets), conventional commits, branch strategy |
| [git/merge-request.md](git/merge-request.md) | MR size guidelines, description template, review priorities, ML-specific checklist, merge strategies |

---

### Language Selection

| File | Contents |
|------|---------|
| [language/readme.md](language/readme.md) | Language-to-module mapping (SQL, Python, Go, Rust, C++, HCL, Bash, JS/TS), decision framework, anti-patterns |

---

### Machine Learning — Algorithms

| File | Contents |
|------|---------|
| [machine-learning/algorithm/readme.md](machine-learning/algorithm/readme.md) | Quick selection guide across all domains |
| [machine-learning/algorithm/computer-vision.md](machine-learning/algorithm/computer-vision.md) | VGG, ResNet, EfficientNet, MobileNet, ViT, ConvNeXt, YOLOv8/v10, DETR, YOLO-World, Mask R-CNN, U-Net, SAM/SAM2, MoViNet, VideoMAE, multimodal models, data augmentation |
| [machine-learning/algorithm/nlp.md](machine-learning/algorithm/nlp.md) | Word2Vec, GloVe, Transformer, RoPE, BERT, GPT, open-weight LLMs (LLaMA, Mistral, Qwen), LoRA/QLoRA, DeepSeek MoE, reasoning models, RAG, agentic AI |
| [machine-learning/algorithm/generative-ai.md](machine-learning/algorithm/generative-ai.md) | LLM comparison, generation parameters, advanced RAG, agentic frameworks, FLUX/Stable Diffusion, ControlNet, video generation, evaluation metrics |
| [machine-learning/algorithm/rlhf-dpo.md](machine-learning/algorithm/rlhf-dpo.md) | RLHF (PPO), DPO with full implementation, GRPO (DeepSeek-R1), SimPO, β tuning |
| [machine-learning/algorithm/graph-networks.md](machine-learning/algorithm/graph-networks.md) | Matrix factorisation, Node2Vec, GCN, GraphSAGE, GAE, GNN tasks, off-policy evaluation (IPS, Doubly Robust) |

### Machine Learning — Concepts

| File | Contents |
|------|---------|
| [machine-learning/concept/readme.md](machine-learning/concept/readme.md) | Overview, core principles, anti-patterns |
| [machine-learning/concept/data.md](machine-learning/concept/data.md) | Train/val/test splits, temporal and group-aware splits, data leakage, feature engineering, class imbalance, data validation |
| [machine-learning/concept/training.md](machine-learning/concept/training.md) | Reproducibility, cross-validation (stratified, group, time-series), Bayesian HPT, early stopping, pipeline structure |
| [machine-learning/concept/evaluation.md](machine-learning/concept/evaluation.md) | Metric selection, baselines, calibration, threshold selection, in-training evaluation with TensorBoard, offline vs online, champion/challenger |
| [machine-learning/concept/model-design.md](machine-learning/concept/model-design.md) | Simple-first, feature selection, regularisation, monotonic constraints, SHAP interpretability, complexity vs maintainability |
| [machine-learning/concept/production.md](machine-learning/concept/production.md) | Model versioning, input validation, fallback hierarchy, batch vs online serving, caching, latency management, model loading |
| [machine-learning/concept/monitoring.md](machine-learning/concept/monitoring.md) | Three-layer monitoring (infra → model → KPIs), PSI, KS test, z-test on acceptance rate, retraining triggers, structured log fields |
| [machine-learning/concept/memory-and-io.md](machine-learning/concept/memory-and-io.md) | Parquet vs Feather, file size guidance, batch loading with PyArrow, streaming IterableDataset, batch evaluation accumulator, batch prediction writer |
| [machine-learning/concept/testing.md](machine-learning/concept/testing.md) | pytest fixtures, parametrize, mocking, ML-specific tests (data quality, model output, monotonicity, leakage), FastAPI endpoint testing |
| [machine-learning/concept/model-registry.md](machine-learning/concept/model-registry.md) | Model registry patterns, lineage tracking, MLflow (tracking, registry, promotion), Vertex AI Model Registry, SageMaker Model Registry |
| [machine-learning/concept/experiment-tracking.md](machine-learning/concept/experiment-tracking.md) | MLflow tracking, W&B, HPT with nested runs, notebook best practices, notebook-to-production transition |

---

### Optimisation

| File | Contents |
|------|---------|
| [optimisation/readme.md](optimisation/readme.md) | 7-level hierarchy (algorithm → cache → NumPy → memory → concurrency → Cython → rewrite), `numpy.vectorize`, avoid unnecessary copies, asyncio vs multiprocessing |
| [optimisation/profiling.md](optimisation/profiling.md) | cProfile, line_profiler, py-spy (flame graphs), timeit, memory_profiler, tracemalloc, profiling workflow |
| [optimisation/cython.md](optimisation/cython.md) | Setup, typed declarations, memory views, `nogil` + `prange`, compiler directives, annotated HTML, Numba vs Cython |

---

### Package & Code Structure

| File | Contents |
|------|---------|
| [package/readme.md](package/readme.md) | Function vs class, public vs private methods, abstract base classes, pipeline functions, modularisation with shared dependencies, unit testing with dependency injection, `pyproject.toml`, `src/` layout, `__init__.py` public API |

---

### Pipeline

| File | Contents |
|------|---------|
| [pipeline/readme.md](pipeline/readme.md) | Overview, decision guide (training tool, data tool, serving tool, streaming tool), ML pipeline lifecycle diagram |
| [pipeline/kubeflow.md](pipeline/kubeflow.md) | Kubeflow Pipelines on Vertex AI — components, pipeline DAGs, caching, resource config, conditional execution |
| [pipeline/spark.md](pipeline/spark.md) | Apache Spark — Spark SQL vs DataFrame API, feature engineering, batch prediction with `pandas_udf`, performance tuning, Databricks |
| [pipeline/beam.md](pipeline/beam.md) | Apache Beam — batch processing with DoFn + BatchElements (100x speedup), unified batch + streaming, windowing, Dataflow runner |
| [pipeline/endpoint.md](pipeline/endpoint.md) | REST vs gRPC vs WebSocket comparison, scalable deployment (HPA, request batching, caching, blue/green, canary), FastAPI, Triton, TorchServe, BentoML, Ray Serve |
| [pipeline/feature-store.md](pipeline/feature-store.md) | Training-serving skew, point-in-time join, Feast, Vertex AI Feature Store, SageMaker Feature Store |
| [pipeline/workflow-orchestration.md](pipeline/workflow-orchestration.md) | Airflow (DAGs, sensors, XCom), Prefect (flows, tasks, schedules), Dagster (asset-based pipelines) |
| [pipeline/streaming.md](pipeline/streaming.md) | Kafka (producer/consumer), real-time ML inference, streaming feature computation, Kinesis, Pub/Sub, delivery guarantees |

---

### Semantic Build

| File | Contents |
|------|---------|
| [semantic-build/readme.md](semantic-build/readme.md) | SemVer (MAJOR.MINOR.PATCH), conventional commits → version bumps, application vs model versioning |
| [semantic-build/git-update.md](semantic-build/git-update.md) | Commitizen, semantic-release, manual release steps, CHANGELOG format, CI integration |

---

## Cross-Reference Map

Topics that appear in multiple files — use the primary file for the full treatment:

| Topic | Primary | Also in |
|-------|---------|---------|
| Delta load / watermark | [data/sql.md](data/sql.md) | [dashboard/readme.md](dashboard/readme.md), [machine-learning/concept/memory-and-io.md](machine-learning/concept/memory-and-io.md) |
| Mid-layer aggregation | [dashboard/readme.md](dashboard/readme.md) | [data/sql.md](data/sql.md) |
| Hive partitioning | [data/hive-partition.md](data/hive-partition.md) | [machine-learning/concept/memory-and-io.md](machine-learning/concept/memory-and-io.md) |
| Parquet / Feather formats | [machine-learning/concept/memory-and-io.md](machine-learning/concept/memory-and-io.md) | [data/hive-partition.md](data/hive-partition.md) |
| Drift detection (PSI, KS) | [machine-learning/concept/monitoring.md](machine-learning/concept/monitoring.md) | [dashboard/readme.md](dashboard/readme.md) |
| DPO / RLHF / GRPO | [machine-learning/algorithm/rlhf-dpo.md](machine-learning/algorithm/rlhf-dpo.md) | [machine-learning/algorithm/nlp.md](machine-learning/algorithm/nlp.md) |
| NumPy zero-copy | [data/data-types.md](data/data-types.md) | [optimisation/readme.md](optimisation/readme.md) |
| CI variables vs config files | [cicd/readme.md](cicd/readme.md) | [environment/readme.md](environment/readme.md), [cicd/configuration.md](cicd/configuration.md) |
| Conventional commits | [git/readme.md](git/readme.md) | [semantic-build/readme.md](semantic-build/readme.md), [cicd/configuration.md](cicd/configuration.md) |
| Champion / challenger | [machine-learning/concept/evaluation.md](machine-learning/concept/evaluation.md) | [machine-learning/concept/production.md](machine-learning/concept/production.md) |
| Feature store | [pipeline/feature-store.md](pipeline/feature-store.md) | [machine-learning/concept/production.md](machine-learning/concept/production.md) |
| Batch prediction | [machine-learning/concept/memory-and-io.md](machine-learning/concept/memory-and-io.md) | [pipeline/beam.md](pipeline/beam.md), [pipeline/spark.md](pipeline/spark.md) |
| Serving endpoint patterns | [pipeline/endpoint.md](pipeline/endpoint.md) | [machine-learning/concept/production.md](machine-learning/concept/production.md) |
