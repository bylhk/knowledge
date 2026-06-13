# Knowledge Base

A structured reference for MLOps, software engineering, and machine learning best practices.

---

## Folder Structure

```
knowledge/
├── ai-engineering/
│   ├── agents/             Agentic patterns, memory, tool use, MCP
│   ├── evaluation/         Measuring retrieval and generation quality
│   ├── llm_core/           Core LLM techniques: prompts, output, customisation, fine-tuning
│   ├── production/         Deployment, efficiency, cost, security, observability
│   ├── rag/                RAG pipeline: ingestion → quality → chunking → embedding → retrieval
│   └── tooling/            Frameworks, UI, and package references
│       └── packages/       Package comparison tables
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

### AI-Engineering

#### agents/ — Agentic Architectures

| File | Contents |
|------|----------|
| [ai-engineering/agents/agentic_architectures.md](ai-engineering/agents/agentic_architectures.md) | Agent patterns: ReAct, plan-and-execute, tool calling, multi-agent, reflection, human-in-the-loop, LangGraph |
| [ai-engineering/agents/memory_and_conversation.md](ai-engineering/agents/memory_and_conversation.md) | Memory types: buffer, window, summary, entity, vector-backed, LangGraph store, session persistence |
| [ai-engineering/agents/model_context_protocol.md](ai-engineering/agents/model_context_protocol.md) | MCP standard: building servers, tools/resources/prompts, transports (stdio/SSE), configuration, testing |

#### evaluation/ — Quality Measurement

| File | Contents |
|------|----------|
| [ai-engineering/evaluation/retrieval_evaluation.md](ai-engineering/evaluation/retrieval_evaluation.md) | Retrieval metrics: precision@k, recall@k, MRR, NDCG, hit rate — with evaluation harnesses |
| [ai-engineering/evaluation/agent_response_evaluation.md](ai-engineering/evaluation/agent_response_evaluation.md) | Response quality: LLM-as-judge, RAGAS, DeepEval, ROUGE, BERTScore, trajectory eval, safety eval |

#### llm_core/ — Core LLM Techniques

| File | Contents |
|------|----------|
| [ai-engineering/llm_core/llm_customisation_strategies.md](ai-engineering/llm_core/llm_customisation_strategies.md) | When to use prompting vs RAG vs fine-tuning vs compound systems |
| [ai-engineering/llm_core/prompt_engineering_patterns.md](ai-engineering/llm_core/prompt_engineering_patterns.md) | Prompt patterns: system prompts, few-shot, CoT, ToT, ReAct, self-consistency, decomposition |
| [ai-engineering/llm_core/prompt_optimisation.md](ai-engineering/llm_core/prompt_optimisation.md) | Automated prompt improvement: DSPy, APE, PromptFoo, TextGrad, OPRO |
| [ai-engineering/llm_core/structured_output.md](ai-engineering/llm_core/structured_output.md) | Forcing structured LLM output: Guidance, Outlines, LangChain, Instructor, provider-native JSON mode |
| [ai-engineering/llm_core/fine_tuning_lora.md](ai-engineering/llm_core/fine_tuning_lora.md) | Parameter-efficient fine-tuning: LoRA, QLoRA, GLoRA, Unsloth, LLaMA-Factory, DeepSpeed |

#### production/ — Production Systems

| File | Contents |
|------|----------|
| [ai-engineering/production/deployment_and_serving.md](ai-engineering/production/deployment_and_serving.md) | Serving patterns: FastAPI, vLLM, Docker, Kubernetes, CI/CD, A/B testing, canary, serverless |
| [ai-engineering/production/llm_efficiency.md](ai-engineering/production/llm_efficiency.md) | Inference optimisation: quantisation (GPTQ/AWQ/GGUF), KV cache, vLLM, speculative decoding, Flash Attention |
| [ai-engineering/production/cost_management.md](ai-engineering/production/cost_management.md) | Cost control: model routing, caching, token reduction, budget tracking, spend monitoring |
| [ai-engineering/production/observability.md](ai-engineering/production/observability.md) | LLM observability: Langfuse, LangSmith, Phoenix, OpenTelemetry, custom logging, dashboards |
| [ai-engineering/production/guardrails.md](ai-engineering/production/guardrails.md) | Input/output protection: NeMo Guardrails, Guardrails AI, LLM-as-judge, prompt injection defence |
| [ai-engineering/production/llm_security_red_teaming.md](ai-engineering/production/llm_security_red_teaming.md) | Security: prompt injection, jailbreaking, data extraction, red teaming methodology, defence-in-depth |

#### rag/ — RAG Pipeline

| File | Contents |
|------|----------|
| [ai-engineering/rag/document_ingestion.md](ai-engineering/rag/document_ingestion.md) | PDF/image/scan extraction: Tesseract, Unstructured, Vision LLMs, PyMuPDF, LlamaParse, Marker |
| [ai-engineering/rag/data_quality_post_ingestion.md](ai-engineering/rag/data_quality_post_ingestion.md) | Post-extraction validation: encoding fixes, OCR quality scoring, language detection, deduplication |
| [ai-engineering/rag/chunking_strategies.md](ai-engineering/rag/chunking_strategies.md) | 9 chunking methods: recursive, semantic, agentic, structure-aware, parent-child, proposition, late chunking, contextual headers, multi-vector |
| [ai-engineering/rag/data_quality_post_chunking.md](ai-engineering/rag/data_quality_post_chunking.md) | Chunk validation: length checks, deduplication, coherence scoring, distribution analysis |
| [ai-engineering/rag/embedding_methods.md](ai-engineering/rag/embedding_methods.md) | Embedding models and techniques: cloud APIs, local models, code-specific, hybrid dense+sparse, fine-tuned, multi-modal |
| [ai-engineering/rag/vector_databases.md](ai-engineering/rag/vector_databases.md) | Vector DB operations: ChromaDB, Qdrant, Pinecone, FAISS, Weaviate — insert, query, filter, update |
| [ai-engineering/rag/retrieval_methods.md](ai-engineering/rag/retrieval_methods.md) | Retrieval strategies: similarity, MMR, multi-query, hybrid, reranking, compression, HyDE, step-back |

#### tooling/ — Frameworks & Packages

| File | Contents |
|------|----------|
| [ai-engineering/tooling/langchain_stack.md](ai-engineering/tooling/langchain_stack.md) | LangChain ecosystem architecture diagram and data flow |
| [ai-engineering/tooling/ai_ui.md](ai-engineering/tooling/ai_ui.md) | Chat UI frameworks: Chainlit, Gradio, Streamlit, Mesop, Panel, Open WebUI |
| [ai-engineering/tooling/packages/langchain_ecosystem.md](ai-engineering/tooling/packages/langchain_ecosystem.md) | LangChain-specific package table: all categories with descriptions |
| [ai-engineering/tooling/packages/useful_packages.md](ai-engineering/tooling/packages/useful_packages.md) | General AI/LLM package comparison table across all categories |

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
