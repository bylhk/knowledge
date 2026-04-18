# Pipeline Reference

A pipeline is a sequence of steps that transforms data, trains a model, or serves predictions. Different pipeline tools are suited to different scales, environments, and latency requirements.

---

## Contents

| File | What it covers |
|------|---------------|
| [readme.md](readme.md) | This overview and decision guide |
| [kubeflow.md](kubeflow.md) | Kubeflow Pipelines on Vertex AI — ML training DAGs |
| [spark.md](spark.md) | Apache Spark — distributed data processing pipelines |
| [beam.md](beam.md) | Apache Beam — unified batch and streaming pipelines |
| [endpoint.md](endpoint.md) | Online serving endpoints — FastAPI, Triton, TorchServe, BentoML |
| [feature-store.md](feature-store.md) | Feature stores — Feast, Vertex AI, SageMaker, Tecton |
| [workflow-orchestration.md](workflow-orchestration.md) | Workflow orchestrators — Airflow, Prefect, Dagster |
| [streaming.md](streaming.md) | Real-time streaming pipelines — Kafka, Kinesis, Pub/Sub |

---

## Pipeline Types

```
Data pipeline        → ingest, transform, aggregate, store
Training pipeline    → load data, engineer features, train, evaluate, register
Serving pipeline     → receive request, fetch features, predict, respond
Streaming pipeline   → consume events, process in real time, emit results
```

---

## Decision Guide

### Training pipeline tool

```
Running on GCP / Vertex AI?              ──yes──→  Kubeflow Pipelines
Running on AWS SageMaker?                ──yes──→  SageMaker Pipelines
Simple sequential steps, any cloud?      ──yes──→  Prefect / Dagster
Complex DAG with many dependencies?      ──yes──→  Airflow / Dagster
Notebook-first, quick iteration?         ──yes──→  Metaflow
```

### Data processing tool

```
Data in warehouse (TB+)?                 ──yes──→  SQL (BigQuery, Redshift)
Distributed processing (GB–PB)?          ──yes──→  Spark
Unified batch + streaming?               ──yes──→  Apache Beam
In-memory tabular (GB)?                  ──yes──→  Polars / DuckDB
Numerical array operations?              ──yes──→  NumPy
```

### Serving tool

```
Simple REST API, Python model?           ──yes──→  FastAPI
High-throughput, multi-model, GPU?       ──yes──→  Triton Inference Server
PyTorch model, production serving?       ──yes──→  TorchServe
Package model as service quickly?        ──yes──→  BentoML
Managed endpoint, no infra?              ──yes──→  Vertex AI Endpoint / SageMaker Endpoint
```

### Streaming tool

```
High-throughput event streaming?         ──yes──→  Kafka
AWS-native streaming?                    ──yes──→  Kinesis
GCP-native streaming?                    ──yes──→  Pub/Sub
Lightweight, managed?                    ──yes──→  Confluent Cloud / AWS MSK
```

---

## ML Pipeline Lifecycle

```
Raw data (warehouse / lake)
    ↓
Feature pipeline  →  Feature store  ←──────────────────────┐
    ↓                                                        │
Training pipeline                                           │
    ↓                                                        │
Model registry                                              │
    ↓                                                        │
Serving pipeline  ←  Online request  →  Feature fetch ──────┘
    ↓
Response + logs
    ↓
Monitoring pipeline  →  Drift alerts  →  Retraining trigger
```

Every arrow is a pipeline boundary. Each boundary should be explicit, versioned, and observable.
