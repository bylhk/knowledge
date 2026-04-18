# Kubeflow Pipelines

Kubeflow Pipelines (KFP) is a platform for building and running ML workflows as directed acyclic graphs (DAGs) on Kubernetes. On GCP it runs as **Vertex AI Pipelines** — a fully managed service with no cluster management required.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Component** | A self-contained step — a Python function or container that takes inputs and produces outputs |
| **Pipeline** | A DAG of components connected by their inputs and outputs |
| **Artifact** | A typed output passed between components (Dataset, Model, Metrics, HTML) |
| **Run** | A single execution of a pipeline with specific input parameters |
| **Experiment** | A named group of runs for comparison |

---

## Component Types

### Lightweight Python component

The simplest component — a decorated Python function compiled to a container at submission time. No Dockerfile needed.

```python
from kfp.v2 import dsl
from kfp.v2.dsl import Dataset, Model, Output, Input, Metrics

@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["scikit-learn==1.6.1", "numpy>=2.0", "pyarrow>=14.0"],
)
def train_model(
    train_dataset: Input[Dataset],
    model_output:  Output[Model],
    metrics:       Output[Metrics],
    n_estimators:  int = 100,
    max_depth:     int = 4,
    random_state:  int = 42,
) -> None:
    """
    Train an XGBoost classifier and output the model artefact and metrics.

    Parameters
    ----------
    train_dataset : Input[Dataset]
        Path to the training Parquet dataset.
    model_output : Output[Model]
        Output path for the serialised model.
    metrics : Output[Metrics]
        Output path for evaluation metrics.
    """
    import joblib
    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    # Load data
    table    = pq.read_table(train_dataset.path)
    X        = table.select([c for c in table.column_names if c != "label"]).to_pandas().values
    y        = table.column("label").to_pylist()

    # Train
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)

    # Evaluate
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    metrics.log_metric("train_auc", auc)

    # Save
    joblib.dump(model, model_output.path)
```

### Container component

For steps that require a custom Docker image or non-Python logic:

```python
@dsl.container_component
def build_features(
    raw_data_path: str,
    output_dataset: Output[Dataset],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image="gcr.io/my-project/feature-builder:v1.2.0",
        command=["python", "build_features.py"],
        args=[
            "--input",  raw_data_path,
            "--output", output_dataset.path,
        ],
    )
```

---

## Pipeline Definition

Connect components by passing their outputs as inputs to downstream components:

```python
from kfp.v2 import dsl
from kfp.v2.dsl import pipeline

@pipeline(
    name="training-pipeline",
    description="End-to-end training: feature engineering → train → evaluate → register",
)
def training_pipeline(
    raw_data_path:  str,
    n_estimators:   int = 100,
    max_depth:      int = 4,
    min_auc:        float = 0.75,
) -> None:

    # Step 1 — build features
    feature_step = build_features(raw_data_path=raw_data_path)

    # Step 2 — train (receives output of step 1)
    train_step = train_model(
        train_dataset=feature_step.outputs["output_dataset"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    # Step 3 — evaluate with a quality gate
    eval_step = evaluate_model(
        model=train_step.outputs["model_output"],
        min_auc=min_auc,
    )

    # Step 4 — register only if evaluation passes
    with dsl.Condition(eval_step.outputs["passed"] == "true", name="quality-gate"):
        register_model(
            model=train_step.outputs["model_output"],
            metrics=train_step.outputs["metrics"],
        )
```

---

## Compiling and Submitting

```python
from kfp.v2 import compiler
from google.cloud import aiplatform

# Compile to a pipeline spec JSON
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path="training_pipeline.json",
)

# Submit to Vertex AI Pipelines
aiplatform.init(project="my-project", location="europe-west2")

job = aiplatform.PipelineJob(
    display_name="training-pipeline-v1",
    template_path="training_pipeline.json",
    parameter_values={
        "raw_data_path": "gs://bucket/data/features/",
        "n_estimators":  200,
        "max_depth":     5,
        "min_auc":       0.78,
    },
    enable_caching=True,   # skip steps whose inputs have not changed
)
job.submit()
```

---

## Caching

Vertex AI Pipelines caches component outputs by hashing inputs and the component spec. If a step's inputs have not changed since the last run, the cached output is reused — skipping the computation entirely.

```python
# Enable caching at job level (default: True)
job = aiplatform.PipelineJob(..., enable_caching=True)

# Disable caching for a specific component (e.g. data loading always re-runs)
feature_step = build_features(...).set_caching_options(enable_caching=False)
```

Caching is especially valuable for expensive steps (data loading, HPT) that rarely change between runs.

---

## Resource Configuration

Set compute resources per component:

```python
train_step = train_model(...).set_resources(
    cpu="4",
    memory="16G",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

# Or use a named machine type
train_step = train_model(...).set_resources(
    machine_type="n1-standard-8",
)
```

---

## Conditional Execution and Loops

```python
# Conditional — only run if a condition is met
with dsl.Condition(eval_step.outputs["passed"] == "true"):
    deploy_step = deploy_model(model=train_step.outputs["model_output"])

# ParallelFor — run a component for each item in a list
with dsl.ParallelFor(items=["exp", "stag", "prod"], name="deploy-envs") as env:
    deploy_to_env(model=train_step.outputs["model_output"], environment=env)
```

---

## Best Practices

- **One concern per component** — data loading, training, evaluation, and registration are separate components
- **Pass artefacts, not paths** — use `Input[Dataset]`, `Output[Model]` typed artefacts rather than raw string paths; Vertex AI tracks lineage automatically
- **Enable caching** — expensive steps (HPT, large data loads) should be cached; disable only for steps that must always re-run (e.g. data freshness checks)
- **Pin image versions** — use `gcr.io/my-project/image:v1.2.0`, never `latest`
- **Log metrics as artefacts** — use `Output[Metrics]` so metrics appear in the Vertex AI UI and are queryable
- **Use `dsl.Condition` for quality gates** — never deploy a model that has not passed an automated evaluation step
- **Keep pipeline parameters minimal** — only expose parameters that legitimately vary between runs; hardcode everything else in the component
