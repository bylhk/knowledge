# Model Registry & Lineage

A model registry is the single source of truth for trained model artefacts. It stores not just the model file but the full provenance — the code, data, config, and metrics that produced it — so any deployed model can be traced back to its exact origin and reproduced.

---

## Why a Model Registry

Without a registry, model management degrades into:
- Model files named `model_final_v2_REAL.pkl` in S3 with no metadata
- No record of which data or config produced a given model
- No way to roll back to a previous version when a deployment degrades
- Training and serving using different preprocessing logic

A registry enforces discipline: every model is versioned, every version has lineage, and promotion between environments is explicit and auditable.

---

## What to Store Per Model Version

| Artefact | Why |
|----------|-----|
| Model file (`.pkl`, `.onnx`, `.pt`) | The serialised model weights and structure |
| Git commit SHA | Exact code that produced the model |
| Data snapshot ID | Exact dataset used for training |
| Training config | Full hyperparameter dict — not just the ones that changed |
| Config hash | Detect if config changed between runs |
| Evaluation metrics | Train, validation, and test metrics |
| Feature schema | Feature names, dtypes, expected ranges |
| Training date | When the model was trained |
| Parent model version | If fine-tuned from a previous version |

---

## MLflow

MLflow is the most widely used open-source experiment tracking and model registry. It runs locally, self-hosted, or as a managed service (Databricks, Azure ML).

### Tracking experiments

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn-model-training")

with mlflow.start_run(run_name="gbm-v12") as run:
    # Log hyperparameters
    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}
    mlflow.log_params(params)

    # Log environment metadata
    mlflow.set_tags({
        "git_sha":          "abc123f",
        "data_snapshot_id": "snapshot-2025-01-15",
        "environment":      "exp",
    })

    # Train
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Log metrics
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc   = roc_auc_score(y_val,   model.predict_proba(X_val)[:, 1])
    mlflow.log_metrics({"train_auc": train_auc, "val_auc": val_auc})

    # Log the model with input schema
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict_proba(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name="churn-model",
    )

    print(f"Run ID: {run.info.run_id}")
```

### Registering and promoting models

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a run's model in the registry
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model",
)

# Add a description
client.update_model_version(
    name="churn-model",
    version=model_version.version,
    description="GBM trained on 2025-01-15 snapshot. Val AUC: 0.823",
)

# Transition through stages: None → Staging → Production
client.transition_model_version_stage(
    name="churn-model",
    version=model_version.version,
    stage="Staging",
    archive_existing_versions=False,
)

# After validation, promote to Production
client.transition_model_version_stage(
    name="churn-model",
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True,   # archive the previous Production version
)
```

### Loading a model from the registry

```python
# Load the current Production model
model = mlflow.sklearn.load_model("models:/churn-model/Production")

# Load a specific version
model = mlflow.sklearn.load_model("models:/churn-model/12")

# Load from a run
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

### Querying the registry

```python
# List all versions of a model
versions = client.search_model_versions("name='churn-model'")
for v in versions:
    print(f"v{v.version} | {v.current_stage} | {v.run_id}")

# Get the latest Production version
prod_versions = client.get_latest_versions("churn-model", stages=["Production"])
latest_prod   = prod_versions[0]
```

---

## Vertex AI Model Registry (GCP)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="europe-west2")

# Upload a model to the registry
model = aiplatform.Model.upload(
    display_name="churn-model-v12",
    artifact_uri="gs://bucket/models/churn/v12/",
    serving_container_image_uri="europe-docker.pkg.dev/my-project/serving:v1.2.0",
    labels={
        "environment": "exp",
        "git_sha":     "abc123f",
        "team":        "ml-pricing",
    },
)

# List models
models = aiplatform.Model.list(filter="display_name=churn-model*")

# Deploy to an endpoint
endpoint = aiplatform.Endpoint.create(display_name="churn-endpoint")
endpoint.deploy(
    model=model,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
)
```

---

## SageMaker Model Registry (AWS)

```python
import boto3

sm = boto3.client("sagemaker", region_name="eu-west-1")

# Create a model package group (equivalent to a named model in MLflow)
sm.create_model_package_group(
    ModelPackageGroupName="churn-model",
    ModelPackageGroupDescription="Churn prediction model",
)

# Register a model version
response = sm.create_model_package(
    ModelPackageGroupName="churn-model",
    ModelPackageDescription="GBM v12 — val AUC 0.823",
    InferenceSpecification={
        "Containers": [{
            "Image":        "123456789.dkr.ecr.eu-west-1.amazonaws.com/serving:v1.2.0",
            "ModelDataUrl": "s3://bucket/models/churn/v12/model.tar.gz",
        }],
        "SupportedContentTypes":    ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
)

# Approve for deployment
sm.update_model_package(
    ModelPackageName=response["ModelPackageArn"],
    ModelApprovalStatus="Approved",
)
```

---

## Lineage Tracking

Lineage answers: given a model in production, what data, code, and config produced it?

### Minimum lineage record

```python
from dataclasses import dataclass, asdict
import json
import hashlib

@dataclass
class ModelLineage:
    model_name:       str
    model_version:    int
    git_sha:          str
    data_snapshot_id: str
    training_date:    str
    feature_names:    list[str]
    hyperparameters:  dict
    metrics:          dict[str, float]
    config_hash:      str

    @classmethod
    def create(
        cls,
        model_name: str,
        version: int,
        git_sha: str,
        data_snapshot_id: str,
        feature_names: list[str],
        hyperparameters: dict,
        metrics: dict[str, float],
    ) -> "ModelLineage":
        from datetime import datetime, timezone
        config_hash = hashlib.md5(
            json.dumps(hyperparameters, sort_keys=True).encode()
        ).hexdigest()
        return cls(
            model_name=model_name,
            model_version=version,
            git_sha=git_sha,
            data_snapshot_id=data_snapshot_id,
            training_date=datetime.now(timezone.utc).isoformat(),
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            metrics=metrics,
            config_hash=config_hash,
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
```

Store `lineage.json` alongside every model artefact:

```
s3://bucket/models/churn/v12/
├── model.pkl
├── lineage.json      ← full provenance
└── feature_schema.json
```

---

## Model Promotion Workflow

```
train (exp) → register → evaluate → approve → deploy (stag) → validate → promote (prod)
```

Never deploy a model that has not passed an automated evaluation gate. Define promotion criteria before training — not after seeing the results.

```python
PROMOTION_CRITERIA = {
    "min_val_auc":            0.78,
    "max_train_val_auc_gap":  0.05,   # gap > 0.05 suggests overfitting
    "min_val_samples":        5_000,
}

def can_promote(metrics: dict[str, float]) -> tuple[bool, str]:
    if metrics["val_auc"] < PROMOTION_CRITERIA["min_val_auc"]:
        return False, f"val_auc {metrics['val_auc']:.4f} below threshold"
    gap = metrics["train_auc"] - metrics["val_auc"]
    if gap > PROMOTION_CRITERIA["max_train_val_auc_gap"]:
        return False, f"train/val gap {gap:.4f} suggests overfitting"
    return True, "passed"
```

---

## Rules

- Every model artefact must have a `lineage.json` stored alongside it — no anonymous model files
- Register models in the registry before deploying — never deploy directly from a training run output
- Use semantic versioning for model versions — increment on every registered model, not just deployments
- Archive the previous Production version when promoting a new one — keep it available for rollback
- Never overwrite a registered model version — versions are immutable once registered
- Store the feature schema with the model — serving code loads feature names from the registry, never hardcodes them
