# Experiment Tracking

Experiment tracking records every training run — its parameters, metrics, artefacts, and environment — so runs can be compared, reproduced, and promoted. Without it, the question "which config produced our best model?" has no reliable answer.

---

## What to Track Per Run

| Category | What to log |
|----------|------------|
| Parameters | All hyperparameters — not just the ones that changed |
| Metrics | Train, validation, and test metrics at each epoch or fold |
| Artefacts | Model file, feature schema, evaluation plots, confusion matrix |
| Environment | Git SHA, Python version, library versions (`pip freeze`) |
| Data | Data snapshot ID, row count, date range, class distribution |
| Tags | Environment (`exp`/`stag`), team, experiment name, run notes |

Log everything at the start of the run — not just the final result. A run that fails halfway through should still have its parameters logged so you know what was tried.

---

## MLflow Tracking

MLflow is the standard open-source experiment tracker. It stores runs locally or on a remote server and provides a UI for comparing runs.

### Setup

```bash
pip install mlflow

# Start the tracking server locally
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### Logging a training run

```python
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("churn-model-v2")

def run_experiment(config: dict) -> None:
    with mlflow.start_run(run_name=f"gbm-depth{config['max_depth']}") as run:

        # 1. Log all parameters upfront
        mlflow.log_params(config)
        mlflow.set_tags({
            "git_sha":          get_git_sha(),
            "data_snapshot_id": config["data_snapshot_id"],
            "environment":      "exp",
        })

        # 2. Train
        model = train(config)

        # 3. Log metrics
        for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
            probs = model.predict_proba(X)[:, 1]
            mlflow.log_metrics({
                f"{split}_auc":    roc_auc_score(y, probs),
                f"{split}_auc_pr": average_precision_score(y, probs),
            })

        # 4. Log artefacts
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log a feature importance plot
        fig, ax = plt.subplots()
        ax.barh(feature_names, model.feature_importances_)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)

        # Log the config as a JSON artefact
        mlflow.log_dict(config, "config.json")

        print(f"Run: {run.info.run_id}")
```

### Comparing runs in the UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
# Select multiple runs → Compare → view metric charts side by side
```

### Querying runs programmatically

```python
from mlflow.tracking import MlflowClient
import pandas as pd

client = MlflowClient()

# Search runs in an experiment, sorted by val_auc descending
runs = mlflow.search_runs(
    experiment_names=["churn-model-v2"],
    filter_string="metrics.val_auc > 0.75",
    order_by=["metrics.val_auc DESC"],
    max_results=10,
)

# runs is a DataFrame — each row is a run
print(runs[["run_id", "params.max_depth", "metrics.val_auc", "metrics.train_auc"]])
```

### Autologging — log sklearn/XGBoost automatically

```python
# Log all params, metrics, and the model with one line
mlflow.sklearn.autolog(log_models=True, log_input_examples=True)

# Now just train — MLflow captures everything
model = GradientBoostingClassifier(n_estimators=200, max_depth=4)
model.fit(X_train, y_train)
```

---

## Weights & Biases (W&B)

W&B is a managed experiment tracking service with richer visualisation than MLflow — particularly useful for deep learning with per-step metric logging.

```python
import wandb
import numpy as np

wandb.init(
    project="churn-model",
    name="gbm-v12",
    config={
        "n_estimators":  200,
        "max_depth":     4,
        "learning_rate": 0.05,
        "data_snapshot": "2025-01-15",
    },
    tags=["exp", "gbm"],
)

# Log metrics per epoch (deep learning)
for epoch in range(n_epochs):
    train_loss, val_loss = run_epoch(model, epoch)
    wandb.log({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
    })

# Log final metrics
wandb.summary["val_auc"]    = val_auc
wandb.summary["train_auc"]  = train_auc

# Log artefacts
artifact = wandb.Artifact("churn-model", type="model")
artifact.add_file("model.pkl")
wandb.log_artifact(artifact)

wandb.finish()
```

---

## Hyperparameter Search with Experiment Tracking

Track every trial of a hyperparameter search as a separate run — not just the best result.

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import numpy as np

def objective(params: dict) -> dict:
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)

        model = train_model(params)
        val_auc = evaluate(model, X_val, y_val)

        mlflow.log_metric("val_auc", val_auc)
        return {"loss": -val_auc, "status": STATUS_OK}


search_space = {
    "max_depth":     hp.quniform("max_depth", 3, 8, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "n_estimators":  hp.quniform("n_estimators", 100, 500, 50),
}

with mlflow.start_run(run_name="hpt-search"):
    mlflow.log_params({"search_space": str(search_space), "max_evals": 50})
    trials = Trials()
    best   = fmin(objective, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    mlflow.log_params({f"best_{k}": v for k, v in best.items()})
```

---

## Notebook Best Practices

Notebooks are the primary experimentation environment. They accumulate state silently — a variable set three cells ago affects the current cell in ways that are invisible to a reader.

### Rules for reproducible notebooks

```python
# Cell 1 — always set seeds at the top
import numpy as np
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
```

```python
# Cell 2 — log the environment
import subprocess
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
print(f"Git SHA: {git_sha}")

import sys
print(f"Python: {sys.version}")

import importlib.metadata
for pkg in ["numpy", "scikit-learn", "xgboost"]:
    print(f"{pkg}: {importlib.metadata.version(pkg)}")
```

- Clear all outputs before committing — `Cell → All Output → Clear` — outputs contain data that should not be in version control
- Restart and run all before sharing — confirms the notebook runs top-to-bottom without hidden state
- Keep notebooks thin — extract reusable functions into `src/` and import them; notebooks should orchestrate, not implement

### Transition from notebook to production

```
Notebook (explore)
    ↓ extract pure functions
src/ml/features.py, src/ml/metrics.py
    ↓ add tests
tests/unit/test_features.py
    ↓ add docstrings
    ↓ wire into pipeline
src/pipeline/training.py
    ↓ register in CI
.gitlab-ci.yml
```

The signal to extract from a notebook: a function is called more than once, or it is needed in a different notebook or script.

---

## Experiment Comparison Checklist

Before promoting a challenger model, compare it against the champion on:

| Dimension | What to check |
|-----------|--------------|
| Offline metrics | Val AUC, AUC-PR, calibration — is the improvement statistically significant? |
| Train/val gap | Is the challenger overfitting more than the champion? |
| Feature importance | Did the top features change? Are new top features explainable? |
| Score distribution | Did the prediction distribution shift significantly? |
| Latency | Is inference time within the serving budget? |
| Data used | Was the challenger trained on the same data snapshot? |

---

## Rules

- Log all parameters at the start of every run — not just the ones you changed
- Log metrics on both train and validation sets — train-only metrics hide overfitting
- Never delete experiment runs — a failed run is still useful information about what was tried
- Use nested runs for HPT — one parent run per search, one child run per trial
- Clear notebook outputs before committing — outputs contain data that should not be in version control
- Restart and run all notebooks before sharing — confirms reproducibility
- Extract functions from notebooks into `src/` as soon as they are used more than once
- Link every registered model version to its MLflow run ID — the run is the full provenance record
