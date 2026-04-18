# Training Best Practices

A training pipeline that cannot be reproduced, validated, or debugged is a liability. Every training run should produce the same result given the same inputs, and every decision made during training should be traceable.

---

## 1. Reproducibility

Every training run must be fully reproducible from four fixed inputs: code version, data snapshot, random seeds, and hyperparameters. Without this, debugging a degraded model is guesswork.

```python
import numpy as np
import random
import os

def set_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied to numpy, Python random, and OS environment.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch — if used
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
```

### What to log per training run

Every run should record enough information to reproduce it exactly and compare it against other runs:

| Artefact | What to record |
|----------|---------------|
| Code | Git commit SHA |
| Data | Data snapshot ID, row count, date range |
| Config | Full hyperparameter dict (not just the ones that changed) |
| Environment | Python version, library versions (`pip freeze`) |
| Metrics | Train, validation, and test metrics |
| Model | Saved artefact path, checksum |

```python
import json
import hashlib

def log_run_metadata(
    config: dict,
    metrics: dict[str, float],
    data_snapshot_id: str,
    model_path: str,
) -> dict:
    return {
        "git_sha":          os.getenv("CI_COMMIT_SHA", "local"),
        "data_snapshot_id": data_snapshot_id,
        "config":           config,
        "metrics":          metrics,
        "model_path":       model_path,
        "config_hash":      hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
```

---

## 2. Cross-Validation

Cross-validation gives a more reliable estimate of generalisation performance than a single train/val split, especially on smaller datasets where a single split may be unrepresentative.

### K-Fold

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """
    Run stratified k-fold cross-validation.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible model with fit() and predict_proba().
    X : np.ndarray
        Feature matrix, shape (n, d).
    y : np.ndarray
        Binary labels, shape (n,).
    n_splits : int
        Number of folds.
    seed : int
        Random seed for fold assignment.

    Returns
    -------
    dict[str, float]
        Mean and std of AUC across folds.
    """
    from sklearn.metrics import roc_auc_score

    kf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        aucs.append(auc)

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
    }
```

### Group K-Fold — prevent entity leakage across folds

When rows share an entity (customer, session, device), use `GroupKFold` to ensure all rows for an entity stay in the same fold:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=entity_ids):
    ...
```

### Rules

- Use `StratifiedKFold` for classification — preserves class ratio in every fold
- Use `GroupKFold` when rows share an entity — prevents leakage across folds
- Use `TimeSeriesSplit` for temporal data — respects time ordering
- Never shuffle time-series data before splitting
- Report mean ± std across folds — a high std signals instability

---

## 3. Hyperparameter Tuning

### Start simple

Before tuning, establish a baseline with default hyperparameters. Tuning is only worthwhile if the baseline is already reasonable.

### Search strategies

| Strategy | When to use | Efficiency |
|----------|------------|------------|
| Grid search | Small search space, few parameters | Low — exhaustive |
| Random search | Medium space, unknown landscape | Medium — samples randomly |
| Bayesian (TPE) | Large space, expensive evaluations | High — learns from previous trials |

Bayesian optimisation (via `hyperopt` or `optuna`) is the preferred approach for ML models — it focuses evaluations on promising regions of the search space.

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import roc_auc_score
import numpy as np

def objective(params: dict) -> dict:
    model = XGBClassifier(
        max_depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        n_estimators=int(params["n_estimators"]),
        subsample=params["subsample"],
        random_state=42,
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return {"loss": -auc, "status": STATUS_OK}


search_space = {
    "max_depth":     hp.quniform("max_depth", 3, 10, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "n_estimators":  hp.quniform("n_estimators", 100, 1000, 50),
    "subsample":     hp.uniform("subsample", 0.6, 1.0),
}

trials    = Trials()
best      = fmin(objective, search_space, algo=tpe.suggest, max_evals=100, trials=trials)
```

### Rules

- Tune on the validation set — never on the test set
- Log every trial (params + metric) — not just the best result
- Use early stopping to avoid overfitting during tuning
- Fix the random seed for the tuning process itself — results should be reproducible
- Set a budget (max evaluations or wall-clock time) — unbounded tuning is a cost risk

---

## 4. Early Stopping

Early stopping halts training when validation performance stops improving, preventing overfitting and reducing training time.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=1000,      # upper bound — early stopping will stop before this
    learning_rate=0.05,
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,   # stop if no improvement for 50 rounds
    verbose=False,
)

print(f"Best iteration: {model.best_iteration}")
```

---

## 5. Training Pipeline Structure

A training pipeline should be a sequence of independently testable steps. See [package/readme.md](../package/readme.md) for the pipeline function pattern.

```
load_data → validate → engineer_features → split → train → evaluate → save
```

```python
def run_training_pipeline(config: dict) -> dict[str, float]:
    """
    Orchestrate the full training pipeline.

    Returns
    -------
    dict[str, float]
        Evaluation metrics from the held-out test set.
    """
    set_seeds(config["seed"])

    features, labels = load_data(config["data_path"])
    validate_training_data(features, labels, config["n_features"])

    features = engineer_features(features, config)
    X_train, X_val, X_test, y_train, y_val, y_test = split(
        features, labels, config
    )

    model   = train(X_train, y_train, X_val, y_val, config)
    metrics = evaluate(model, X_test, y_test)

    save_model(model, config["output_path"])
    log_run_metadata(config, metrics, config["data_snapshot_id"], config["output_path"])

    return metrics
```

### Rules

- Set seeds at the very start of the pipeline — before any data loading or splitting
- Validate data before training — fail fast on bad input
- Save the full config alongside the model artefact — not just the best hyperparameters
- Log metrics on train, validation, and test sets — train-only metrics hide overfitting
