# Machine Learning Best Practices

A reliable ML system is not just a well-trained model — it is a disciplined pipeline from raw data to production monitoring. Each stage has its own failure modes, and a mistake at any stage compounds downstream.

```
data → training → evaluation → model design → production → monitoring
```

---

## Overview

| Area | Key concern | File |
|------|------------|------|
| Data | Leakage, splits, imbalance, feature engineering | [data.md](data.md) |
| Training | Reproducibility, cross-validation, hyperparameter tuning | [training.md](training.md) |
| Evaluation | Metric selection, offline vs online, champion/challenger | [evaluation.md](evaluation.md) |
| Model Design | Feature selection, regularisation, constraints | [model-design.md](model-design.md) |
| Production | Versioning, serving, input validation, fallback logic | [production.md](production.md) |
| Monitoring | Drift detection, retraining triggers, KPI tracking | [monitoring.md](monitoring.md) |
| Memory & I/O | Batch loading, file formats, Parquet vs Feather | [memory-and-io.md](memory-and-io.md) |
| Testing | pytest, mocking, ML-specific tests, endpoint tests | [testing.md](testing.md) |
| Model Registry | Lineage, MLflow, Vertex AI, SageMaker registries | [model-registry.md](model-registry.md) |
| Experiment Tracking | MLflow, W&B, HPT tracking, notebook best practices | [experiment-tracking.md](experiment-tracking.md) |

---

## Core Principles

### Fail on data, not on code

Most ML failures are data failures — leakage, distribution shift, silent schema changes, or incorrect labels. Validate data at every stage boundary. A model trained on leaked labels will look excellent offline and fail immediately in production.

### Reproducibility is not optional

Every training run must be reproducible from a fixed set of inputs: code version, data snapshot, random seeds, and hyperparameters. Without reproducibility, debugging a degraded model is guesswork.

### Offline metrics are not a proxy for business value

A model with higher AUC does not always produce better business outcomes. Always define the business metric first, then choose the ML metric that best correlates with it. Validate the correlation with an online experiment before declaring a model better.

### Simple models first

Start with the simplest model that could plausibly work — logistic regression, a shallow tree, a linear baseline. A complex model that is only marginally better than a simple one is harder to debug, slower to retrain, and more likely to degrade unexpectedly. Complexity must earn its place.

### Production is a different distribution

The data a model sees in production is never identical to training data. Features go missing, upstream schemas change, user behaviour shifts. Design every component to degrade gracefully rather than fail silently.

---

## Anti-Patterns

| Anti-pattern | Consequence |
|-------------|-------------|
| Training on the full dataset before splitting | Data leakage — inflated offline metrics, poor production performance |
| Tuning hyperparameters on the test set | Overfitting to the test set — metrics do not generalise |
| No random seed | Non-reproducible results — impossible to debug regressions |
| Deploying without input validation | Silent failures when upstream data changes schema |
| No fallback when the model fails | Hard failure in production instead of a safe default |
| Monitoring only infrastructure, not model behaviour | Model degrades silently while the service stays healthy |
| Skipping a simple baseline | No reference point — impossible to know if the complex model is actually better |
