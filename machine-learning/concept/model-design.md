# Model Design Best Practices

Model design decisions — which features to include, how to constrain the model, and how complex it should be — have a larger impact on production reliability than hyperparameter tuning. A well-designed simple model outperforms a poorly designed complex one.

---

## 1. Start Simple

Always establish a simple baseline before building a complex model. Complexity must earn its place by delivering a meaningful improvement that justifies the added maintenance cost.

```
linear model → shallow tree → gradient boosting → deep learning
```

| Model | When it is enough |
|-------|-----------------|
| Logistic regression | Linearly separable features, interpretability required |
| Decision tree (depth ≤ 5) | Non-linear but interpretable, small feature set |
| Gradient boosting (XGBoost, LightGBM) | Tabular data, most production ML use cases |
| Neural network | High-dimensional unstructured data (text, images, sequences) |

A gradient boosting model on well-engineered tabular features beats a neural network in most production ML scenarios — and is far easier to debug, retrain, and explain.

---

## 2. Feature Selection

Including irrelevant or redundant features increases noise, slows training, and makes the model harder to interpret and maintain.

### Remove low-variance features

Features with near-zero variance carry no signal:

```python
import numpy as np

def remove_low_variance(
    features: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
) -> tuple[np.ndarray, list[str]]:
    variances   = features.var(axis=0)
    keep_mask   = variances > threshold
    return features[:, keep_mask], [n for n, k in zip(feature_names, keep_mask) if k]
```

### Remove highly correlated features

Correlated features add redundancy without adding signal, and can destabilise tree-based models:

```python
def remove_correlated(
    features: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95,
) -> tuple[np.ndarray, list[str]]:
    corr_matrix = np.corrcoef(features.T)
    upper       = np.triu(np.abs(corr_matrix), k=1)
    drop_mask   = (upper > threshold).any(axis=0)
    keep_mask   = ~drop_mask
    return features[:, keep_mask], [n for n, k in zip(feature_names, keep_mask) if k]
```

### Feature importance

Use permutation importance rather than impurity-based importance for tree models — impurity importance is biased towards high-cardinality features:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",
)

importance = dict(zip(feature_names, result.importances_mean))
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
```

### Rules

- Remove features with zero or near-zero variance before training
- Remove one of any pair of features with correlation > 0.95
- Validate that every feature is available at prediction time in production
- Document the business meaning of every feature — undocumented features are a maintenance risk
- Prefer fewer, well-understood features over many weakly-understood ones

---

## 3. Regularisation

Regularisation prevents overfitting by penalising model complexity. Without it, a model memorises the training data and generalises poorly.

### L1 vs L2

| Regularisation | Effect | When to use |
|---------------|--------|------------|
| L1 (Lasso) | Drives some weights to exactly zero — sparse model | Feature selection, interpretability |
| L2 (Ridge) | Shrinks all weights towards zero — no sparsity | When all features are relevant |
| ElasticNet | Combination of L1 and L2 | High-dimensional, correlated features |

```python
from sklearn.linear_model import LogisticRegression

# L2 regularisation (default) — C is inverse of regularisation strength
model_l2 = LogisticRegression(penalty="l2", C=1.0, random_state=42)

# L1 regularisation — produces sparse coefficients
model_l1 = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", random_state=42)
```

### Tree model regularisation

For gradient boosting, regularisation is controlled through depth, subsampling, and learning rate:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth=4,            # shallow trees — reduces overfitting
    learning_rate=0.05,     # small steps — requires more trees but generalises better
    subsample=0.8,          # row subsampling — reduces variance
    colsample_bytree=0.8,   # feature subsampling — reduces correlation between trees
    reg_alpha=0.1,          # L1 regularisation on leaf weights
    reg_lambda=1.0,         # L2 regularisation on leaf weights
    random_state=42,
)
```

---

## 4. Monotonic Constraints

Monotonic constraints enforce that the model's output moves in a known direction as a feature increases. They encode domain knowledge directly into the model, preventing it from learning spurious non-monotonic relationships that would not hold in production.

```python
# Example: higher discount → higher acceptance probability (monotonically increasing)
# Feature index 0 = discount: constrain to be monotonically increasing (+1)
# Feature index 1 = price:    constrain to be monotonically decreasing (-1)
# Feature index 2 = tenure:   no constraint (0)

model = XGBClassifier(
    monotone_constraints=(1, -1, 0),   # tuple of +1, -1, or 0 per feature
    random_state=42,
)
```

### When to use monotonic constraints

- When domain knowledge dictates a clear directional relationship
- When the model will be used to optimise a value (e.g. price) — unconstrained models can produce counter-intuitive optima
- When the model output must be explainable to stakeholders — constraints make behaviour predictable

### Validate constraints hold

After training, verify the constraint is respected empirically:

```python
def validate_monotonic_constraint(
    model,
    X_sample: np.ndarray,
    feature_idx: int,
    direction: int,
    n_steps: int = 20,
) -> bool:
    """
    Verify that predictions move in `direction` as feature_idx increases.

    Parameters
    ----------
    direction : int
        +1 for increasing, -1 for decreasing.
    """
    x = X_sample[0:1].copy()
    feature_range = np.linspace(
        X_sample[:, feature_idx].min(),
        X_sample[:, feature_idx].max(),
        n_steps,
    )
    preds = []
    for val in feature_range:
        x[0, feature_idx] = val
        preds.append(model.predict_proba(x)[0, 1])

    diffs = np.diff(preds)
    return bool((direction * diffs >= -1e-6).all())
```

---

## 5. Model Interpretability

Interpretability is not just a nice-to-have — it is essential for debugging, stakeholder trust, and regulatory compliance.

### SHAP values

SHAP (SHapley Additive exPlanations) provides consistent, theoretically grounded feature attributions for any model:

```python
import shap
import numpy as np

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Global feature importance — mean absolute SHAP value per feature
mean_shap = np.abs(shap_values).mean(axis=0)
importance = dict(zip(feature_names, mean_shap))
```

### Partial dependence

Partial dependence plots show the marginal effect of a single feature on the prediction, averaged over all other features:

```python
from sklearn.inspection import partial_dependence

pd_result = partial_dependence(
    model, X_val,
    features=[feature_idx],
    kind="average",
)
```

### Rules

- For tree models, always use permutation importance or SHAP — not impurity importance
- Validate that the top features make business sense — a feature that is important but inexplicable is a risk
- Document the expected direction of each important feature — use monotonic constraints to enforce it
- If a model cannot be explained to a stakeholder, it should not be deployed

---

## 6. Model Complexity vs Maintainability

| Consideration | Simple model | Complex model |
|--------------|-------------|---------------|
| Training time | Fast | Slow |
| Debugging | Easy | Hard |
| Retraining frequency | Can retrain often | Expensive to retrain |
| Explainability | High | Low |
| Performance ceiling | Lower | Higher |
| Failure mode | Predictable | Unpredictable |

A model that is 2% better in AUC but takes 10x longer to retrain, requires a specialist to debug, and fails in unpredictable ways is not a better model for production.

### Rules

- Start with the simplest model that could plausibly work
- Add complexity only when a simpler model has been proven insufficient
- Every increase in complexity must be justified by a measurable improvement on the business metric
- Prefer models that degrade gracefully — a linear model that is slightly wrong is safer than a neural network that is catastrophically wrong
