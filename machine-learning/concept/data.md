# Data Best Practices

Data quality determines the ceiling of model performance. No amount of model complexity or hyperparameter tuning recovers from leakage, a bad split, or unrepresentative training data.

---

## 1. Train / Validation / Test Splits

### The three-way split

| Split | Purpose | Touched during training? |
|-------|---------|--------------------------|
| Train | Model learns from this | Yes |
| Validation | Hyperparameter tuning, early stopping | Yes — indirectly |
| Test | Final unbiased evaluation | Never until final evaluation |

The test set is used exactly once — after all modelling decisions are finalised. Using it to guide any decision (feature selection, threshold tuning, model choice) invalidates it.

```python
from sklearn.model_selection import train_test_split
import numpy as np

rng = np.random.default_rng(42)

# Split once — stratify to preserve class ratio
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, stratify=y_train_val, random_state=42
)
```

### Temporal splits for time-series data

Never use random splits on time-series data — future data leaks into training. Always split by time: train on past, validate and test on future.

```python
# ✅ Temporal split — no future leakage
cutoff_train = "2024-09-30"
cutoff_val   = "2024-11-30"

train = data[data["date"] <= cutoff_train]
val   = data[(data["date"] > cutoff_train) & (data["date"] <= cutoff_val)]
test  = data[data["date"] > cutoff_val]
```

### Group-aware splits

When multiple rows belong to the same entity (e.g. multiple sessions per customer), a random split leaks entity-level patterns into validation. Use `GroupShuffleSplit` or `GroupKFold` to keep all rows for an entity in the same split.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=customer_ids))
```

---

## 2. Data Leakage

Leakage is when information from outside the training window contaminates the training data. It produces models that look excellent offline and fail immediately in production.

### Common leakage sources

| Source | Example | Fix |
|--------|---------|-----|
| Future data in features | Rolling average computed on the full dataset before splitting | Compute rolling features inside the training window only |
| Target encoding before split | Mean-encoding a category using the full dataset | Fit encoders on train set only, apply to val/test |
| Preprocessing before split | Scaling using the full dataset's mean/std | Fit scaler on train, transform val/test |
| Label derived from the future | Using a 30-day outcome as a feature | Ensure all features are available at prediction time |

```python
from sklearn.preprocessing import StandardScaler

# ❌ Bad — scaler fitted on full dataset, leaks val/test statistics into training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val = train_test_split(X_scaled, ...)

# ✅ Good — scaler fitted on train only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)    # transform only — no fit
X_test_scaled  = scaler.transform(X_test)
```

### The leakage test

If a feature has suspiciously high importance or the model performs far better than expected, check for leakage:
- Is this feature available at prediction time in production?
- Was it computed using data from after the prediction date?
- Does removing it cause a large drop in performance?

---

## 3. Feature Engineering

### Rules

- Compute all features as they would be available at prediction time — never use information the model would not have in production
- Validate feature distributions between train and production before deploying — a feature that looks different in production is a silent failure
- Prefer simple, interpretable features over complex derived ones — they are easier to validate and debug
- Document the business meaning of every feature — a feature without a clear interpretation is a liability

### Numerical features

```python
import numpy as np

# Clip outliers before scaling — extreme values distort the scale
def clip_and_normalise(
    values: np.ndarray,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> np.ndarray:
    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)
    clipped = np.clip(values, lower, upper)
    mean = clipped.mean()
    std  = clipped.std() + 1e-8
    return (clipped - mean) / std
```

### Categorical features

- Fit encoders on the training set only — never on the full dataset
- Handle unseen categories explicitly — a category that appears in production but not in training must not cause a crash
- High-cardinality categoricals (thousands of values) are better handled with target encoding or embeddings than one-hot encoding

```python
# Handle unseen categories safely
def safe_encode(
    values: np.ndarray,
    mapping: dict[str, int],
    unknown_value: int = -1,
) -> np.ndarray:
    return np.array([mapping.get(v, unknown_value) for v in values])
```

### Feature validation

Check that features at serving time match the training distribution:

```python
def validate_feature_range(
    values: np.ndarray,
    expected_min: float,
    expected_max: float,
    feature_name: str,
) -> None:
    if values.min() < expected_min or values.max() > expected_max:
        raise ValueError(
            f"Feature '{feature_name}' out of expected range "
            f"[{expected_min}, {expected_max}]: got [{values.min()}, {values.max()}]"
        )
```

---

## 4. Class Imbalance

Imbalanced classes (e.g. 5% positive, 95% negative) cause models to predict the majority class almost exclusively. A model that always predicts negative achieves 95% accuracy — but is useless.

### Detection

```python
unique, counts = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"Class {cls}: {cnt} ({cnt / len(y_train) * 100:.1f}%)")
```

### Strategies

| Strategy | When to use | How |
|----------|------------|-----|
| Class weights | Mild imbalance (1:5 to 1:10) | `class_weight="balanced"` in sklearn |
| Oversampling (SMOTE) | Moderate imbalance | Synthesise minority class samples |
| Undersampling | Large dataset, severe imbalance | Randomly remove majority class samples |
| Threshold tuning | Any imbalance | Adjust decision threshold on validation set |
| Use the right metric | Always | Precision/recall/F1/AUC, not accuracy |

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute balanced class weights
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# Pass to model
model = XGBClassifier(scale_pos_weight=weights[1] / weights[0])
```

### Rules

- Never evaluate an imbalanced classifier with accuracy — use precision, recall, F1, or AUC-PR
- Apply oversampling only to the training set — never to validation or test
- Threshold tuning on the validation set is often more effective than resampling

---

## 5. Data Validation

Validate data at every pipeline boundary — between ingestion and training, and between feature store and serving.

```python
def validate_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    expected_features: int,
) -> None:
    assert features.ndim == 2, "Features must be 2-D"
    assert features.shape[1] == expected_features, (
        f"Expected {expected_features} features, got {features.shape[1]}"
    )
    assert len(features) == len(labels), "Feature and label counts must match"
    assert not np.isnan(features).any(), "Features contain NaN values"
    assert not np.isinf(features).any(), "Features contain infinite values"

    unique_labels = np.unique(labels)
    assert len(unique_labels) > 1, "Labels contain only one class — check data pipeline"
```

### Rules

- Validate schema, dtypes, row counts, and null rates at every stage boundary
- Fail fast with a clear error — a silent bad dataset produces a silently bad model
- Log validation results (row count, null rate, class distribution) as pipeline metadata
