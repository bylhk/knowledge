# Package & Code Structure

Good package structure makes code easier to test, maintain, and extend. The core question at every level is the same: does this piece of logic have a single, clear responsibility?

---

## When to Use a Function vs a Class

### Use a function when

- The logic is stateless — it takes inputs and returns outputs with no side effects
- It represents a single, well-defined transformation or calculation
- It does not need to share state across multiple calls

```python
# ✅ Function — stateless transformation, clear input/output
def normalise(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Standardise an array to zero mean and unit variance."""
    return (values - mean) / std


def compute_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean, normalising weights internally."""
    return float(np.dot(values, weights) / weights.sum())
```

### Use a class when

- The logic requires shared state across multiple method calls
- Multiple operations act on the same underlying data or resources
- Initialisation is expensive (loading a model, opening a DB connection) and should happen once

```python
# ✅ Class — shared state: model and DB connection loaded once, reused across calls
class BatchScorer:
    def __init__(self, model_path: str, db_client: DatabaseClient) -> None:
        self.db_client = db_client
        self._model    = self._load_model(model_path)   # expensive — done once at init

    def score(self, entity_ids: list[str]) -> np.ndarray:
        features = self.db_client.fetch(entity_ids)
        return self._model.predict(features)

    def _load_model(self, path: str):
        ...
```

### Decision guide

| Scenario | Use |
|----------|-----|
| Pure calculation, no state | Function |
| Single transformation step | Function |
| Shared resource (model, DB connection) | Class |
| Multiple operations on the same data | Class |
| Configuration-driven behaviour | Class |
| Single-use utility | Function |

---

## When to Use Methods vs Sub-functions Inside a Class

### Public methods — the external interface

Public methods are the contract between the class and its callers. Keep them high-level and focused on what the class does, not how.

```python
class ReportBuilder:
    def build(self, raw_data: pd.DataFrame) -> dict[str, float]:
        """Public interface — callers only need to know about this."""
        cleaned = self._clean(raw_data)
        return self._aggregate(cleaned)
```

### Private methods — internal implementation

Private methods (prefixed `_`) break complex logic into named, testable steps. They are implementation details — callers should never call them directly.

```python
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop nulls, enforce dtypes — hidden from callers
        return df.dropna().astype({"value": "float32"})

    def _aggregate(self, df: pd.DataFrame) -> dict[str, float]:
        return {
            "mean":   float(df["value"].mean()),
            "median": float(df["value"].median()),
            "std":    float(df["value"].std()),
        }
```

### Rule of thumb

If a method is longer than ~20 lines, it is doing too much — extract the logical sub-steps into private methods with descriptive names. The public method then reads like a summary of the algorithm.

```python
# ✅ Public method reads as a clear sequence of named steps
def process(self, request: dict) -> dict:
    validated = self._validate(request)
    features  = self._fetch_features(validated)
    scores    = self._score(features)
    return self._format_response(scores)
```

---

## When to Use a Base Class

### Use a base class (abstract class) when

- Multiple concrete classes share the same interface but have different implementations
- You want to enforce that all subclasses implement specific methods
- You are building a strategy pattern — swappable implementations selected at runtime via config

```python
from abc import ABC, abstractmethod

# ✅ Base class — defines the interface, enforces the contract
class SamplingStrategy(ABC):
    @abstractmethod
    def sample(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a sample of n rows from features and labels."""
        ...


# Concrete implementations — all share the same signature
class RandomSampling(SamplingStrategy):
    def sample(
        self, features: np.ndarray, labels: np.ndarray, n: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng     = np.random.default_rng(seed)
        indices = rng.choice(len(features), size=n, replace=False)
        return features[indices], labels[indices]


class StratifiedSampling(SamplingStrategy):
    def sample(
        self, features: np.ndarray, labels: np.ndarray, n: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # preserve class distribution across unique label values
        rng     = np.random.default_rng(seed)
        classes = np.unique(labels)
        indices = np.concatenate([
            rng.choice(np.where(labels == c)[0], size=n // len(classes), replace=False)
            for c in classes
        ])
        return features[indices], labels[indices]


class FingerprintSampling(SamplingStrategy):
    def sample(
        self, features: np.ndarray, labels: np.ndarray, n: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # deterministic hash-based sample — same rows every run
        ...
```

The strategy is then selected at runtime from config — no `if/elif` chains in the calling code:

```python
SAMPLING_STRATEGIES: dict[str, type[SamplingStrategy]] = {
    "random":      RandomSampling,
    "stratified":  StratifiedSampling,
    "fingerprint": FingerprintSampling,
}

strategy           = SAMPLING_STRATEGIES[config["sampling_strategy"]]()
sampled_features, sampled_labels = strategy.sample(features, labels, n=10_000, seed=42)
```

### Don't use a base class when

- There is only one implementation — a base class with one subclass adds indirection with no benefit
- The shared logic is a utility function — extract it as a standalone function instead
- Subclasses would override every method — the base class provides no shared contract

---

## Pipeline Functions — Use to Prevent Oversized Scripts

Pipeline functions are not the preferred pattern — business logic should live in small, independently testable functions and classes. However, they are necessary in practice to prevent a single script from growing into an unmanageable monolith.

The rule is: a pipeline function contains **only orchestration** — it calls other functions in sequence and passes outputs forward. No business logic, no data transformation, no validation belongs inside it.

```python
# ❌ Bad — business logic mixed into the pipeline script
def run():
    df = pd.read_parquet("data/features.parquet")
    df = df.dropna(subset=["label"])                    # validation logic here
    df["score"] = df["value"] / df["value"].max()       # transformation logic here
    model.fit(df.drop("label", axis=1), df["label"])
    metrics = {"auc": roc_auc_score(...)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


# ✅ Good — pipeline function is pure orchestration
def run():
    df      = load_features("data/features.parquet")    # each step is its own function
    df      = validate(df)
    df      = engineer_features(df)
    model   = train(df, config)
    metrics = evaluate(model, df)
    save_metrics(metrics, "metrics.json")
```

### When a pipeline function is justified

- The script would otherwise exceed ~100 lines of sequential logic
- You need a single entry point callable from a CI job, a CLI, or a cloud workflow
- You want to make the high-level flow readable without reading every implementation detail

### Rules

- Pipeline functions contain no business logic — only calls to named functions
- Each step function must be independently importable and testable in isolation
- Keep the pipeline function short enough to read in one screen — if it grows, the steps are too coarse
- Name steps clearly — `validate`, `engineer_features`, `train`, `evaluate` — so the pipeline reads as documentation

---

## Modularisation — Shared Dependencies Across Similar Measurements

Modularisation is most valuable when multiple modules share similar measurements or calculations. Extracting the shared element into one place means a change to the calculation propagates everywhere automatically — nothing is missed.

### The shared dependency pattern

When two or more metrics share a common intermediate value, compute it once and pass it as a parameter. This is especially important when the shared element is expensive to compute or must be consistent across metrics.

```python
# ❌ Bad — preprocess_features() called independently inside each postprocess function.
# If preprocessing changes, every function must be updated and re-tested separately.
def postprocess_churn(
    raw_features: np.ndarray,
    actuals: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    features      = preprocess_features(raw_features)   # duplicated
    baseline_rate = actuals.mean()
    top_mask      = scores >= np.percentile(scores, 90)
    return {
        "churn_lift":    actuals[top_mask].mean() / baseline_rate,
        "feature_mean":  features.mean(),
    }


def postprocess_retention(
    raw_features: np.ndarray,
    actuals: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    features      = preprocess_features(raw_features)   # duplicated
    baseline_rate = actuals.mean()
    top_mask      = scores >= np.percentile(scores, 90)
    return {
        "retention_lift": (1 - actuals[top_mask].mean()) / (1 - baseline_rate),
        "feature_std":    features.std(),
    }
```

```python
# ✅ Good — preprocess_features() called once by the caller, injected as a parameter.
# Both postprocess functions receive the same preprocessed features — consistent and efficient.
def preprocess_features(raw_features: np.ndarray) -> np.ndarray:
    """
    Normalise and clip raw features.

    Parameters
    ----------
    raw_features : np.ndarray
        Raw input features, shape (n, d).

    Returns
    -------
    np.ndarray
        Normalised features clipped to [-3, 3], shape (n, d).
    """
    mean       = raw_features.mean(axis=0)
    std        = raw_features.std(axis=0) + 1e-8   # avoid division by zero
    normalised = (raw_features - mean) / std
    return np.clip(normalised, -3.0, 3.0)


def postprocess_churn(
    features: np.ndarray,       # preprocessed — injected by caller
    actuals: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    baseline_rate = actuals.mean()
    top_mask      = scores >= np.percentile(scores, 90)
    return {
        "churn_lift":   actuals[top_mask].mean() / baseline_rate,
        "feature_mean": features.mean(),
    }


def postprocess_retention(
    features: np.ndarray,       # preprocessed — injected by caller
    actuals: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    baseline_rate = actuals.mean()
    top_mask      = scores >= np.percentile(scores, 90)
    return {
        "retention_lift": (1 - actuals[top_mask].mean()) / (1 - baseline_rate),
        "feature_std":    features.std(),
    }


# Caller preprocesses once — both postprocess functions receive the same array
features  = preprocess_features(raw_features)
churn     = postprocess_churn(features, actuals, scores)
retention = postprocess_retention(features, actuals, scores)
```

This pattern also makes each function independently testable — `postprocess_churn` and `postprocess_retention` can be tested with any pre-built `np.ndarray`, without needing to replicate the preprocessing logic in every test.

### When to modularise

- The same logic appears in more than one place — extract it
- A function is doing more than one thing (validate + transform + log = three functions)
- A module is growing beyond ~300 lines — split by concern
- A calculation must be consistent across multiple metrics — make it a shared dependency

### Why modularisation makes unit testing easier

A function that does one thing with explicit inputs and outputs can be tested in complete isolation — no database, no model, no filesystem required.

```python
# ✅ Pure function — trivial to unit test
def compute_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / weights.sum())


def test_compute_weighted_mean_uniform():
    values  = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert compute_weighted_mean(values, weights) == pytest.approx(2.0)

def test_compute_weighted_mean_skewed():
    values  = np.array([1.0, 3.0])
    weights = np.array([1.0, 3.0])
    assert compute_weighted_mean(values, weights) == pytest.approx(2.5)
```

Because `preprocess_features` is a separate function, it can be tested independently — and the postprocess functions can be tested by passing a pre-built array directly, without needing to replicate preprocessing in every test:

```python
def test_preprocess_clips_outliers():
    raw = np.array([[0.0, 100.0], [0.0, -100.0]])
    out = preprocess_features(raw)
    assert out.max() <= 3.0
    assert out.min() >= -3.0


def test_postprocess_churn_lift_above_one_when_model_is_good():
    # inject pre-built features — no preprocessing needed in this test
    features = np.zeros((100, 5))
    actuals  = np.zeros(100)
    actuals[:10] = 1                              # top 10% are churners
    scores   = np.linspace(0, 1, 100)

    result = postprocess_churn(features, actuals, scores)
    assert result["churn_lift"] > 1.0
```

### Testing a class with dependency injection

Pass dependencies (model, DB client, logger) into the constructor so tests can inject fakes:

```python
class BatchScorer:
    def __init__(self, model, db_client) -> None:
        self.model     = model
        self.db_client = db_client

    def score(self, entity_ids: list[str]) -> np.ndarray:
        features = self.db_client.fetch(entity_ids)
        return self.model.predict(features)


# Test — inject fakes, no real model or DB needed
def test_batch_scorer_output_shape():
    fake_model     = FakeModel(output=np.array([0.9, 0.4, 0.7]))
    fake_db_client = FakeDBClient(features=np.zeros((3, 10)))

    scorer = BatchScorer(model=fake_model, db_client=fake_db_client)
    scores = scorer.score(["id_1", "id_2", "id_3"])

    assert scores.shape == (3,)
```

### Rules

- Every public function and method should have at least one unit test
- Tests should not require a database, filesystem, or network — inject fakes for all external dependencies
- One test per behaviour — not one test per function
- Test edge cases explicitly: empty input, null values, boundary conditions
- Keep test files mirroring the source structure: `tests/ml/test_metrics.py` tests `src/ml/metrics.py`

---

## Pydantic — Structured Data with Validation

Pydantic v2 provides runtime type validation, serialisation, and schema generation for Python dataclasses. Use it wherever data crosses a boundary — API requests, config loading, pipeline outputs — to catch bad data immediately rather than silently propagating it.

### BaseModel vs dataclass vs dict

| | `dict` | `dataclass` | `pydantic.BaseModel` |
|---|--------|-------------|---------------------|
| Type validation | ❌ | ❌ (hints only) | ✅ Runtime enforced |
| Default values | Manual | ✅ | ✅ |
| Nested validation | ❌ | ❌ | ✅ |
| JSON serialisation | Manual | Manual | ✅ `.model_dump()` |
| Schema generation | ❌ | ❌ | ✅ `.model_json_schema()` |
| Immutability | ❌ | Optional | Optional (`frozen=True`) |

Use `BaseModel` when data comes from outside the process (API, config file, message queue). Use a plain `dataclass` for internal data structures that never cross a boundary.

### Defining models

```python
from pydantic import BaseModel, Field, field_validator
import numpy as np

class PredictRequest(BaseModel):
    entity_id:   str
    features:    list[float]
    model_version: str | None = None

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: list[float]) -> list[float]:
        if any(not np.isfinite(x) for x in v):
            raise ValueError("features must all be finite — no NaN or Inf")
        return v


class PredictResponse(BaseModel):
    entity_id: str
    score:     float = Field(ge=0.0, le=1.0)   # ge = greater-or-equal, le = less-or-equal
    model_version: str
```

### Nested models

```python
class FeatureConfig(BaseModel):
    feature_names: list[str]
    n_features:    int
    dtype:         str = "float32"

class TrainingConfig(BaseModel):
    features:      FeatureConfig
    n_estimators:  int   = Field(default=100, gt=0)
    max_depth:     int   = Field(default=4,   gt=0, le=20)
    learning_rate: float = Field(default=0.05, gt=0.0, lt=1.0)
    random_state:  int   = 42

# Pydantic validates all nested fields on construction
config = TrainingConfig(
    features=FeatureConfig(feature_names=["age", "spend"], n_features=2),
    n_estimators=200,
)
```

### Serialisation and deserialisation

```python
# dict → model (validates on construction)
request = PredictRequest.model_validate({"entity_id": "e1", "features": [0.5, 0.3]})

# JSON string → model
request = PredictRequest.model_validate_json('{"entity_id": "e1", "features": [0.5, 0.3]}')

# model → dict
request.model_dump()                        # {"entity_id": "e1", "features": [0.5, 0.3], ...}

# model → JSON string
request.model_dump_json()

# Generate JSON schema (used by FastAPI for OpenAPI docs automatically)
PredictRequest.model_json_schema()
```

### Rules

- Use `BaseModel` at every external boundary — API, config, message queue payload
- Use `Field(gt=0, le=1.0)` constraints to encode valid ranges — fail at construction, not deep in business logic
- Use `@field_validator` for cross-field or domain-specific validation
- Prefer `model_validate` over constructing directly from `**kwargs` — it runs all validators
- Use `frozen=True` for config models that must not be mutated after loading

---

## Python Package Best Practices

### Package layout

```
my-package/
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── model.py        # model loading and inference
│       │   ├── features.py     # feature engineering
│       │   └── metrics.py      # evaluation metrics
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── training.py     # training pipeline orchestration
│       │   └── evaluation.py   # evaluation pipeline orchestration
│       └── utils/
│           ├── __init__.py
│           ├── logger.py       # shared logging setup
│           └── validation.py   # shared input validation
├── tests/
│   ├── ml/
│   │   ├── test_model.py
│   │   ├── test_features.py
│   │   └── test_metrics.py
│   ├── pipeline/
│   │   └── test_training.py
│   ├── utils/
│   │   ├── test_logger.py
│   │   └── test_validation.py
│   └── conftest.py             # shared fixtures (fake model, fake DB client, etc.)
├── pyproject.toml
└── README.md
```

Each module has one concern. `pipeline/` modules only import from `ml/` and `utils/` — never the reverse. Circular imports are a sign that concerns are not properly separated.

### pyproject.toml — single source of truth

```toml
[build-system]
requires      = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name            = "my-package"
version         = "1.0.0"
description     = "Short description of the package"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scikit-learn>=1.6",
    "fastapi>=0.115",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "black>=24.0",
    "isort>=5.13",
    "ruff>=0.4",
]

[tool.black]
line-length = 120

[tool.isort]
profile     = "black"
line_length = 120

[tool.ruff]
line-length = 120
select      = ["E", "F", "I"]
```

### `__init__.py` — control the public API

Use `__init__.py` to define what is importable from the package. Anything not listed here is an internal implementation detail.

```python
# src/my_package/__init__.py
from my_package.ml.model import BatchScorer
from my_package.ml.metrics import compute_weighted_mean, compute_lift_metrics

__all__ = ["BatchScorer", "compute_weighted_mean", "compute_lift_metrics"]
```

### Editable install for development

```bash
pip install -e ".[dev]"   # installs package + dev dependencies in editable mode
```

### Rules

- Use `src/` layout — prevents accidental imports of the local directory instead of the installed package
- Define `__all__` in `__init__.py` — makes the public API explicit and prevents internal modules from leaking
- Use absolute imports only — `from my_package.ml.model import BatchScorer`, never relative `from ..model import`
- Version the package in `pyproject.toml` only — never duplicate the version string in `__init__.py` or `setup.cfg`
- Separate `dependencies` (runtime) from `optional-dependencies.dev` (development tools) — production installs should not include pytest or black
- `pipeline/` modules orchestrate only — they import from `ml/` and `utils/`, never the reverse
