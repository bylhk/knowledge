# Testing

Tests are the primary mechanism for catching regressions, validating business logic, and enabling safe refactoring. In ML projects, tests cover three distinct layers: unit tests on pure functions, integration tests on pipeline steps, and ML-specific tests on data and model behaviour.

---

## Test Structure

```
tests/
├── conftest.py              # shared fixtures available to all tests
├── unit/
│   ├── test_features.py     # pure function tests — no I/O
│   ├── test_metrics.py
│   └── test_validation.py
├── integration/
│   ├── test_pipeline.py     # end-to-end pipeline step tests
│   └── test_serving.py      # API endpoint tests
└── ml/
    ├── test_data_quality.py # data schema, distribution, leakage checks
    └── test_model_output.py # model output shape, range, monotonicity
```

Mirror the source structure — `tests/unit/test_features.py` tests `src/ml/features.py`.

---

## pytest Fundamentals

### Basic test

```python
import numpy as np
import pytest

def compute_discount(price: float, avg_price: float) -> float:
    if avg_price == 0:
        return 0.0
    return (avg_price - price) / avg_price


def test_discount_positive_when_below_avg():
    assert compute_discount(8.0, 10.0) == pytest.approx(0.2)

def test_discount_zero_when_at_avg():
    assert compute_discount(10.0, 10.0) == pytest.approx(0.0)

def test_discount_negative_when_above_avg():
    assert compute_discount(12.0, 10.0) == pytest.approx(-0.2)

def test_discount_zero_when_avg_is_zero():
    assert compute_discount(5.0, 0.0) == pytest.approx(0.0)
```

Use `pytest.approx` for all floating-point comparisons — direct `==` on floats fails due to precision.

### Parametrize — test multiple cases without duplication

```python
@pytest.mark.parametrize("price, avg_price, expected", [
    (8.0,  10.0,  0.2),
    (10.0, 10.0,  0.0),
    (12.0, 10.0, -0.2),
    (5.0,   0.0,  0.0),   # edge case: zero avg
])
def test_compute_discount(price: float, avg_price: float, expected: float) -> None:
    assert compute_discount(price, avg_price) == pytest.approx(expected)
```

### Fixtures — shared setup without repetition

Fixtures are functions that provide test data or objects. Define them in `conftest.py` to share across test files.

```python
# conftest.py
import numpy as np
import pytest

@pytest.fixture
def sample_features() -> np.ndarray:
    """100 rows of 10 float32 features, seeded for reproducibility."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 10)).astype(np.float32)

@pytest.fixture
def sample_labels() -> np.ndarray:
    """Binary labels with ~30% positive rate."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=100, dtype=np.int32)

@pytest.fixture
def fake_model():
    """A model stub that returns a fixed score for any input."""
    class _FakeModel:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.column_stack([
                np.full(len(X), 0.3),
                np.full(len(X), 0.7),
            ])
    return _FakeModel()
```

```python
# test_metrics.py — fixtures injected by name
def test_lift_above_one_for_good_model(
    sample_features: np.ndarray,
    sample_labels: np.ndarray,
    fake_model,
) -> None:
    scores = fake_model.predict_proba(sample_features)[:, 1]
    result = compute_lift_metrics(sample_labels, scores)
    assert result["churn_lift"] > 0.0
```

### Fixture scope — control how often setup runs

```python
@pytest.fixture(scope="session")
def loaded_model():
    """Load the model once for the entire test session — expensive setup."""
    return joblib.load("tests/fixtures/model.pkl")

@pytest.fixture(scope="function")   # default — fresh copy per test
def mutable_array() -> np.ndarray:
    return np.zeros((10, 5), dtype=np.float32)
```

| Scope | When setup runs |
|-------|----------------|
| `function` | Before every test (default) |
| `class` | Once per test class |
| `module` | Once per test file |
| `session` | Once for the entire test run |

---

## Mocking — Isolate External Dependencies

Use `unittest.mock` to replace external dependencies (databases, APIs, file systems) with controlled fakes.

### Mock a database call

```python
from unittest.mock import MagicMock, patch
import numpy as np

def test_batch_scorer_calls_db_once() -> None:
    mock_db = MagicMock()
    mock_db.fetch.return_value = np.zeros((3, 10), dtype=np.float32)

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 3)

    scorer = BatchScorer(model=mock_model, db_client=mock_db)
    scores = scorer.score(["id_1", "id_2", "id_3"])

    mock_db.fetch.assert_called_once_with(["id_1", "id_2", "id_3"])
    assert scores.shape == (3,)
```

### Patch a module-level dependency

```python
from unittest.mock import patch

def test_load_config_reads_correct_path() -> None:
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = (
            '{"n_estimators": 100}'
        )
        config = load_config("config/exp/training_config.json")

    assert config["n_estimators"] == 100
```

### pytest-mock — cleaner mock syntax

```python
# pip install pytest-mock
def test_feature_store_timeout_raises(mocker) -> None:
    mocker.patch(
        "src.feature_store.FeatureStoreClient.fetch",
        side_effect=TimeoutError("connection timed out"),
    )
    client = FeatureStoreClient(host="localhost")

    with pytest.raises(TimeoutError, match="connection timed out"):
        client.fetch(["entity_1"])
```

---

## Testing Exceptions and Edge Cases

```python
def test_validate_features_raises_on_nan() -> None:
    features = np.array([[1.0, float("nan"), 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        validate_features(features)

def test_validate_features_raises_on_wrong_shape() -> None:
    features = np.zeros((10, 5))
    with pytest.raises(ValueError, match="Expected 10 features, got 5"):
        validate_features(features, expected_n_features=10)

def test_empty_input_returns_empty_output() -> None:
    result = compute_lift_metrics(
        actuals=np.array([]),
        scores=np.array([]),
    )
    assert result == {}
```

---

## ML-Specific Tests

### Test data quality

```python
import numpy as np
import pytest

def test_features_have_no_nulls(sample_features: np.ndarray) -> None:
    assert not np.isnan(sample_features).any(), "Features contain NaN values"

def test_features_have_correct_shape(sample_features: np.ndarray) -> None:
    assert sample_features.ndim == 2
    assert sample_features.shape[1] == 10

def test_labels_are_binary(sample_labels: np.ndarray) -> None:
    unique = np.unique(sample_labels)
    assert set(unique).issubset({0, 1}), f"Unexpected label values: {unique}"

def test_labels_have_both_classes(sample_labels: np.ndarray) -> None:
    assert 0 in sample_labels and 1 in sample_labels, (
        "Labels contain only one class — check data pipeline"
    )

def test_no_duplicate_entity_ids(entity_ids: list[str]) -> None:
    assert len(entity_ids) == len(set(entity_ids)), "Duplicate entity IDs found"
```

### Test model output properties

```python
def test_scores_are_probabilities(fake_model, sample_features: np.ndarray) -> None:
    scores = fake_model.predict_proba(sample_features)[:, 1]
    assert scores.min() >= 0.0, "Scores below 0"
    assert scores.max() <= 1.0, "Scores above 1"

def test_scores_have_correct_shape(fake_model, sample_features: np.ndarray) -> None:
    scores = fake_model.predict_proba(sample_features)[:, 1]
    assert scores.shape == (len(sample_features),)

def test_scores_are_float32(fake_model, sample_features: np.ndarray) -> None:
    scores = fake_model.predict_proba(sample_features)[:, 1].astype(np.float32)
    assert scores.dtype == np.float32
```

### Test monotonic constraints

```python
def test_lower_discount_yields_higher_score(trained_model) -> None:
    """Verify the monotonic constraint: lower discount → higher acceptance score."""
    base_features = np.zeros((1, 10), dtype=np.float32)
    discount_idx  = 2   # index of the discount feature

    discounts = np.linspace(-0.2, 0.2, 20)
    scores    = []
    for d in discounts:
        features = base_features.copy()
        features[0, discount_idx] = d
        scores.append(float(trained_model.predict_proba(features)[0, 1]))

    # Scores should be non-increasing as discount increases
    diffs = np.diff(scores)
    assert (diffs <= 1e-6).all(), (
        "Monotonic constraint violated: higher discount produced higher score"
    )
```

### Test for data leakage

```python
def test_no_future_features_in_training_data(
    train_df,
    label_timestamps: dict[str, str],
) -> None:
    """Verify no feature was computed using data after the label timestamp."""
    for entity_id, label_ts in label_timestamps.items():
        row = train_df[train_df["entity_id"] == entity_id].iloc[0]
        assert row["feature_timestamp"] <= label_ts, (
            f"Entity {entity_id}: feature computed after label timestamp — data leakage"
        )
```

---

## Testing FastAPI Endpoints

```python
from fastapi.testclient import TestClient
from unittest.mock import patch
import numpy as np

from main import app

client = TestClient(app)

def test_predict_returns_200_with_valid_input() -> None:
    with patch("main.ModelServer.model") as mock_model:
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        response = client.post("/predict", json={
            "entity_id": "e1",
            "features":  [0.5, 0.3, 0.8],
        })

    assert response.status_code == 200
    body = response.json()
    assert "score" in body
    assert 0.0 <= body["score"] <= 1.0

def test_predict_returns_422_on_nan_feature() -> None:
    response = client.post("/predict", json={
        "entity_id": "e1",
        "features":  [0.5, float("nan"), 0.8],
    })
    assert response.status_code == 422

def test_liveness_returns_ok() -> None:
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

---

## pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths     = ["tests"]
addopts       = "-v --tb=short --strict-markers"
markers       = [
    "slow: marks tests as slow (deselect with -m 'not slow')",
    "integration: marks integration tests requiring external services",
]
```

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run excluding slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

---

## Rules

- Every public function must have at least one test — one for the happy path, one for each edge case
- Tests must not require a database, network, or filesystem — inject fakes for all external dependencies
- Use `pytest.approx` for all floating-point assertions
- Use `@pytest.mark.parametrize` instead of loops inside tests — each case gets its own pass/fail
- Seed all random operations in tests — `np.random.default_rng(42)` — results must be deterministic
- Test ML constraints explicitly — monotonicity, output range, no NaN — not just that the code runs
- Keep fixtures in `conftest.py` — shared setup belongs in one place, not duplicated across test files
- Mark slow or integration tests with `@pytest.mark.slow` / `@pytest.mark.integration` — run them separately in CI
