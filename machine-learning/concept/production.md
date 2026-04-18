# Production Best Practices

A model in production faces conditions that never exist in training: missing features, schema changes, unexpected input distributions, and latency constraints. Every production ML system must be designed to handle these gracefully.

---

## 1. Model Versioning

Every deployed model must be uniquely identified and traceable back to the code, data, and config that produced it.

### Version scheme

```
{model_name}/v{version}/{artefact}

e.g.
churn_model/v12/model.pkl
churn_model/v12/metadata.json
churn_model/v12/feature_schema.json
```

### Metadata to store alongside every model

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class ModelMetadata:
    model_name:       str
    version:          int
    git_sha:          str
    data_snapshot_id: str
    training_date:    str
    feature_names:    list[str]
    feature_schema:   dict[str, str]   # feature_name → dtype
    hyperparameters:  dict
    metrics:          dict[str, float]
    config_hash:      str

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelMetadata":
        with open(path) as f:
            return cls(**json.load(f))
```

### Rules

- Never overwrite a published model version — versions are immutable
- Store metadata alongside the model artefact — not in a separate system
- Load the feature schema from metadata at serving time — never hardcode feature names in serving code
- Tag the deployed version in your monitoring system — every metric should be attributable to a model version

---

## 2. Input Validation

Input validation is the first line of defence against silent failures. A model that receives malformed input will produce wrong predictions silently — no error, no alert, just bad output.

```python
import numpy as np

class InputValidator:
    """
    Validate inference requests against the training feature schema.

    Parameters
    ----------
    feature_names : list[str]
        Expected feature names in order.
    feature_ranges : dict[str, tuple[float, float]]
        Expected (min, max) range per feature from training data.
    """

    def __init__(
        self,
        feature_names: list[str],
        feature_ranges: dict[str, tuple[float, float]],
    ) -> None:
        self.feature_names  = feature_names
        self.feature_ranges = feature_ranges

    def validate(self, features: np.ndarray) -> None:
        """
        Validate feature array against schema.

        Raises
        ------
        ValueError
            If shape, nulls, or feature ranges are invalid.
        """
        if features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
            )

        if np.isnan(features).any():
            null_cols = [
                self.feature_names[i]
                for i in range(features.shape[1])
                if np.isnan(features[:, i]).any()
            ]
            raise ValueError(f"Null values in features: {null_cols}")

        for i, name in enumerate(self.feature_names):
            if name not in self.feature_ranges:
                continue
            lo, hi = self.feature_ranges[name]
            col    = features[:, i]
            if col.min() < lo or col.max() > hi:
                raise ValueError(
                    f"Feature '{name}' out of expected range [{lo}, {hi}]: "
                    f"got [{col.min():.4f}, {col.max():.4f}]"
                )
```

### Warn on distribution shift, fail on schema errors

Not all validation failures should raise an exception. Schema errors (wrong number of features, nulls) should fail hard. Distribution warnings (values outside training range) should log a warning and continue — the model may still produce a reasonable prediction.

```python
def validate_or_warn(
    features: np.ndarray,
    validator: InputValidator,
    logger,
    request_id: str,
) -> None:
    try:
        validator.validate(features)
    except ValueError as e:
        logger.warning("Input distribution warning. request_id=%s error=%s", request_id, e)
        # continue — do not raise for range warnings
```

---

## 3. Fallback Logic

Every model endpoint must have a fallback for when the model cannot produce a prediction — missing features, timeout, or unexpected error. A hard failure in production is always worse than a safe default.

```python
def predict_with_fallback(
    model,
    features: np.ndarray | None,
    fallback_score: float,
    logger,
    request_id: str,
) -> float:
    """
    Return model prediction, falling back to a safe default on any failure.

    Parameters
    ----------
    fallback_score : float
        Safe default score returned when the model cannot predict.
    """
    if features is None:
        logger.warning(
            "Step: predict FALLBACK — features unavailable. request_id=%s", request_id
        )
        return fallback_score

    try:
        score = float(model.predict_proba(features)[0, 1])
        logger.info("Step: predict COMPLETE. score=%.4f request_id=%s", score, request_id)
        return score
    except Exception as e:
        logger.error(
            "ERROR500: predict failed. request_id=%s error=%s", request_id, e
        )
        return fallback_score
```

### Fallback hierarchy

Define a hierarchy of fallbacks — each level is safer but less personalised:

```
1. Full model prediction          — all features available
2. Partial model prediction       — subset of features, degraded accuracy
3. Segment-level average          — use cohort statistics (e.g. product average)
4. Global average                 — use population-level statistic
5. Hard-coded safe default        — last resort, always available
```

---

## 4. Serving Patterns

### Batch prediction

Pre-compute predictions for all entities on a schedule and store results. Serving is a simple lookup — no model at request time.

```
scheduler → batch predict → write to feature store → serve from store
```

```python
def run_batch_prediction(
    model,
    entity_ids: list[str],
    feature_store,
    output_store,
    run_date: str,
    logger,
) -> None:
    features = feature_store.fetch_batch(entity_ids)
    scores   = model.predict_proba(features)[:, 1]

    output_store.write(
        entity_ids=entity_ids,
        scores=scores,
        run_date=run_date,
    )
    logger.info(
        "batch_prediction COMPLETE. n=%d run_date=%s", len(entity_ids), run_date
    )
```

**When to use:** predictions are needed for a known set of entities, latency is not critical, features are expensive to compute at request time.

### Online prediction

Score each request in real time. Features are fetched and the model is called synchronously.

```
request → fetch features → validate → predict → respond
```

**When to use:** personalisation requires real-time context, the entity set is unknown in advance, or predictions must reflect the latest data.

### Caching

Cache predictions for entities that are scored repeatedly within a short window:

```python
import redis
import json

class PredictionCache:
    def __init__(self, client: redis.Redis, ttl_seconds: int = 300) -> None:
        self.client      = client
        self.ttl_seconds = ttl_seconds

    def get(self, cache_key: str) -> float | None:
        value = self.client.get(cache_key)
        return float(value) if value is not None else None

    def set(self, cache_key: str, score: float) -> None:
        self.client.setex(cache_key, self.ttl_seconds, str(score))
```

---

## 5. Latency Management

Every component in the serving path adds latency. Define a latency budget and allocate it across components.

```
total budget: 100ms
├── feature fetch:  40ms
├── model predict:  10ms
├── postprocess:     5ms
└── network:        45ms
```

```python
import time

def predict_with_timing(
    model,
    features: np.ndarray,
    logger,
    request_id: str,
) -> tuple[float, float]:
    start = time.perf_counter()
    score = float(model.predict_proba(features)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict latency_ms=%.2f request_id=%s", latency_ms, request_id
    )
    return score, latency_ms
```

### Rules

- Set a timeout on every external call (feature store, cache, model) — never wait indefinitely
- Log latency at every step — not just end-to-end
- Alert when P95 latency exceeds the budget — not just when the service is down
- Use batch prediction when real-time latency cannot be met

---

## 6. Model Loading

Load the model once at startup — never on every request.

```python
class ModelServer:
    """
    Singleton model server — loads artefacts once at startup.

    Notes
    -----
    All artefacts are loaded in __init__ via _load_artefacts().
    Subsequent calls to predict() reuse the loaded objects.
    """

    def __init__(self, artefact_root: str, metadata_path: str) -> None:
        self.metadata  = ModelMetadata.load(metadata_path)
        self._model    = None
        self._load_artefacts(artefact_root)

    def _load_artefacts(self, artefact_root: str) -> None:
        import joblib
        import os
        self._model = joblib.load(os.path.join(artefact_root, "model.pkl"))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(features)[:, 1]
```

### Rules

- Load models at application startup — not at request time
- Validate the model loaded successfully before accepting traffic — fail the health check if loading fails
- Store the model version in the server instance — log it with every prediction
- Never reload the model mid-request — use a blue/green or atomic swap pattern for updates
