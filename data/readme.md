# Data Processing Methods

Choosing the right data processing tool at each stage of an ML pipeline has a direct impact on speed, cost, and scalability. The general performance hierarchy is:

```
SQL (in-warehouse) > Spark SQL > Cython/C > NumPy ≈ Polars > Pandas > Pure Python
```

Each tool has a sweet spot — the key is matching the tool to the data size and the complexity of the operation.

---

## Comparison Overview

| Method | Best for | Data scale | Speed | Ease of use | ML ecosystem |
|--------|----------|-----------|-------|-------------|-------------|
| SQL (BigQuery, Redshift, Snowflake) | Aggregation, joins, filtering, feature engineering | TB+ | ★★★★★ | ★★★★ | ★★ |
| Spark | Distributed transforms when SQL cannot express the logic | GB–PB | ★★★★ | ★★★ | ★★★ |
| Cython | CPU-bound bottlenecks — faster than NumPy and Polars | Any | ★★★★★+ | ★★ | ★★ |
| NumPy | Vectorised numerical computation, array math | MB–GB (in-memory) | ★★★★ | ★★★★ | ★★★★★ |
| Polars | Fast tabular processing, modern Pandas alternative | MB–GB (in-memory) | ★★★★ | ★★★★ | ★★★ |
| Pandas (not preferred) | Exploration and prototyping only | MB–low GB (in-memory) | ★★ | ★★★★★ | ★★★★★ |
| Pure Python | Simple scripts, glue logic | KB–MB | ★ | ★★★★★ | ★★★★★ |

---

## SQL (In-Warehouse)

Best for: large-scale data processing, feature engineering, aggregation, joins.

```sql
SELECT customer_id,
       AVG(price) OVER (PARTITION BY product_id ORDER BY date ROWS 21 PRECEDING) AS rolling_avg
FROM regrade_sessions
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
```

| Pros | Cons |
|------|------|
| Runs where the data lives — no data movement | Limited expressiveness (no custom algorithms) |
| Query engine handles parallelism, partitioning, and optimisation | Dialect differences across platforms |
| Scales to terabytes without memory constraints | Harder to unit test than Python |
| Cost-efficient — cloud warehouses charge per query, not per compute hour | Complex logic requires nested CTEs or UDFs |

**When to use:** the data is in a warehouse and the operation is filtering, joining, aggregating, or windowing. Always prefer SQL over pulling data into Python for these tasks.

**When to avoid:** custom ML algorithms, iterative logic, or operations that require Python libraries.

---

## Spark

Best for: distributed processing when data exceeds single-machine memory.

### Spark SQL (preferred)

Always prefer Spark SQL over the PySpark DataFrame API when the logic can be expressed in SQL. The Catalyst optimiser handles both identically, but SQL is more readable, easier to review, and portable across platforms.

```python
df = spark.read.parquet("s3://bucket/features/")
df.createOrReplaceTempView("features")

result = spark.sql("""
    SELECT customer_id, AVG(price) AS avg_price
    FROM features
    WHERE date >= '2024-01-01'
    GROUP BY customer_id
""")
result.write.parquet("s3://bucket/output/")
```

### PySpark DataFrame API (when SQL is not enough)

Fall back to the DataFrame API only when the logic requires Python UDFs, complex custom transforms, or programmatic column generation that SQL cannot express cleanly.

```python
df = spark.read.parquet("s3://bucket/features/")
result = (
    df.filter(col("date") >= "2024-01-01")
      .groupBy("customer_id")
      .agg(avg("price").alias("avg_price"))
)
result.write.parquet("s3://bucket/output/")
```

| Pros | Cons |
|------|------|
| Scales horizontally — handles PB-scale data | High overhead for small datasets (cluster startup, shuffles) |
| Lazy evaluation — optimises the full execution plan before running | Debugging is harder than local tools |
| Spark SQL is readable and portable across platforms | Requires cluster management (EMR, Dataproc, Databricks) |
| Native integration with cloud storage (S3, GCS) | Slower than in-warehouse SQL for standard aggregations |

**When to use:** data is too large for a single machine (typically > 10–50 GB), or you need complex transforms across distributed storage. Prefer Spark SQL; use PySpark DataFrame API only when SQL cannot express the logic.

**When to avoid:** data fits in memory — NumPy or Polars will be faster and simpler. Don't use Spark for < 1 GB datasets.

---

## Cython / C Extensions

Best for: CPU-bound bottlenecks where NumPy vectorisation is not possible.

```cython
# example.pyx
def custom_monotonic_enforce(double[:] prices, double[:] floors):
    cdef int i
    cdef int n = prices.shape[0]
    for i in range(1, n):
        if prices[i] < floors[i]:
            prices[i] = floors[i]
        if prices[i] < prices[i - 1]:
            prices[i] = prices[i - 1]
    return prices
```

| Pros | Cons |
|------|------|
| Near-C speed — faster than NumPy and Polars for equivalent operations | Requires compilation step |
| Integrates directly with Python — call like a normal function | Harder to write, read, and debug than Python |
| Useful when NumPy vectorisation doesn't fit the algorithm | Adds build complexity (setup.py, Cython dependency) |
| No GIL for `nogil` blocks — true parallelism | Small community compared to NumPy/Pandas |
| Can outperform NumPy by avoiding intermediate array allocations | Maintenance burden — fewer developers can review Cython code |

Cython compiles to C and avoids Python interpreter overhead entirely. For loop-heavy algorithms and custom numerical logic, it is faster than both NumPy and Polars because it eliminates intermediate array allocations and function call overhead.

**When to use:** you have a proven bottleneck that is loop-based and cannot be vectorised with NumPy, or when NumPy's intermediate arrays cause memory pressure. Profile first — don't optimise prematurely.

**When to avoid:** the operation can be expressed as NumPy array operations. NumPy is easier to maintain and fast enough for most cases.

---

## NumPy

Best for: vectorised numerical computation, array math, batch operations.

```python
import numpy as np

# Vectorised — fast
cum_floors = np.maximum.accumulate(floors)
preds = np.clip(preds, cum_floors, ceilings)
preds = np.maximum.accumulate(preds)

# Loop — slow (avoid this)
for i in range(len(preds)):
    preds[i] = max(preds[i], floors[i])
```

| Pros | Cons |
|------|------|
| 10–100x faster than Python loops, comparable to Polars for numerical tasks | Requires thinking in array operations, not loops |
| Much faster than Pandas for any numerical operation | Not suited for heterogeneous tabular data (use Polars) |
| Memory-efficient contiguous arrays | Operations on non-numeric data are awkward |
| Foundation of the entire Python ML stack | Intermediate arrays can spike memory usage |
| Broadcasting eliminates explicit loops | No built-in I/O for tabular formats (CSV, Parquet) |

NumPy operates on contiguous memory with C-level loops internally — for numerical work it is significantly faster than Pandas and performs on par with Polars. Prefer NumPy over Pandas whenever the data is numeric.

**When to use:** numerical computation on arrays — math, statistics, linear algebra, element-wise operations. All pricing math, prediction scoring, and constraint enforcement should use NumPy.

**When to avoid:** tabular data with mixed types (strings, dates, numerics) — use Polars. Data larger than memory — use SQL or Spark.

---

## Pandas (Not Preferred for Production)

Pandas is widely used but **not preferred** for production ML pipelines. It is slow, memory-hungry, and single-threaded. Use it for quick exploration and prototyping only — for production workloads, prefer NumPy (numerical), Polars (tabular), or SQL (large-scale).

```python
import pandas as pd

df = pd.read_parquet("features.parquet")
result = (
    df[df["date"] >= "2024-01-01"]
      .groupby("customer_id")["price"]
      .mean()
      .reset_index()
)
```

| Pros | Cons |
|------|------|
| Intuitive API for tabular data — filter, group, join, pivot | Slow — single-threaded, significantly slower than NumPy and Polars |
| Excellent for exploration and prototyping | High memory usage — typically 2–5x the raw data size |
| Rich I/O (CSV, Parquet, SQL, Excel, JSON) | Not suited for production performance-critical code |
| Tight integration with scikit-learn, XGBoost, plotting | Chained operations can be hard to debug |

**When to use:** quick data exploration in notebooks, one-off analysis, or when a downstream library strictly requires a Pandas DataFrame.

**When to avoid:** production pipelines, numerical computation (use NumPy), tabular processing where speed matters (use Polars), large-scale data (use SQL or Spark).

---

## Polars

Best for: fast tabular processing as a modern Pandas alternative.

```python
import polars as pl

df = pl.read_parquet("features.parquet")
result = (
    df.filter(pl.col("date") >= "2024-01-01")
      .group_by("customer_id")
      .agg(pl.col("price").mean().alias("avg_price"))
)
```

| Pros | Cons |
|------|------|
| 5–20x faster than Pandas for most operations | Smaller community and ecosystem than Pandas |
| Lazy evaluation — optimises query plan before execution | API differs from Pandas — learning curve for existing teams |
| Multi-threaded by default | Some scikit-learn/XGBoost integrations expect Pandas DataFrames |
| Lower memory usage than Pandas | Less mature — fewer tutorials, Stack Overflow answers |

**When to use:** tabular processing where Pandas is too slow but the data still fits on one machine. Good for feature engineering pipelines that run repeatedly.

**When to avoid:** quick exploration in notebooks where Pandas familiarity speeds things up. Interop-heavy workflows where downstream tools expect Pandas DataFrames.

---

## Pure Python

Best for: glue logic, config parsing, simple scripts.

| Pros | Cons |
|------|------|
| No dependencies — always available | 10–100x slower than NumPy for numerical operations |
| Easy to read and write | Not suited for any data processing at scale |
| Good for orchestration, file I/O, API calls | Python loops over data are the #1 ML performance anti-pattern |

**When to use:** orchestration, config loading, API calls, file management. Never for numerical computation or data processing.

---

## Decision Framework

```
Data in a warehouse (TB+)?          ──yes──→  SQL
        │ no
Data too large for one machine?     ──yes──→  Spark
        │ no
Tabular data, mixed types?          ──yes──→  Polars (or Pandas for quick exploration only)
        │ no
Numerical array operations?         ──yes──→  NumPy
        │ no
Loop-heavy, proven bottleneck?      ──yes──→  Cython / C extension
        │ no
Simple glue logic?                  ──yes──→  Pure Python
```

---

## Performance Comparison (Approximate)

Operation: group-by aggregation on 10M rows, single machine.

| Method | Time | Relative speed |
|--------|------|---------------|
| SQL (in-warehouse) | ~1s | Fastest (runs on cluster) |
| Cython | ~0.1s | ~50x Pandas |
| NumPy | ~0.3s | ~15x Pandas |
| Polars | ~0.5s | ~10x Pandas |
| Pandas | ~5s | Baseline (not preferred) |
| Pure Python loop | ~60s | ~12x slower than Pandas |

*Times are illustrative — actual performance depends on hardware, data shape, and operation complexity.*

---

## Anti-Patterns

- **Pulling TB-scale data into Pandas** — process it in SQL or Spark where it lives
- **Python loops over arrays** — use NumPy vectorised operations; this is the single biggest performance mistake in ML code
- **Spark for small datasets** — cluster overhead makes Spark slower than Pandas for < 1 GB
- **Pandas in production** — Pandas is not preferred for production pipelines; use NumPy, Polars, or SQL instead
- **Pandas for pure numerical math** — extract the array with `.values` and use NumPy, which is significantly faster
- **Premature Cython optimisation** — profile first; NumPy vectorisation solves most bottlenecks without the complexity
- **Mixing tools unnecessarily** — don't chain SQL → Spark → Pandas → NumPy when SQL alone can do the job

---

## Pydantic BaseSettings — Typed Config from Environment Variables

`pydantic-settings` replaces manual `os.getenv()` chains with a typed, validated settings model. Every environment variable is declared once with its type, default, and constraints — missing or malformed values raise a clear error at startup rather than a cryptic failure later.

```bash
pip install pydantic-settings
```

### Replacing os.getenv with BaseSettings

```python
# ❌ Before — manual, untyped, no validation
import os

run_environment = os.getenv("RUN_ENVIRONMENT", "exp")
is_deployment   = os.getenv("IS_DEPLOYMENT", "false").lower() == "true"
port            = int(os.getenv("PORT", "8080"))
artifact_root   = os.getenv("ARTIFACT_LOCATION")
if not artifact_root:
    raise EnvironmentError("ARTIFACT_LOCATION is required")
```

```python
# ✅ After — typed, validated, self-documenting
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """
    Application runtime configuration loaded from environment variables.

    All fields are validated at startup. Missing required fields raise
    a ValidationError with a clear message before the app accepts traffic.
    """

    model_config = SettingsConfigDict(
        env_file="env/local.env",   # loaded only if the variable is not already set
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    run_environment:   str  = Field(default="exp", pattern="^(exp|stag|ref1|prod)$")
    is_deployment:     bool = False
    port:              int  = Field(default=8080, ge=1, le=65535)
    artifact_location: str                       # required — no default
    rds_host:          str | None = None         # optional override
    log_level:         str  = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")


# Load once at startup — raises ValidationError immediately if config is invalid
settings = AppSettings()
```

### Nested settings for complex configs

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class FeatureStoreConfig(BaseModel):
    host:      str
    port:      int = 5432
    database:  str = "features"
    pool_size: int = Field(default=5, ge=1, le=20)

class ModelConfig(BaseModel):
    name:    str
    version: int
    path:    str

class ServiceSettings(BaseSettings):
    feature_store: FeatureStoreConfig
    model:         ModelConfig
    environment:   str = "exp"

# Set via environment variables using double-underscore for nesting:
# FEATURE_STORE__HOST=localhost
# FEATURE_STORE__PORT=5432
# MODEL__NAME=churn_model
# MODEL__VERSION=12
settings = ServiceSettings()
```

### Accessing settings

```python
# ✅ Pass settings as a dependency — never import as a global in business logic
def load_model(settings: AppSettings) -> object:
    return joblib.load(settings.artifact_location)


# ✅ In FastAPI — use dependency injection
from functools import lru_cache
from fastapi import Depends

@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()

@app.get("/metadata")
def metadata(settings: AppSettings = Depends(get_settings)) -> dict:
    return {"environment": settings.run_environment, "port": settings.port}
```

### Rules

- Declare every environment variable in `BaseSettings` — no bare `os.getenv()` calls in application code
- Use `Field(pattern=...)` for string enums — catches typos in environment variable values at startup
- Mark required fields with no default — the app fails fast with a clear error if they are missing
- Use `lru_cache` on the settings factory in FastAPI — settings are loaded once, not per request
- Keep `env_file` as a fallback only — in CI and production, variables are injected by the environment; the file is for local development
