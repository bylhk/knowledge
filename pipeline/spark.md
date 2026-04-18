# Apache Spark

Apache Spark is a distributed processing engine for large-scale data transformation, feature engineering, and batch prediction. It processes data that does not fit on a single machine by distributing work across a cluster.

---

## When to Use Spark

```
Data fits in memory (< ~10 GB)    → Polars or DuckDB — faster, simpler
Data exceeds one machine (> 10 GB) → Spark
Data is in a warehouse             → SQL first — cheaper and faster than Spark for standard aggregations
Custom ML transforms at scale      → Spark + PySpark DataFrame API
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **RDD** | Resilient Distributed Dataset — low-level distributed collection. Avoid in favour of DataFrames |
| **DataFrame** | Distributed table with named columns and schema — the primary API |
| **Spark SQL** | SQL interface over DataFrames — preferred for readable, portable queries |
| **Catalyst** | Query optimiser — rewrites and optimises execution plans automatically |
| **Lazy evaluation** | Transformations are not executed until an action (`.write`, `.collect`) is called |
| **Partition** | A chunk of data processed by one executor — the unit of parallelism |

---

## Spark SQL vs DataFrame API

Always prefer Spark SQL when the logic can be expressed in SQL. The Catalyst optimiser handles both identically, but SQL is more readable and portable.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("feature-pipeline").getOrCreate()

df = spark.read.parquet("s3://bucket/sessions/")
df.createOrReplaceTempView("sessions")

# ✅ Preferred — Spark SQL
result = spark.sql("""
    SELECT
        customer_id,
        DATE(session_date)                              AS date,
        COUNT(*)                                        AS session_count,
        SUM(predicted_score)                            AS score_sum,
        SUM(predicted_score * predicted_score)          AS score_sum_sq,
        COUNTIF(accepted = 1)                           AS acceptance_count
    FROM sessions
    WHERE session_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    GROUP BY customer_id, DATE(session_date)
""")

result.write.mode("overwrite").partitionBy("date").parquet("s3://bucket/features/")
```

Fall back to the DataFrame API only when the logic requires Python UDFs or programmatic column generation:

```python
from pyspark.sql import functions as F

# DataFrame API — use when SQL cannot express the logic
result = (
    df.filter(F.col("session_date") >= F.date_sub(F.current_date(), 90))
      .groupBy("customer_id", F.to_date("session_date").alias("date"))
      .agg(
          F.count("*").alias("session_count"),
          F.sum("predicted_score").alias("score_sum"),
      )
)
```

---

## Feature Engineering Pipeline

```python
from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder \
    .appName("feature-engineering") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Load raw data — partition pruning via predicate pushdown
raw = spark.read.parquet("s3://bucket/raw_sessions/") \
    .filter(F.col("date") >= "2025-01-01")

# Rolling average — window function
window_21d = Window.partitionBy("customer_id", "product_id") \
    .orderBy("date") \
    .rowsBetween(-20, 0)

features = raw.withColumn(
    "rolling_avg_score",
    F.avg("predicted_score").over(window_21d),
).withColumn(
    "discount",
    (F.col("avg_recommended_price") - F.col("predicted_score"))
    / F.col("avg_recommended_price"),
)

# Write as Hive-partitioned Parquet
features.repartition(4, "date") \
    .write \
    .mode("overwrite") \
    .partitionBy("date") \
    .parquet("s3://bucket/features/")
```

---

## Batch Prediction at Scale

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType
import joblib

# Load model once per executor using a broadcast variable
model_broadcast = spark.sparkContext.broadcast(
    joblib.load("/tmp/model.pkl")
)

@pandas_udf(FloatType())
def predict_udf(*feature_cols: pd.Series) -> pd.Series:
    """
    Apply the model to each partition as a Pandas batch.
    pandas_udf avoids Python row-by-row overhead — processes a full partition at once.
    """
    import numpy as np
    features = np.stack([col.values for col in feature_cols], axis=1).astype("float32")
    model    = model_broadcast.value
    return pd.Series(model.predict_proba(features)[:, 1])

FEATURE_COLS = ["feature_a", "feature_b", "feature_c"]

predictions = features_df.withColumn(
    "score",
    predict_udf(*[F.col(c) for c in FEATURE_COLS]),
)

predictions.select("customer_id", "product_id", "score", "date") \
    .repartition(4, "date") \
    .write \
    .mode("overwrite") \
    .partitionBy("date") \
    .parquet("s3://bucket/predictions/")
```

---

## Performance Tuning

### Partitioning

```python
# Check current partition count
print(df.rdd.getNumPartitions())

# Repartition before a wide shuffle (join, groupBy)
df = df.repartition(200, "customer_id")

# Coalesce to reduce partitions before writing (no shuffle)
df.coalesce(4).write.parquet(...)

# Rule of thumb: 128–256 MB per partition
# Too few partitions → underutilised cluster
# Too many partitions → scheduling overhead
```

### Broadcast joins

When one side of a join is small (< ~10 MB), broadcast it to all executors to avoid a shuffle:

```python
from pyspark.sql.functions import broadcast

# ✅ Broadcast the small lookup table — no shuffle
result = large_df.join(
    broadcast(small_lookup_df),
    on="product_id",
    how="left",
)
```

### Caching

Cache a DataFrame that is used multiple times in the same job:

```python
# Cache in memory (default)
features_df.cache()

# Cache to disk if too large for memory
features_df.persist(StorageLevel.DISK_ONLY)

# Always unpersist when done
features_df.unpersist()
```

### Avoid UDFs where possible

Python UDFs break Catalyst optimisation and serialise data through the Python interpreter row by row. Use built-in Spark functions or `pandas_udf` instead:

```python
# ❌ Bad — Python UDF, row-by-row, breaks Catalyst
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

@udf(FloatType())
def compute_discount(price, avg_price):
    return (avg_price - price) / avg_price if avg_price else None

# ✅ Good — built-in functions, Catalyst-optimised
result = df.withColumn(
    "discount",
    (F.col("avg_price") - F.col("price")) / F.col("avg_price"),
)
```

---

## Spark on Cloud

| Platform | Service | Notes |
|----------|---------|-------|
| AWS | EMR | Managed Spark, integrates with S3 and Glue |
| GCP | Dataproc | Managed Spark, integrates with GCS and BigQuery |
| Azure | HDInsight / Synapse | Managed Spark on Azure |
| Any | Databricks | Managed Spark with Delta Lake, Unity Catalog, MLflow |

### Databricks-specific features

- **Delta Lake** — ACID transactions on Parquet, time travel, schema enforcement
- **Unity Catalog** — centralised data governance and lineage
- **MLflow** — experiment tracking and model registry integrated into the platform
- **Auto Loader** — incremental file ingestion from cloud storage with schema inference

---

## Rules

- Prefer Spark SQL over the DataFrame API — same performance, more readable
- Never use Python row-level UDFs — use built-in functions or `pandas_udf`
- Always partition output by date — enables downstream partition pruning
- Broadcast small lookup tables — avoids expensive shuffles
- Set `spark.sql.shuffle.partitions` explicitly — default of 200 is wrong for most jobs
- Cache only DataFrames used multiple times in the same job — unpersist when done
- Profile with Spark UI before optimising — the query plan shows where time is spent
