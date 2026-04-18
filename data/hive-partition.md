# Hive Partitioning

Hive partitioning is a cross-platform standard for organising large datasets on distributed storage. It is important because it is understood natively by every major data processing tool — SQL warehouses, Spark, Python, and cloud storage — making it the lingua franca for pipeline outputs in ML systems.

---

## What Is Hive Partitioning

Hive partitioning physically organises files into a folder hierarchy based on column values. Instead of one flat directory of files, data is split into subfolders named `column=value`.

```
s3://bucket/regrade_sessions/
├── date=2025-01-13/
│   ├── part-00000.parquet
│   └── part-00001.parquet
├── date=2025-01-14/
│   └── part-00000.parquet
└── date=2025-01-15/
    └── part-00000.parquet
```

Multi-level partitioning adds nested subfolders:

```
s3://bucket/predictions/
├── environment=prod/
│   ├── date=2025-01-15/
│   │   └── part-00000.parquet
├── environment=exp/
│   ├── date=2025-01-15/
│   │   └── part-00000.parquet
```

The folder names encode the partition key and value — any tool that understands the Hive convention can read the correct subset of files without scanning everything.

---

## Why Hive Partitioning Is the Standard Pipeline Output

### Cross-platform compatibility

The same partitioned dataset can be read by every tool in the ML stack without conversion or reformatting:

| Tool | How it reads Hive partitions |
|------|------------------------------|
| BigQuery | External tables with `hive_partition_uri_prefix` — partition columns auto-inferred |
| Spark / PySpark | `spark.read.parquet(path)` — partition columns auto-inferred from folder names |
| AWS Athena | `MSCK REPAIR TABLE` or partition projection — folder names become filterable columns |
| Pandas / PyArrow | `pyarrow.dataset.dataset(path, partitioning="hive")` — partition columns added to schema |
| dbt | Sources point to the partitioned path; partition columns available in models |
| Hive | Native — the format originates here |

A training pipeline can write to S3 in Hive format, and the same data can be queried in BigQuery, processed in Spark, and loaded in Python — all without any transformation step.

### Partition pruning — only read what you need

Without partitioning, every query or data load scans all files regardless of the filter. With Hive partitioning, the tool resolves the filter to a folder path and reads only those files.

```
Filter: date = '2025-01-15'
Without partition → scan all files → read 90 days of data → filter in memory
With partition    → resolve to date=2025-01-15/ → read 1 day of data only
```

For a 90-day dataset, a single-day query reads 1/90th of the data. This directly reduces cost (cloud warehouses charge per byte scanned) and latency.

---

## Why It Matters in ML Pipelines

### Training data loading

ML training pipelines rarely need the full history on every run. Hive partitioning lets you load exactly the date range needed without pulling unnecessary data into memory.

```python
import pyarrow.dataset as ds

# Load only the training window — partition pruning happens at file level
dataset = ds.dataset(
    "s3://bucket/regrade_sessions/",
    partitioning="hive",
    format="parquet"
)

training_data = dataset.to_table(
    filter=(
        (ds.field("date") >= "2025-01-01") &
        (ds.field("date") <= "2025-01-31") &
        (ds.field("environment") == "prod")
    )
)
df = training_data.to_pandas()
```

Only the matching partition folders are opened — no full dataset scan.

### Incremental / delta training

Partition by date and you get delta loading for free — the pipeline reads only the new partition rather than reprocessing history.

```python
from datetime import date, timedelta

yesterday = date.today() - timedelta(days=1)
partition_path = f"s3://bucket/regrade_sessions/date={yesterday}/"

# Load only yesterday's new data
new_data = ds.dataset(partition_path, format="parquet").to_table().to_pandas()
```

### Experiment vs production isolation

Partition by `environment` to keep experiment and production data in the same storage path without mixing them. Training runs on `environment=prod`, experiments on `environment=exp`.

```python
prod_data = dataset.to_table(
    filter=ds.field("environment") == "prod"
)
exp_data = dataset.to_table(
    filter=ds.field("environment") == "exp"
)
```

### Reproducible training snapshots

Partition by `run_date` or `model_version` to snapshot the exact data used for each training run. This makes experiments reproducible — you can always reload the exact dataset a model was trained on.

```
s3://bucket/training_snapshots/
├── run_date=2025-01-15/
│   └── part-00000.parquet    ← exact data used for model v12
├── run_date=2025-02-01/
│   └── part-00000.parquet    ← exact data used for model v13
```

### Batch prediction output

Batch prediction pipelines write one partition per run date. Downstream consumers (dashboards, feature stores, evaluation jobs) always know exactly where to find the latest predictions without listing files or parsing filenames.

```python
import pandas as pd

predictions_df.to_parquet(
    f"s3://bucket/batch_predictions/date={run_date}/predictions.parquet",
    index=False
)
```

---

## Reading and Writing Hive Partitions in Python

### Writing with Pandas / PyArrow

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pa.Table.from_pandas(df)

# Write with Hive partitioning — creates date=YYYY-MM-DD/ subfolders automatically
pq.write_to_dataset(
    table,
    root_path="s3://bucket/regrade_sessions/",
    partition_cols=["date", "environment"],
)
```

### Writing with PySpark

```python
(
    df.write
      .mode("overwrite")
      .partitionBy("date", "environment")
      .parquet("s3://bucket/regrade_sessions/")
)
```

### Reading with PyArrow (partition pruning in Python)

```python
import pyarrow.dataset as ds

dataset = ds.dataset(
    "s3://bucket/regrade_sessions/",
    partitioning="hive",
    format="parquet"
)

# Partition columns (date, environment) are automatically added to the schema
table = dataset.to_table(
    filter=(
        (ds.field("date") >= "2025-01-01") &
        (ds.field("environment") == "prod")
    ),
    columns=["customer_id", "product_id", "predicted_score"]  # column pruning too
)
```

### Reading with PySpark

```python
df = spark.read.parquet("s3://bucket/regrade_sessions/")

# Partition columns are automatically available as DataFrame columns
df.filter(
    (df.date >= "2025-01-01") & (df.environment == "prod")
).select("customer_id", "product_id", "predicted_score")
```

---

## Reading Hive Partitions in SQL

### BigQuery external table

```sql
CREATE EXTERNAL TABLE regrade_sessions
WITH PARTITION COLUMNS (
    date        DATE,
    environment STRING
)
OPTIONS (
    format              = 'PARQUET',
    uris                = ['gs://bucket/regrade_sessions/*'],
    hive_partition_uri_prefix = 'gs://bucket/regrade_sessions/'
);

-- Partition pruning — only reads date=2025-01-15/ folder
SELECT customer_id, predicted_score
FROM regrade_sessions
WHERE date = '2025-01-15'
  AND environment = 'prod';
```

### AWS Athena

```sql
CREATE EXTERNAL TABLE regrade_sessions (
    customer_id     STRING,
    product_id      STRING,
    predicted_score DOUBLE
)
PARTITIONED BY (date STRING, environment STRING)
STORED AS PARQUET
LOCATION 's3://bucket/regrade_sessions/';

-- Register new partitions after writing
MSCK REPAIR TABLE regrade_sessions;

-- Query with partition filter — only scans matching folders
SELECT customer_id, predicted_score
FROM regrade_sessions
WHERE date = '2025-01-15'
  AND environment = 'prod';
```

---

## Choosing Partition Columns

| Partition column | When to use |
|-----------------|-------------|
| `date` | Almost always — the most common filter in ML pipelines |
| `environment` | When prod and exp data share the same storage path |
| `model_version` | When snapshotting predictions per model version for evaluation |
| `run_date` | When each pipeline run produces a distinct snapshot |
| `region` / `country` | When pipelines are scoped by geography |

### Rules

- Partition by the column you filter on most — if every query filters by `date`, partition by `date`
- Keep partition cardinality manageable — daily partitions over 2 years = ~730 folders, which is fine; partitioning by `customer_id` with millions of customers creates millions of tiny files (the small file problem)
- Avoid high-cardinality columns as partition keys (`customer_id`, `session_id`) — use them as cluster/sort keys instead
- Multi-level partitions (`date` + `environment`) are useful when both columns appear together in filters, but add nesting cost for queries that only filter on one

---

## Common Pitfalls

### Small file problem

Writing too many small files per partition degrades read performance — every file requires a separate I/O operation. This happens when partitioning by a high-cardinality column or when many small writes accumulate in the same partition.

```python
# ❌ Bad — writes one tiny file per customer_id partition
pq.write_to_dataset(table, root_path="s3://bucket/", partition_cols=["customer_id"])

# ✅ Good — coalesce before writing to control file count
import pyarrow.dataset as ds

pq.write_to_dataset(
    table,
    root_path="s3://bucket/regrade_sessions/",
    partition_cols=["date"],
    max_rows_per_file=500_000   # controls file size
)
```

In Spark, use `repartition` or `coalesce` before writing:

```python
(
    df.repartition(4, "date")   # 4 files per date partition
      .write
      .mode("overwrite")
      .partitionBy("date")
      .parquet("s3://bucket/regrade_sessions/")
)
```

### Missing partition filter in queries

A query without a partition column filter defeats pruning entirely — it scans all partitions.

```sql
-- ❌ Bad — scans all date partitions
SELECT customer_id, predicted_score
FROM regrade_sessions
WHERE environment = 'prod'   -- environment is not the partition column

-- ✅ Good — always include the partition column
SELECT customer_id, predicted_score
FROM regrade_sessions
WHERE date >= '2025-01-01'
  AND environment = 'prod'
```

### Overwriting a partition without isolating it

When reprocessing a single date, write only to that partition's path — not the root. Writing to the root with `overwrite` deletes all other partitions.

```python
# ❌ Bad — overwrites the entire dataset
df.write.mode("overwrite").partitionBy("date").parquet("s3://bucket/sessions/")

# ✅ Good — overwrite only the target partition
(
    df.filter(df.date == "2025-01-15")
      .write
      .mode("overwrite")
      .parquet("s3://bucket/sessions/date=2025-01-15/")
)
```

---

## Recommended File Format

Always pair Hive partitioning with **Parquet** (or ORC for Hive-native workloads):

| Format | Columnar | Compression | Cross-platform | Recommended |
|--------|----------|-------------|----------------|-------------|
| Parquet | ✅ | ✅ Snappy / ZSTD | ✅ Universal | ✅ Default choice |
| ORC | ✅ | ✅ ZLIB | ⚠️ Hive / Spark only | Use only for Hive-native pipelines |
| CSV | ❌ | ❌ | ✅ | ❌ No column pruning, no compression |
| JSON | ❌ | ❌ | ✅ | ❌ Verbose, slow to scan |

Parquet with Hive partitioning gives you both partition pruning (skip irrelevant folders) and column pruning (skip irrelevant columns within a file) — two independent layers of I/O reduction.
