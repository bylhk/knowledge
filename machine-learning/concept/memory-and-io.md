# Memory Control & Data I/O

Loading a full dataset into memory is the most common cause of OOM crashes, slow pipelines, and unnecessarily expensive cloud jobs. Every stage of an ML pipeline — data loading, training, evaluation, and prediction — must be designed to process data in bounded chunks rather than all at once.

---

## Core Principle — Never Load the Full Dataset

```
❌  load all → process all → write all
✅  load chunk → process chunk → write chunk → repeat
```

A dataset that fits in memory today will not fit when the data doubles. Design for streaming and batching from the start — retrofitting it later is expensive.

---

## 1. File Formats

### Parquet

Parquet is a columnar binary format designed for analytical workloads. It is the standard format for ML data pipelines.

**How it works:** data is stored column-by-column rather than row-by-row. Reading only the columns you need skips the rest entirely at the file level — no parsing, no memory allocation for unused columns.

```
Row-based (CSV):    [row1_col1, row1_col2, row1_col3, row2_col1, row2_col2, ...]
Columnar (Parquet): [col1_row1, col1_row2, ..., col2_row1, col2_row2, ..., col3_row1, ...]
```

| Pros | Cons |
|------|------|
| Columnar — read only the columns you need | Not human-readable |
| Compressed per column — Snappy/ZSTD, typically 5–10x smaller than CSV | Requires a library to read/write |
| Schema embedded — dtypes preserved on read | Slower row-level random access than row-based formats |
| Splittable — multiple files can be read in parallel | Append is not efficient — write new files instead |
| Native partition pruning with Hive partitioning | Row group size must be tuned for optimal read performance |
| Supported everywhere — Spark, BigQuery, Athena, PyArrow, DuckDB | |

**Best for:** training data, feature stores, batch prediction outputs, any dataset > 100MB that is read repeatedly.

```python
import pyarrow.parquet as pq
import pyarrow as pa

# Write with explicit schema and compression
table = pa.Table.from_arrays(
    [pa.array(features[:, i], type=pa.float32()) for i in range(features.shape[1])],
    names=feature_names,
)
pq.write_table(
    table,
    "features.parquet",
    compression="snappy",       # fast decompression — good for ML reads
    row_group_size=100_000,     # tune for your read pattern (see section 4)
)

# Read only the columns you need — other columns never touch memory
table = pq.read_table("features.parquet", columns=["score", "discount", "tenure"])
```

---

### Arrow / Feather (IPC format)

Arrow is an in-memory columnar format. Feather (`.feather` or `.arrow`) is Arrow serialised to disk — it is essentially a memory dump of an Arrow table.

**How it works:** the on-disk layout matches the in-memory layout exactly. Reading a Feather file is a memory-map operation — the OS maps the file into the process's address space without copying. The data is available immediately with zero deserialisation cost.

| Pros | Cons |
|------|------|
| Near-zero read latency — memory-mapped, no deserialisation | Larger files than Parquet — minimal compression |
| Zero-copy reads — data shared between processes without copying | Not splittable across workers without manual partitioning |
| Ideal for IPC — share data between Python processes or languages | Not suitable for long-term storage — format evolves |
| Preserves all Arrow types including nested structs and lists | Less ecosystem support than Parquet for cloud warehouses |
| Fast writes — no compression overhead | Not supported natively by BigQuery, Athena, Redshift |

**Best for:** inter-process communication, passing data between pipeline steps in the same machine, caching intermediate results that are read many times in a session, and datasets that fit on local disk and are read repeatedly within a single job.

```python
import pyarrow as pa
import pyarrow.feather as feather

# Write — fast, minimal compression
feather.write_feather(table, "features.feather", compression="uncompressed")

# Read — memory-mapped, near-zero latency
table = feather.read_table("features.feather", memory_map=True)

# Zero-copy: multiple processes can read the same file without copying
# Process A writes → Process B reads via memory map → shared physical memory
```

---

### Format Decision Guide

| Scenario | Use |
|----------|-----|
| Long-term storage, cloud warehouse, cross-platform | Parquet |
| Repeated reads within a single job on one machine | Feather (memory-mapped) |
| Passing data between pipeline steps on the same machine | Feather |
| Data shared with BigQuery / Athena / Redshift | Parquet only |
| Hive-partitioned dataset on S3 / GCS | Parquet |
| Intermediate cache during training | Feather |
| Final model training data | Parquet |

---

### Best File Size

File size affects read parallelism, memory pressure, and I/O efficiency.

| File size | Problem |
|-----------|---------|
| < 10 MB | Too many small files — high metadata overhead, slow parallel reads |
| 10 MB – 1 GB | Good range for most workloads |
| > 1 GB | Single file becomes a bottleneck — cannot parallelise reads |

**Target: 128 MB – 512 MB per file** for Parquet on distributed storage (S3, GCS). This matches the default HDFS block size and allows efficient parallel reads across workers.

```python
import pyarrow.parquet as pq

# Control file size by setting max_rows_per_file
pq.write_to_dataset(
    table,
    root_path="s3://bucket/features/",
    partition_cols=["date"],
    max_rows_per_file=500_000,    # tune based on row width to hit ~256MB per file
)
```

For Parquet **row groups** (internal to a file): target 50K–100K rows per row group. Smaller row groups allow finer-grained column pruning; larger row groups compress better.

---

## 2. Batch Loading — Never Load the Full Dataset

### PyArrow dataset API — lazy, streaming reads

```python
import pyarrow.dataset as ds
import pyarrow as pa

# Open the dataset — no data loaded yet
dataset = ds.dataset(
    "s3://bucket/features/",
    partitioning="hive",
    format="parquet",
)

# Iterate in batches — only one batch in memory at a time
for batch in dataset.to_batches(
    batch_size=50_000,
    columns=["feature_a", "feature_b", "label"],
    filter=(ds.field("date") >= "2025-01-01"),
):
    features = batch.column("feature_a").to_pylist()
    # process batch — previous batch is garbage collected
```

### PyArrow ParquetFile — row group iteration

Each Parquet file is divided into row groups. Iterate row groups to process one chunk at a time:

```python
import pyarrow.parquet as pq
import numpy as np

pf = pq.ParquetFile("features.parquet")

for batch in pf.iter_batches(batch_size=50_000, columns=["score", "label"]):
    scores = batch.column("score").to_pyarray()
    labels = batch.column("label").to_pyarray()
    # process — only this batch is in memory
```

---

## 3. Batch Training

### PyTorch IterableDataset — streaming from disk

For large training datasets that do not fit in memory, use `IterableDataset` to stream directly from Parquet or Feather files:

```python
import pyarrow.dataset as ds
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class ParquetStreamDataset(IterableDataset):
    """
    Stream training batches directly from a partitioned Parquet dataset.

    Never loads the full dataset into memory — one batch at a time.

    Parameters
    ----------
    path : str
        Root path of the Hive-partitioned Parquet dataset.
    feature_cols : list[str]
        Feature column names.
    label_col : str
        Label column name.
    batch_size : int
        Rows per batch yielded to the DataLoader.
    date_filter : str | None
        Optional date lower bound for partition pruning.
    """

    def __init__(
        self,
        path: str,
        feature_cols: list[str],
        label_col: str,
        batch_size: int = 10_000,
        date_filter: str | None = None,
    ) -> None:
        self.path         = path
        self.feature_cols = feature_cols
        self.label_col    = label_col
        self.batch_size   = batch_size
        self.date_filter  = date_filter

    def __iter__(self):
        dataset = ds.dataset(self.path, partitioning="hive", format="parquet")

        scan_filter = None
        if self.date_filter:
            scan_filter = ds.field("date") >= self.date_filter

        for batch in dataset.to_batches(
            batch_size=self.batch_size,
            columns=self.feature_cols + [self.label_col],
            filter=scan_filter,
        ):
            features = np.stack(
                [batch.column(c).to_pylist() for c in self.feature_cols], axis=1
            ).astype(np.float32)
            labels = np.array(batch.column(self.label_col).to_pylist(), dtype=np.float32)

            # zero-copy: from_numpy shares memory with the numpy array
            yield torch.from_numpy(features), torch.from_numpy(labels)


# Usage — DataLoader handles batching and worker processes
dataset    = ParquetStreamDataset("s3://bucket/features/", feature_cols, "label")
dataloader = DataLoader(dataset, batch_size=None, num_workers=2)

for features, labels in dataloader:
    loss = model(features, labels)
    loss.backward()
```

---

## 4. Batch Evaluation

Never load all predictions and labels into memory to compute metrics. Accumulate sufficient statistics across batches and compute the final metric once.

```python
import numpy as np

class BatchMetricAccumulator:
    """
    Accumulate additive metric elements across batches.

    Computes AUC-approximation, mean prediction, and acceptance rate
    without holding all predictions in memory simultaneously.
    """

    def __init__(self) -> None:
        self.n_total    = 0
        self.n_positive = 0
        self.score_sum  = 0.0
        self.score_sum_sq = 0.0

    def update(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self.n_total      += len(labels)
        self.n_positive   += int(labels.sum())
        self.score_sum    += float(scores.sum())
        self.score_sum_sq += float((scores ** 2).sum())

    def compute(self) -> dict[str, float]:
        mean = self.score_sum / self.n_total
        var  = self.score_sum_sq / self.n_total - mean ** 2
        return {
            "acceptance_rate": self.n_positive / self.n_total,
            "mean_score":      mean,
            "std_score":       float(np.sqrt(max(var, 0))),
            "n_total":         self.n_total,
        }


# Evaluate across batches — only one batch in memory at a time
accumulator = BatchMetricAccumulator()

for batch in dataset.to_batches(batch_size=50_000):
    features = batch.column("features").to_pyarray()
    labels   = batch.column("label").to_pyarray()
    scores   = model.predict(features)
    accumulator.update(scores, labels)

metrics = accumulator.compute()
```

---

## 5. Batch Prediction

Write predictions in partitioned chunks — never accumulate all predictions in memory before writing.

```python
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def run_batch_prediction(
    model,
    input_path: str,
    output_path: str,
    feature_cols: list[str],
    run_date: str,
    batch_size: int = 50_000,
    logger = None,
) -> None:
    """
    Run batch prediction in streaming fashion — one batch at a time.

    Writes predictions to a Hive-partitioned Parquet dataset.
    Peak memory usage is bounded to one batch regardless of dataset size.

    Parameters
    ----------
    input_path : str
        Root path of the input Hive-partitioned Parquet dataset.
    output_path : str
        Root path for prediction output.
    run_date : str
        Partition value written to the output dataset.
    batch_size : int
        Rows processed per iteration.
    """
    dataset    = ds.dataset(input_path, partitioning="hive", format="parquet")
    writer     = None
    n_written  = 0

    for batch in dataset.to_batches(batch_size=batch_size, columns=feature_cols):
        features = np.stack(
            [batch.column(c).to_pylist() for c in feature_cols], axis=1
        ).astype(np.float32)

        scores = model.predict_proba(features)[:, 1].astype(np.float32)

        out_batch = pa.table({
            "entity_id": batch.column("entity_id"),
            "score":     pa.array(scores, type=pa.float32()),
            "run_date":  pa.array([run_date] * len(scores), type=pa.string()),
        })

        # Write incrementally — no accumulation
        if writer is None:
            writer = pq.ParquetWriter(
                f"{output_path}/date={run_date}/predictions.parquet",
                out_batch.schema,
                compression="snappy",
            )
        writer.write_table(out_batch)
        n_written += len(scores)

    if writer:
        writer.close()

    if logger:
        logger.info("batch_prediction COMPLETE. n_written=%d run_date=%s", n_written, run_date)
```

---

## 6. Memory Monitoring

Track peak memory usage during development to catch regressions before they reach production:

```python
import tracemalloc
import os

def get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# Context manager for tracking peak allocation in a block
class MemoryTracker:
    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mb = peak / 1024 / 1024

    def report(self, label: str) -> None:
        print(f"{label}: peak allocation {self.peak_mb:.1f} MB")


# Usage
with MemoryTracker() as tracker:
    run_batch_prediction(model, input_path, output_path, feature_cols, run_date)
tracker.report("batch_prediction")
```

---

## Rules

- Never load the full dataset into memory — use batch iteration at every stage
- Set `batch_size` based on available memory, not convenience — profile peak usage
- Use Parquet for storage and cross-platform exchange; use Feather for in-process caching and IPC
- Target 128–512 MB per Parquet file — avoid both small files and single large files
- Set `row_group_size=50_000–100_000` in Parquet writes — enables fine-grained column pruning
- Always specify `columns=` when reading Parquet — never read all columns if only a subset is needed
- Accumulate additive statistics across batches for evaluation — never collect all predictions in memory
- Write predictions incrementally using `ParquetWriter` — never accumulate before writing
- Monitor peak memory during development — a batch size that works on 1M rows may OOM on 10M
