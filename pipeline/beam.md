# Apache Beam

Apache Beam is a unified programming model for both batch and streaming data pipelines. The same pipeline code runs on multiple execution engines (runners) — Dataflow, Spark, Flink, or locally — without modification.

---

## When to Use Beam

```
Batch only, large scale          → Spark or SQL (simpler)
Streaming only, low latency      → Kafka Streams or Flink (lower overhead)
Unified batch + streaming        → Beam (write once, run both modes)
GCP-native managed streaming     → Beam on Dataflow
Exactly-once semantics required  → Beam on Dataflow
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Pipeline** | The top-level object representing the entire data flow |
| **PCollection** | A distributed, immutable collection of elements — the data flowing through the pipeline |
| **PTransform** | A transformation applied to a PCollection — `Map`, `Filter`, `GroupByKey`, `Flatten` |
| **Runner** | The execution engine — `DirectRunner` (local), `DataflowRunner` (GCP), `SparkRunner`, `FlinkRunner` |
| **Window** | Groups elements by time for streaming aggregations |
| **Trigger** | Controls when windowed results are emitted |
| **Watermark** | Tracks event-time progress — determines when a window is considered complete |

---

## Basic Pipeline

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

options = PipelineOptions(
    runner="DataflowRunner",
    project="my-project",
    region="europe-west2",
    temp_location="gs://bucket/tmp/",
    job_name="feature-pipeline",
)

with beam.Pipeline(options=options) as pipeline:
    (
        pipeline
        | "ReadParquet"    >> beam.io.ReadFromParquet("gs://bucket/sessions/*.parquet")
        | "FilterRecent"   >> beam.Filter(lambda row: row["date"] >= "2025-01-01")
        | "ComputeFeature" >> beam.Map(compute_discount)
        | "WriteParquet"   >> beam.io.WriteToParquet(
            "gs://bucket/features/output",
            schema=output_schema,
        )
    )


def compute_discount(row: dict) -> dict:
    avg   = row.get("avg_recommended_price") or 0
    price = row.get("predicted_price") or 0
    return {
        **row,
        "discount": (avg - price) / avg if avg else 0.0,
    }
```

---

## Batch Processing Performance — DoFn and Batching

The single biggest performance lever in Beam batch pipelines is how you structure the processing unit. `beam.Map` calls a Python function once per element — one dict at a time. For numerical computation, this is 100x+ slower than processing a batch of elements as a NumPy array in a single call.

### The problem with element-wise Map

```python
# ❌ Bad — Python function called once per row
# For 10M rows: 10M Python function calls, 10M dict lookups, no vectorisation
def compute_features(row: dict) -> dict:
    discount = (row["avg_price"] - row["price"]) / row["avg_price"]
    score    = row["tenure"] * 0.3 + row["spend"] * 0.7
    return {**row, "discount": discount, "score": score}

pipeline | beam.Map(compute_features)
```

### BatchElements + DoFn — process as NumPy batches

`beam.BatchElements` groups elements into variable-size batches. A `DoFn` with `process_batch` receives the entire batch as a list and can convert it to a NumPy array for vectorised computation.

```python
import apache_beam as beam
import numpy as np
from apache_beam.transforms.util import BatchElements

class FeatureDoFn(beam.DoFn):
    """
    Compute features for a batch of rows using NumPy vectorisation.

    Receives a list of dicts from BatchElements, converts to NumPy arrays,
    applies vectorised operations, and yields one output dict per row.

    Notes
    -----
    Processing a batch of 1000 rows as NumPy arrays is 100–500x faster
    than calling a Python function once per row.
    """

    def process(self, batch: list[dict]):
        # Restructure: list of dicts → columnar NumPy arrays
        avg_price = np.array([r["avg_price"] for r in batch], dtype=np.float32)
        price     = np.array([r["price"]     for r in batch], dtype=np.float32)
        tenure    = np.array([r["tenure"]    for r in batch], dtype=np.float32)
        spend     = np.array([r["spend"]     for r in batch], dtype=np.float32)

        # Vectorised computation — one C-level operation over the whole batch
        discount = (avg_price - price) / np.where(avg_price > 0, avg_price, 1.0)
        score    = tenure * 0.3 + spend * 0.7

        # Yield one output per row
        for i, row in enumerate(batch):
            yield {
                **row,
                "discount": float(discount[i]),
                "score":    float(score[i]),
            }


with beam.Pipeline(options=options) as pipeline:
    (
        pipeline
        | "Read"          >> beam.io.ReadFromParquet(
            "gs://bucket/sessions/*.parquet",
            columns=["customer_id", "avg_price", "price", "tenure", "spend"],
        )
        | "Batch"         >> BatchElements(min_batch_size=500, max_batch_size=5000)
        | "ComputeFeatures" >> beam.ParDo(FeatureDoFn())
        | "Write"         >> beam.io.WriteToParquet(
            "gs://bucket/features/output",
            schema=output_schema,
        )
    )
```

### Restructure input data — columnar layout

The key insight is that `beam.Map` delivers data as a **row-oriented** stream (one dict per element). NumPy is **column-oriented** (one array per field). The conversion from row-oriented to column-oriented inside the DoFn is what unlocks vectorisation.

```
Row-oriented (Beam default):     [{"a": 1, "b": 2}, {"a": 3, "b": 4}, ...]
                                          ↓ restructure inside DoFn
Column-oriented (NumPy):         a = np.array([1, 3, ...])   b = np.array([2, 4, ...])
                                          ↓ vectorised operation
Result (back to rows):           [{"a": 1, "b": 2, "c": 3}, ...]
```

### Batch model inference inside a DoFn

The same pattern applies to model inference — load the model once in `setup()`, batch inputs into a NumPy array, run a single `predict_proba` call:

```python
class BatchScoringDoFn(beam.DoFn):
    """
    Score a batch of rows with a pre-loaded model.

    setup() is called once per worker — model loaded from GCS into memory.
    process() receives a batch and runs a single vectorised predict call.
    """

    def __init__(self, model_path: str, feature_cols: list[str]) -> None:
        self.model_path  = model_path
        self.feature_cols = feature_cols
        self._model      = None

    def setup(self) -> None:
        """Called once per worker process — load model into memory."""
        import joblib
        from apache_beam.io.gcp.gcsio import GcsIO
        import io

        gcs    = GcsIO()
        buffer = io.BytesIO(gcs.open(self.model_path).read())
        self._model = joblib.load(buffer)

    def process(self, batch: list[dict]):
        # Restructure rows → 2-D NumPy array (n_samples, n_features)
        features = np.array(
            [[row[col] for col in self.feature_cols] for row in batch],
            dtype=np.float32,
        )

        # Single model call for the entire batch — not one call per row
        scores = self._model.predict_proba(features)[:, 1]

        for i, row in enumerate(batch):
            yield {**row, "score": float(scores[i])}


FEATURE_COLS = ["avg_price", "tenure", "spend", "discount"]

with beam.Pipeline(options=options) as pipeline:
    (
        pipeline
        | "Read"   >> beam.io.ReadFromParquet("gs://bucket/sessions/*.parquet")
        | "Batch"  >> BatchElements(min_batch_size=1000, max_batch_size=10000)
        | "Score"  >> beam.ParDo(BatchScoringDoFn(
            model_path="gs://bucket/models/churn_model.pkl",
            feature_cols=FEATURE_COLS,
        ))
        | "Write"  >> beam.io.WriteToParquet(
            "gs://bucket/predictions/output",
            schema=output_schema,
        )
    )
```

### Performance comparison

| Approach | 10M rows | Relative |
|----------|---------|----------|
| `beam.Map` (one dict per call) | ~600s | 1x baseline |
| `BatchElements` + list comprehension | ~120s | ~5x |
| `BatchElements` + NumPy vectorisation | ~6s | ~100x |
| `BatchElements` + NumPy + float32 | ~3s | ~200x |

*Times are illustrative — actual speedup depends on operation complexity and hardware.*

### Tuning batch size

```python
# BatchElements auto-tunes batch size based on wall-clock time per batch
# min/max bounds prevent extreme values
BatchElements(
    min_batch_size=100,    # never smaller — avoids Python overhead dominating
    max_batch_size=10_000, # never larger — avoids OOM on large feature vectors
    target_batch_overhead=0.05,  # aim for 5% overhead from batching logic
    target_batch_duration_secs=1.0,  # aim for 1-second batches
)
```

---

## Batch Feature Pipeline

```python
import apache_beam as beam
from apache_beam import GroupByKey, CombinePerKey
from apache_beam.transforms.combiners import MeanCombineFn
import apache_beam.transforms.combiners as combine

def extract_key_value(row: dict) -> tuple[str, float]:
    return row["customer_id"], row["predicted_score"]

def format_output(element: tuple) -> dict:
    customer_id, scores = element
    return {
        "customer_id":  customer_id,
        "mean_score":   sum(scores) / len(scores),
        "session_count": len(scores),
    }

with beam.Pipeline(options=options) as pipeline:
    (
        pipeline
        | "Read"          >> beam.io.ReadFromParquet("gs://bucket/sessions/*.parquet")
        | "ExtractKV"     >> beam.Map(extract_key_value)
        | "GroupByCustomer" >> beam.GroupByKey()
        | "Aggregate"     >> beam.Map(format_output)
        | "Write"         >> beam.io.WriteToParquet(
            "gs://bucket/customer_features/output",
            schema=output_schema,
        )
    )
```

---

## Streaming Pipeline

Beam handles streaming by reading from a message queue (Pub/Sub, Kafka) and applying windowed aggregations.

```python
import apache_beam as beam
from apache_beam.transforms.window import FixedWindows, SlidingWindows
from apache_beam.transforms.trigger import AfterWatermark, AfterProcessingTime, AccumulationMode
import json

def parse_event(message: bytes) -> dict:
    return json.loads(message.decode("utf-8"))

def extract_score(event: dict) -> tuple[str, float]:
    return event["product_id"], event["predicted_score"]

with beam.Pipeline(options=streaming_options) as pipeline:
    events = (
        pipeline
        | "ReadPubSub"  >> beam.io.ReadFromPubSub(topic="projects/my-project/topics/predictions")
        | "Parse"       >> beam.Map(parse_event)
        | "AddTimestamp" >> beam.Map(
            lambda e: beam.window.TimestampedValue(e, e["event_timestamp"])
        )
    )

    # 1-hour fixed windows with watermark trigger
    windowed = (
        events
        | "Window" >> beam.WindowInto(
            FixedWindows(3600),   # 1-hour windows
            trigger=AfterWatermark(late=AfterProcessingTime(300)),
            accumulation_mode=AccumulationMode.DISCARDING,
            allowed_lateness=600,  # accept late data up to 10 minutes
        )
        | "ExtractKV"   >> beam.Map(extract_score)
        | "GroupByKey"  >> beam.GroupByKey()
        | "Aggregate"   >> beam.Map(lambda kv: {
            "product_id":    kv[0],
            "mean_score":    sum(kv[1]) / len(kv[1]),
            "window_count":  len(kv[1]),
        })
        | "WriteBigQuery" >> beam.io.WriteToBigQuery(
            "my-project:dataset.hourly_scores",
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
        )
    )
```

---

## Runners

| Runner | Where it runs | Best for |
|--------|--------------|---------|
| `DirectRunner` | Local machine | Development and testing |
| `DataflowRunner` | GCP Dataflow | Production — managed, autoscaling, exactly-once |
| `SparkRunner` | Spark cluster | When Spark infrastructure already exists |
| `FlinkRunner` | Flink cluster | Low-latency streaming with stateful processing |

### Switching runners

The pipeline code does not change — only the options:

```python
# Local testing
options = PipelineOptions(runner="DirectRunner")

# Production on Dataflow
options = PipelineOptions(
    runner="DataflowRunner",
    project="my-project",
    region="europe-west2",
    temp_location="gs://bucket/tmp/",
    max_num_workers=20,
    machine_type="n1-standard-4",
)
```

---

## Windowing Strategies

| Window type | Description | Use case |
|-------------|-------------|---------|
| `FixedWindows(size)` | Non-overlapping windows of fixed duration | Hourly/daily aggregations |
| `SlidingWindows(size, period)` | Overlapping windows — each element in multiple windows | Rolling averages |
| `SessionWindows(gap)` | Variable-size windows separated by inactivity gaps | User session analysis |
| `GlobalWindow` | All elements in one window (default) | Batch processing |

---

## Rules

- Use `DirectRunner` for local development — it runs the same code without a cluster
- Always add timestamps to streaming events before windowing — Beam uses event time, not processing time
- Set `allowed_lateness` for streaming windows — late data is common in production
- Prefer `CombinePerKey` over `GroupByKey` + manual aggregation — Combiner runs partial aggregation before the shuffle, reducing data movement
- Use `beam.io.ReadFromParquet` with column projection — only read the columns you need
- Test pipeline logic with `DirectRunner` before submitting to Dataflow — Dataflow jobs are slow to debug
