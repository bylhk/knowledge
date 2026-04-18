# Feature Stores

A feature store is a centralised repository for storing, serving, and sharing ML features. It bridges the gap between the data engineering pipeline (which computes features) and the ML pipeline (which consumes them) — ensuring that training and serving use identical feature values.

---

## The Training-Serving Skew Problem

Without a feature store, features are often computed differently in training and serving:

```
Training:  SQL query on historical data → features computed in batch
Serving:   Python code in the API → features computed on the fly

Result: different logic, different values → model trained on data it never sees in production
```

A feature store solves this by computing features once and serving the same values to both training and serving.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Feature** | A single measurable property of an entity (e.g. `customer_avg_spend_30d`) |
| **Entity** | The object a feature describes (customer, product, session) |
| **Feature view** | A group of related features computed from the same source |
| **Online store** | Low-latency key-value store for serving features at request time (Redis, DynamoDB, Bigtable) |
| **Offline store** | Historical feature values for training (S3/GCS Parquet, BigQuery, Redshift) |
| **Materialisation** | The process of computing features and writing them to the online store |
| **Point-in-time join** | Joining feature values to training labels using the timestamp at which the label was observed — prevents future leakage |

---

## Architecture

```
Raw data (warehouse / lake)
    ↓
Feature pipeline (Spark / SQL / Beam)
    ↓
Offline store (Parquet / BigQuery)  ←── Training pipeline reads here
    ↓ materialise
Online store (Redis / DynamoDB)     ←── Serving pipeline reads here
```

---

## Point-in-Time Join

The most critical feature store operation for training. When building a training dataset, each label must be joined to the feature values that were available at the time the label was observed — not the latest values.

```python
# Without point-in-time join — data leakage
# Label observed on 2025-01-10, but feature value from 2025-01-15 is used
labels_df.join(features_df, on="customer_id")   # wrong — uses latest feature value

# With point-in-time join — correct
# For each label, find the most recent feature value BEFORE the label timestamp
labels_df.join(
    features_df,
    on="customer_id",
    how="left",
    # only use feature values where feature_timestamp <= label_timestamp
)
```

Most feature store frameworks handle this automatically via their training data retrieval API.

---

## Feast (Open Source)

Feast is the most widely used open-source feature store. It supports multiple offline stores (Parquet, BigQuery, Redshift) and online stores (Redis, DynamoDB, SQLite).

### Define features

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define the entity
customer = Entity(
    name="customer_id",
    description="Unique customer identifier",
)

# Define the data source
customer_stats_source = FileSource(
    path="s3://bucket/customer_stats/",
    timestamp_field="event_timestamp",
)

# Define the feature view
customer_stats_fv = FeatureView(
    name="customer_stats",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="avg_spend_30d",    dtype=Float32),
        Field(name="session_count_7d", dtype=Int64),
        Field(name="days_since_last_purchase", dtype=Int64),
    ],
    source=customer_stats_source,
)
```

### Retrieve training data (offline, point-in-time)

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Entity dataframe — one row per label with the timestamp of observation
entity_df = pd.DataFrame({
    "customer_id":       ["cust_001", "cust_002", "cust_003"],
    "event_timestamp":   pd.to_datetime(["2025-01-10", "2025-01-11", "2025-01-12"]),
    "label":             [1, 0, 1],
})

# Point-in-time join — retrieves feature values as of each row's timestamp
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_stats:avg_spend_30d",
        "customer_stats:session_count_7d",
        "customer_stats:days_since_last_purchase",
    ],
).to_df()
```

### Materialise to online store

```python
from datetime import datetime

# Push latest feature values to the online store (Redis)
store.materialize_incremental(end_date=datetime.utcnow())
```

### Retrieve features at serving time (online, low-latency)

```python
# Returns the latest feature values for a customer — typically < 5ms
feature_vector = store.get_online_features(
    features=[
        "customer_stats:avg_spend_30d",
        "customer_stats:session_count_7d",
    ],
    entity_rows=[{"customer_id": "cust_001"}],
).to_dict()
```

---

## Managed Feature Stores

### Vertex AI Feature Store (GCP)

```python
from google.cloud.aiplatform import Featurestore

# Create a feature store
fs = Featurestore.create(
    featurestore_id="customer_features",
    online_store_fixed_node_count=1,
    project="my-project",
    location="europe-west2",
)

# Create entity type and features
entity_type = fs.create_entity_type(entity_type_id="customer")
entity_type.batch_create_features(feature_configs={
    "avg_spend_30d":    {"value_type": "DOUBLE"},
    "session_count_7d": {"value_type": "INT64"},
})

# Read features at serving time
feature_values = entity_type.read(
    entity_ids=["cust_001", "cust_002"],
    feature_ids=["avg_spend_30d", "session_count_7d"],
)
```

### SageMaker Feature Store (AWS)

```python
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

session       = sagemaker.Session()
feature_group = FeatureGroup(name="customer-features", sagemaker_session=session)

# Ingest features
feature_group.ingest(
    data_frame=features_df,
    max_workers=4,
    wait=True,
)

# Retrieve at serving time (online store)
record = feature_group.get_record(
    record_identifier_value_as_string="cust_001"
)
```

---

## Feature Store Comparison

| | Feast | Vertex AI Feature Store | SageMaker Feature Store | Tecton |
|---|-------|------------------------|------------------------|--------|
| Type | Open source | Managed (GCP) | Managed (AWS) | Managed (any cloud) |
| Online store | Redis, DynamoDB, SQLite | Bigtable | DynamoDB | Redis, DynamoDB |
| Offline store | Parquet, BigQuery, Redshift | BigQuery | S3 | Any |
| Point-in-time join | ✅ | ✅ | ✅ | ✅ |
| Streaming features | Via Kafka + custom | ✅ | ✅ | ✅ |
| Cost | Infrastructure only | Per node + storage | Per write + storage | Subscription |

---

## Rules

- Always use point-in-time joins for training data — never join on entity ID alone
- Materialise features on a schedule that matches the freshness requirement — a feature that changes daily should be materialised daily
- Store feature computation logic in the feature store definition, not in the serving code — this is what prevents training-serving skew
- Monitor feature null rates and distribution drift — a feature that goes null in production is a silent failure
- Version feature views — a breaking schema change should create a new feature view, not modify the existing one
- Keep online store TTL aligned with feature staleness — a feature with a 30-day window should not be served after 31 days without refresh
