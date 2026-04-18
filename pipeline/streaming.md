# Streaming Pipelines

Streaming pipelines process events continuously as they arrive — enabling real-time feature computation, online model inference, and immediate alerting. Unlike batch pipelines that run on a schedule, streaming pipelines run indefinitely.

---

## When to Use Streaming

```
Predictions needed within seconds of an event    → Streaming inference
Features must reflect the last N minutes         → Streaming feature computation
Fraud / anomaly detection on live transactions   → Streaming ML
Batch is fast enough (minutes to hours)          → Batch pipeline (simpler)
```

---

## Streaming Tool Comparison

| Tool | Type | Best for | Managed options |
|------|------|---------|----------------|
| **Apache Kafka** | Message broker | High-throughput event streaming, durable log | Confluent Cloud, AWS MSK |
| **AWS Kinesis** | Message broker | AWS-native streaming, tight AWS integration | Fully managed |
| **GCP Pub/Sub** | Message broker | GCP-native, serverless, global | Fully managed |
| **Apache Flink** | Stream processor | Stateful streaming, exactly-once, low latency | AWS Kinesis Analytics, Ververica |
| **Apache Kafka Streams** | Stream processor | Lightweight processing co-located with Kafka | Via Confluent |
| **Apache Beam** | Unified batch+stream | Write once, run on Dataflow/Flink/Spark | GCP Dataflow |

---

## Kafka

Kafka is a distributed, durable, high-throughput event log. Producers write events to topics; consumers read from topics at their own pace.

### Core concepts

| Concept | Description |
|---------|-------------|
| **Topic** | A named, ordered, durable log of events |
| **Partition** | A topic is split into partitions for parallelism — each partition is an ordered sequence |
| **Offset** | The position of a message within a partition — consumers track their own offset |
| **Consumer group** | Multiple consumers sharing the work of reading a topic — each partition assigned to one consumer |
| **Retention** | How long messages are kept — configurable per topic (e.g. 7 days) |

### Producer

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=["kafka:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    acks="all",              # wait for all replicas to acknowledge
    retries=3,
    compression_type="snappy",
)

# Publish a prediction request event
producer.send(
    topic="prediction-requests",
    key=b"customer_001",     # key determines partition assignment
    value={
        "customer_id": "customer_001",
        "product_id":  "product_017",
        "timestamp":   "2025-01-15T10:30:00Z",
        "features":    [0.5, 0.3, 0.8, 0.1],
    },
)
producer.flush()
```

### Consumer

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "prediction-requests",
    bootstrap_servers=["kafka:9092"],
    group_id="ml-scoring-service",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",      # start from latest on first run
    enable_auto_commit=False,        # manual commit for exactly-once semantics
    max_poll_records=100,
)

for message in consumer:
    event = message.value
    try:
        score = model.predict(event["features"])
        publish_result(event["customer_id"], score)
        consumer.commit()            # commit only after successful processing
    except Exception as e:
        logger.error("ERROR: scoring failed. customer_id=%s error=%s",
                     event["customer_id"], e)
        # do not commit — message will be reprocessed
```

---

## Real-Time ML Inference with Kafka

A common pattern: events flow through Kafka, a consumer scores them with a model, and results are published to an output topic.

```
Input topic (raw events)
    ↓
ML scoring consumer (fetch features → predict → publish)
    ↓
Output topic (scored events)
    ↓
Downstream consumers (dashboard, alert, feature store update)
```

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np

class StreamingScorer:
    """
    Consume prediction request events, score with the model,
    and publish results to the output topic.

    Notes
    -----
    Features are fetched from the online feature store per event.
    Model is loaded once at startup and reused across all messages.
    """

    def __init__(self, model, feature_store, input_topic: str, output_topic: str) -> None:
        self.model         = model
        self.feature_store = feature_store
        self.consumer      = KafkaConsumer(
            input_topic,
            bootstrap_servers=["kafka:9092"],
            group_id="ml-scoring",
            value_deserializer=lambda v: json.loads(v.decode()),
            enable_auto_commit=False,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=["kafka:9092"],
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        self.output_topic = output_topic

    def run(self) -> None:
        for message in self.consumer:
            event = message.value
            try:
                features = self.feature_store.get(event["customer_id"])
                score    = float(self.model.predict_proba(
                    np.array(features, dtype=np.float32).reshape(1, -1)
                )[0, 1])

                self.producer.send(self.output_topic, value={
                    "customer_id": event["customer_id"],
                    "score":       score,
                    "timestamp":   event["timestamp"],
                })
                self.consumer.commit()

            except Exception as e:
                logger.error("ERROR500: scoring failed. customer_id=%s error=%s",
                             event["customer_id"], e)
```

---

## Streaming Feature Computation

Compute features in real time from an event stream and write them to the online feature store.

```python
from kafka import KafkaConsumer
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

class RollingFeatureComputer:
    """
    Maintain a rolling 30-minute window of events per customer
    and compute features incrementally.
    """

    def __init__(self, window_minutes: int = 30) -> None:
        self.window    = timedelta(minutes=window_minutes)
        self.events    = defaultdict(deque)   # customer_id → deque of (timestamp, value)

    def add_event(self, customer_id: str, value: float, timestamp: datetime) -> None:
        self.events[customer_id].append((timestamp, value))
        self._evict_old(customer_id, timestamp)

    def _evict_old(self, customer_id: str, now: datetime) -> None:
        window = self.events[customer_id]
        while window and (now - window[0][0]) > self.window:
            window.popleft()

    def get_features(self, customer_id: str) -> dict:
        values = [v for _, v in self.events[customer_id]]
        if not values:
            return {"event_count_30m": 0, "avg_value_30m": 0.0}
        return {
            "event_count_30m": len(values),
            "avg_value_30m":   sum(values) / len(values),
        }
```

---

## AWS Kinesis

Kinesis is AWS's managed streaming service. The API is simpler than Kafka but with lower throughput limits per shard.

```python
import boto3
import json
import base64

kinesis = boto3.client("kinesis", region_name="eu-west-1")

# Produce
kinesis.put_record(
    StreamName="prediction-requests",
    Data=json.dumps({"customer_id": "cust_001", "features": [0.5, 0.3]}),
    PartitionKey="cust_001",
)

# Consume via Lambda trigger (serverless)
def lambda_handler(event, context):
    for record in event["Records"]:
        payload = json.loads(base64.b64decode(record["kinesis"]["data"]))
        score   = model.predict(payload["features"])
        # process score
```

---

## GCP Pub/Sub

Pub/Sub is GCP's managed messaging service — serverless, globally distributed, and tightly integrated with Dataflow and BigQuery.

```python
from google.cloud import pubsub_v1
import json

# Publish
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("my-project", "prediction-requests")

publisher.publish(
    topic_path,
    data=json.dumps({"customer_id": "cust_001", "features": [0.5, 0.3]}).encode(),
    customer_id="cust_001",   # message attributes for filtering
)

# Subscribe
subscriber  = pubsub_v1.SubscriberClient()
sub_path    = subscriber.subscription_path("my-project", "scoring-subscription")

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    payload = json.loads(message.data.decode())
    score   = model.predict(payload["features"])
    message.ack()   # acknowledge after successful processing

subscriber.subscribe(sub_path, callback=callback)
```

---

## Delivery Guarantees

| Guarantee | Description | Kafka | Kinesis | Pub/Sub |
|-----------|-------------|-------|---------|---------|
| At-most-once | Message delivered 0 or 1 times — possible loss | ✅ | ✅ | ✅ |
| At-least-once | Message delivered 1+ times — possible duplicates | ✅ | ✅ | ✅ |
| Exactly-once | Message delivered exactly once | ✅ (transactions) | ❌ | ❌ |

For ML scoring, at-least-once with idempotent processing is usually sufficient — scoring the same event twice produces the same result.

---

## Rules

- Make consumers idempotent — the same message processed twice should produce the same result
- Commit offsets only after successful processing — never before
- Set consumer group IDs explicitly — auto-generated IDs lose offset tracking on restart
- Monitor consumer lag — the gap between the latest offset and the consumer's current offset; growing lag means the consumer is falling behind
- Use a dead-letter topic for messages that fail repeatedly — never silently drop failed messages
- Load models once at consumer startup — never reload per message
- Partition by entity ID (customer, product) — ensures all events for the same entity are processed in order by the same consumer
