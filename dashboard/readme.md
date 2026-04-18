# Dashboard & Observability

Effective dashboards in ML systems go beyond uptime checks — they surface data quality issues, model degradation, cost anomalies, and pipeline failures before they impact production.

---

## 1. Measure at the Right Level — Not from Raw

Never aggregate from raw event-level data in a dashboard query. Raw tables grow unboundedly and querying them directly causes slow dashboards, high compute costs, and inconsistent results under load.

**Pattern:** pre-aggregate at ingestion or on a schedule, then query the aggregated layer.

```
raw events → daily aggregate table → dashboard query
```

```sql
-- ❌ Bad — scans the entire raw table on every dashboard refresh
SELECT DATE(event_time), COUNT(*) AS requests, AVG(latency_ms) AS avg_latency
FROM raw_prediction_logs
WHERE event_time >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY 1

-- ✅ Good — query the pre-aggregated table
SELECT date, request_count, avg_latency_ms
FROM prediction_metrics_daily
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
```

### Rules

- Build a dedicated metrics layer (daily/hourly aggregates) as the source of truth for dashboards
- Dashboard queries should only `SELECT` from aggregate tables — no `GROUP BY` on raw columns
- Refresh aggregates on a schedule (e.g. every hour) rather than computing on every dashboard load
- Partition aggregate tables by date so queries only scan the relevant window

---

## 2. Delta Load — Only Process New Data

Reprocessing historical data on every pipeline run is wasteful and slow. Use delta (incremental) loading to process only records that have arrived since the last run.

```
last_processed_timestamp → fetch new records only → append to aggregate table → update watermark
```

```sql
-- Only fetch rows newer than the last watermark
INSERT INTO prediction_metrics_daily
SELECT
    DATE(event_time)           AS date,
    COUNT(*)                   AS request_count,
    COUNTIF(error IS NOT NULL) AS error_count
FROM raw_prediction_logs
WHERE event_time > (
    SELECT MAX(last_processed_at)
    FROM pipeline_watermarks
    WHERE pipeline = 'prediction_metrics'
)
GROUP BY DATE(event_time);

-- Update the watermark after a successful run
UPDATE pipeline_watermarks
SET last_processed_at = CURRENT_TIMESTAMP()
WHERE pipeline = 'prediction_metrics';
```

### Rules

- Store a watermark (last processed timestamp or ID) per pipeline in a control table
- Always use `>` not `>=` on the watermark to avoid double-counting boundary records
- Backfill by resetting the watermark — no code changes needed
- For append-only tables, partition by ingestion date and only scan the latest partition
- Test delta logic explicitly — off-by-one errors on the watermark boundary are a common bug

---

## 3. Multi-Level Aggregation — Avoid Recomputing Shared Elements

When multiple metrics share intermediate calculations, compute those elements once at the lowest granularity and derive all higher-level metrics from them. This avoids redundant computation and ensures consistency across metrics.

**Key insight:** `mean` is not directly additive, but `sum` and `count` are. Store the additive elements at daily grain — then any statistical test or metric can be derived from them without re-scanning raw data.

```
daily (n, sum, sum_sq, count_positive) → t-test, z-test, PSI, weekly rollup, monthly rollup ...
```

### Additive elements to store

| Element | Formula | Derives |
|---------|---------|---------|
| `n` | `COUNT(*)` | sample size for all tests |
| `sum` | `SUM(x)` | mean: `sum / n` |
| `sum_sq` | `SUM(x * x)` | variance: `sum_sq/n - (sum/n)²` → stddev, t-test |
| `count_positive` | `COUNTIF(x = 1)` | proportion: `count_positive / n` → z-test, PSI |

```sql
-- Store additive elements once at daily grain
CREATE TABLE prediction_score_daily AS
SELECT
    DATE(session_date)                          AS date,
    product_id,
    channel,
    COUNT(*)                                    AS n,
    SUM(predicted_score)                        AS score_sum,
    SUM(predicted_score * predicted_score)      AS score_sum_sq,
    COUNTIF(accepted = 1)                       AS acceptance_count
FROM raw_regrade_sessions
GROUP BY DATE(session_date), product_id, channel;
```

### Example 1 — Two-sample t-test (mean predicted score: this week vs last week)

Detects whether the model's predicted score distribution has shifted between two periods — a signal of model drift or upstream data change.

```sql
-- t-statistic = (mean_a - mean_b) / sqrt(var_a/n_a + var_b/n_b)
WITH periods AS (
    SELECT
        CASE
            WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) THEN 'current'
            ELSE 'previous'
        END                                         AS period,
        SUM(n)                                      AS n,
        SUM(score_sum)                              AS score_sum,
        SUM(score_sum_sq)                           AS score_sum_sq
    FROM prediction_score_daily
    WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
    GROUP BY 1
),
stats AS (
    SELECT
        period,
        n,
        score_sum / n                               AS mean,
        (score_sum_sq / n) - POW(score_sum / n, 2) AS variance  -- E[x²] - E[x]²
    FROM periods
)
SELECT
    MAX(IF(period = 'current',  mean, NULL))        AS mean_current,
    MAX(IF(period = 'previous', mean, NULL))        AS mean_previous,
    (MAX(IF(period = 'current',  mean, NULL)) - MAX(IF(period = 'previous', mean, NULL)))
    / SQRT(
        MAX(IF(period = 'current',  variance, NULL)) / MAX(IF(period = 'current',  n, NULL))
      + MAX(IF(period = 'previous', variance, NULL)) / MAX(IF(period = 'previous', n, NULL))
    )                                               AS t_statistic
FROM stats;
-- |t| > 1.96 → statistically significant shift at 95% confidence
```

### Example 2 — Z-test for proportions (acceptance rate: this week vs last week)

Detects whether the acceptance rate has changed significantly — the primary business KPI for the pricing model.

```sql
-- z = (p1 - p2) / sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
WITH periods AS (
    SELECT
        CASE
            WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) THEN 'current'
            ELSE 'previous'
        END                                         AS period,
        SUM(n)                                      AS n,
        SUM(acceptance_count)                       AS acceptance_count
    FROM prediction_score_daily
    WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
    GROUP BY 1
),
stats AS (
    SELECT period, n, acceptance_count / n AS p
    FROM periods
),
pooled AS (
    SELECT SUM(acceptance_count) / SUM(n) AS p_pool
    FROM periods
)
SELECT
    MAX(IF(period = 'current',  p, NULL))           AS acceptance_rate_current,
    MAX(IF(period = 'previous', p, NULL))           AS acceptance_rate_previous,
    (MAX(IF(period = 'current',  p, NULL)) - MAX(IF(period = 'previous', p, NULL)))
    / SQRT(
        p_pool * (1 - p_pool)
        * (  1.0 / MAX(IF(period = 'current',  n, NULL))
           + 1.0 / MAX(IF(period = 'previous', n, NULL)))
    )                                               AS z_statistic
FROM stats, pooled;
-- |z| > 1.96 → statistically significant change in acceptance rate at 95% confidence
```

### Example 3 — PSI (Population Stability Index) for score drift

PSI measures how much the predicted score distribution has shifted between training and production. It reuses the same `n` and bucket counts already stored — no raw scan needed.

```sql
-- PSI = SUM((actual% - expected%) * ln(actual% / expected%)) per bucket
-- PSI < 0.1: stable | 0.1–0.2: minor shift | > 0.2: significant drift
WITH bucket_counts AS (
    SELECT
        score_bucket,
        SUM(IF(period = 'production', n, 0))        AS n_actual,
        SUM(IF(period = 'training',   n, 0))        AS n_expected
    FROM prediction_score_daily_bucketed
    GROUP BY score_bucket
),
bucket_pct AS (
    SELECT
        score_bucket,
        n_actual   / SUM(n_actual)   OVER ()        AS actual_pct,
        n_expected / SUM(n_expected) OVER ()        AS expected_pct
    FROM bucket_counts
)
SELECT
    SUM(
        (actual_pct - expected_pct)
        * LN(actual_pct / NULLIF(expected_pct, 0))
    )                                               AS psi
FROM bucket_pct;
```

### Rules

- Store `n`, `sum`, `sum_sq`, and `count_positive` at the lowest grain — these four elements power t-tests, z-tests, PSI, and any mean/variance metric
- `mean` and `variance` are derived at query time, never stored — storing them loses the ability to re-aggregate across filters
- One daily aggregate table serves all statistical tests across any dimension (product, channel, date range) without touching raw data
- Apply the same pattern to feature distributions — store `n`, `sum`, `sum_sq` per feature per day to detect input drift with the same queries

---

## 4. Centralise Pipeline Logs — All-in-One Health Dashboard

Scatter logs across pipelines and you lose the ability to correlate failures. Centralise all pipeline logs (errors, warnings, row counts, durations) into a single queryable store so one dashboard can show the health of the entire system — training, batch prediction, and feature pipelines together.

```
training pipeline     → log shipper → centralised log store → health dashboard
batch prediction      → log shipper ↗
feature engineering   → log shipper ↗
```

Every pipeline should emit structured JSON logs with consistent fields. The same format across all pipelines is what makes a single dashboard possible.

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)

# Step completion
logger.info("Step 1: data_load COMPLETE", extra={
    "pipeline": "training",
    "environment": "prod",
    "run_id": run_id,
    "step": "data_load",
    "rows_loaded": rows_loaded,
    "duration_seconds": duration,
})

# Business rule warning
logger.warning("WARNING: low sample count for product segment", extra={
    "pipeline": "training",
    "environment": "prod",
    "run_id": run_id,
    "step": "data_validation",
    "product_id": product_id,
    "sample_count": sample_count,
    "threshold": min_sample_threshold,
})

# Pipeline error
logger.error("ERROR: feature store connection failed", extra={
    "pipeline": "batch_prediction",
    "environment": "prod",
    "run_id": run_id,
    "step": "feature_fetch",
    "error_type": "ConnectionTimeout",
    "retry_count": retry_count,
})

# Run summary on completion
logger.info("pipeline_run_complete", extra={
    "pipeline": "batch_prediction",
    "run_id": run_id,
    "run_date": run_date,
    "rows_processed": rows_processed,
    "rows_inserted": rows_inserted,
    "duration_seconds": duration,
    "status": "success",
    "watermark_from": watermark_from,
    "watermark_to": watermark_to,
})
```

### Standard log fields

| Field | Purpose |
|-------|---------|
| `pipeline` | `training`, `batch_prediction`, `feature_engineering` |
| `environment` | `exp` / `stag` / `prod` |
| `level` | `ERROR`, `WARNING`, `INFO` |
| `run_id` | Correlate all log lines belonging to the same pipeline run |
| `step` | Pipeline step — `data_load`, `feature_fetch`, `predict`, `postprocess` |
| `rows_processed` | Volume signal — drops indicate upstream data issues |
| `duration_seconds` | Step duration — spikes indicate performance regression |
| `error_type` | Machine-readable error class — enables grouping in dashboards |
| `timestamp` | ISO 8601, UTC |

### Dashboard panels

| Panel | What it surfaces |
|-------|-----------------|
| Pipeline run status by date | Failed runs at a glance across all pipelines |
| Error count by `pipeline` + `step` | Which step is failing most frequently |
| Warning trend by `pipeline` | Early signal before errors appear |
| Rows processed per run | Volume drop → upstream data issue |
| Step duration trend | Spike → query regression or data volume surge |
| Watermark lag | Current time minus latest watermark > SLA threshold |
| Error type breakdown | `ConnectionTimeout` vs `ValidationError` vs `OOMError` |

### Tooling

| Stack | Log store | Dashboard |
|-------|-----------|-----------|
| AWS | CloudWatch Logs | CloudWatch Dashboards / Grafana |
| GCP | Cloud Logging | Cloud Monitoring / Grafana |
| Self-hosted | Elasticsearch | Kibana / Grafana |
| Any | Loki | Grafana |

### Rules

- Use the same log format across every pipeline — inconsistent fields break cross-pipeline queries
- Always include `run_id` in every log line — it is the primary key for correlating a full pipeline run
- Always log `rows_processed` at each step — silent row count drops are harder to detect than errors
- Always emit a run summary log on completion with watermark bounds and row counts
- Set log retention policies: production logs 90 days, experiment logs 14 days
- Alert on error rate thresholds, not just individual errors — a single flaky run is noise; a trend is a problem

---

## 5. Monitor Cloud Cost by Label

Cloud costs in ML projects spike unpredictably — training runs, large batch jobs, and autoscaling events can generate significant charges. Label every resource so costs can be attributed, monitored, and acted on.

### Labelling strategy

| Label key | Example values | Purpose |
|-----------|---------------|---------|
| `project` | `nba-bb-regrade-r3` | Attribute cost to a project |
| `environment` | `exp`, `stag`, `prod` | Separate experiment spend from production |
| `component` | `training`, `serving`, `feature-store` | Break down cost by system component |
| `team` | `ml-pricing` | Charge back to the owning team |
| `managed-by` | `terraform` | Identify IaC-managed vs manually created resources |

Apply labels via shared locals in Terraform so every resource inherits them consistently:

```hcl
locals {
  common_labels = {
    project     = "nba-bb-regrade-r3"
    environment = var.environment
    component   = "serving"
    team        = "ml-pricing"
    managed-by  = "terraform"
  }
}

resource "aws_ecs_service" "prediction_api" {
  name = "prediction-api-${var.environment}"
  tags = local.common_labels
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id = "prediction-cache-${var.environment}"
  tags       = local.common_labels
}
```

### Dashboard panels

| Panel | What to watch |
|-------|--------------|
| Daily spend by `environment` | Catch experiment cost creep |
| Daily spend by `component` | Identify which component is driving cost |
| Training cost per run | Flag expensive or runaway training jobs |
| Untagged resource spend | Resources that escaped IaC control |
| Month-to-date vs budget | Early warning before budget is exhausted |

### Rules

- Set a budget alert per project label — trigger at 80% and 100% of monthly budget
- Monitor `exp` and `prod` environments separately — experiment costs should never approach production costs
- Flag training jobs that exceed a cost threshold (e.g. > $50 per run) — often caused by misconfigured instance types or runaway HPT
- Review untagged resources weekly — untagged spend is invisible spend
- Use cost anomaly detection (AWS Cost Anomaly Detection, GCP Budget Alerts) to catch unexpected spikes automatically

---

## 6. Model Performance Monitoring

Track model behaviour in production — not just pipeline health. A pipeline that runs successfully with a degraded model is a silent failure.

### Metrics to monitor

| Metric | What it detects |
|--------|----------------|
| Predicted score distribution (PSI) | Model output drift from training baseline |
| Feature distribution shift | Input data changing — upstream pipeline issues |
| Acceptance rate week-over-week (z-test) | Business KPI degradation |
| Null / missing feature rate | Feature store failures or schema changes |
| Model version rollout | Performance change tied to a specific deployment |

All of these can be derived from the daily aggregate table built in section 3 — no additional raw scans needed.

### Rules

- Alert when acceptance rate drops more than a defined threshold week-over-week — this is typically the first business signal of model degradation
- Alert when PSI > 0.2 on any key feature or the predicted score — trigger retraining investigation
- Track metrics by model version so degradation can be tied to a specific deployment
- Monitor null rates per feature daily — a sudden spike indicates a feature store schema change or upstream pipeline failure
