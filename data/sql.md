# High-Performance SQL

SQL is the fastest tool for large-scale data processing — but only when written correctly. Poor SQL patterns can turn a 1-second query into a 10-minute full table scan. These tips apply to any warehouse (BigQuery, Redshift, Snowflake, Spark SQL).

---

## Schema Design

### 1. Never Trust Default Column Order

Never reference columns by position (`SELECT *`, `ORDER BY 1`, `INSERT INTO t VALUES (...)`) — column order in a table definition can change silently and break queries without any error.

```sql
-- ❌ Bad — breaks silently if columns are reordered
INSERT INTO prediction_output VALUES (customer_id, product_id, score, 0.95)

SELECT * FROM prediction_output ORDER BY 3 DESC

-- ✅ Good — explicit column names, order-independent
INSERT INTO prediction_output (customer_id, product_id, score, confidence)
VALUES (customer_id, product_id, score, 0.95)

SELECT customer_id, product_id, score FROM prediction_output ORDER BY score DESC
```

Always name every column explicitly in `INSERT`, `SELECT`, and `CREATE TABLE AS SELECT` statements.

---

### 2. Always Partition Tables

Every large table should be partitioned by a date or timestamp column. Without partitioning, every query scans the full table regardless of the `WHERE` clause.

```sql
-- BigQuery — partition by date, cluster by next most selective column
CREATE TABLE raw_regrade_sessions
PARTITION BY DATE(session_date)
CLUSTER BY product_id, channel
AS SELECT * FROM source_table;

-- Redshift — distribution and sort key
CREATE TABLE raw_regrade_sessions (
    session_date    DATE,
    customer_id     VARCHAR(50),
    product_id      VARCHAR(50),
    predicted_score FLOAT
)
DISTKEY(customer_id)
SORTKEY(session_date);
```

- Partition by the column most commonly used in `WHERE` filters — almost always a date
- Add a cluster/sort key on the next most selective filter column (e.g. `product_id`)
- Always include the partition column in `WHERE` clauses — a query without it defeats the partition entirely

---

### 3. Always Index Serving Tables

Aggregate and serving tables queried at runtime (dashboards, feature stores, APIs) must have indexes on every column used in `WHERE` or `JOIN ON` clauses.

```sql
-- PostgreSQL feature store serving table
CREATE TABLE customer_features (
    customer_id     VARCHAR(50)  NOT NULL,
    product_id      VARCHAR(50)  NOT NULL,
    feature_date    DATE         NOT NULL,
    avg_price       FLOAT,
    acceptance_rate FLOAT,
    PRIMARY KEY (customer_id, product_id, feature_date)
);

-- Composite index matching the most common lookup pattern
CREATE INDEX idx_customer_features_lookup
    ON customer_features (customer_id, feature_date DESC);

-- Partial index for active records only — smaller and faster
CREATE INDEX idx_customer_features_recent
    ON customer_features (customer_id)
    WHERE feature_date >= CURRENT_DATE - INTERVAL '90 days';
```

- Composite indexes should match the column order of your most frequent query pattern
- Use partial indexes to index only the rows that are actually queried (e.g. recent records)
- Avoid over-indexing write-heavy tables — each index adds overhead on `INSERT` and `UPDATE`

---

### 4. Use STRUCT Instead of Key-Value Feature Tables

A key-value schema (`session_id`, `feature_key`, `feature_value`) is flexible but expensive — pivoting N features requires N scans or a wide `GROUP BY MAX`. A STRUCT schema stores all features in one row, and columnar storage means the query engine only reads the fields it needs.

```sql
-- ❌ Bad — key-value table, one row per feature
-- session_features: session_id | feature_key      | feature_value
--                  s001       | predicted_score  | 0.82
--                  s001       | discount         | -0.05
--                  s001       | floor            | 10.00

-- Pivoting two features already requires a full scan + GROUP BY
SELECT
    session_id,
    MAX(CASE WHEN feature_key = 'predicted_score' THEN feature_value END) AS predicted_score,
    MAX(CASE WHEN feature_key = 'discount'        THEN feature_value END) AS discount
FROM session_features
GROUP BY session_id

-- ✅ Good — STRUCT schema, one row per session
-- session_features: session_id | features.predicted_score | features.discount | features.floor

CREATE TABLE session_features (
    session_id  STRING,
    features    STRUCT<
        predicted_score FLOAT64,
        discount        FLOAT64,
        floor           FLOAT64,
        ceiling         FLOAT64
    >
);

-- Direct field access — no pivot, no GROUP BY, columnar read of only the needed fields
SELECT session_id, features.predicted_score, features.discount
FROM session_features
WHERE features.discount < 0;
```

The key-value schema also loses type safety — every value is stored as the same type (usually `STRING` or `FLOAT64`), making schema validation harder. STRUCT enforces a type per field at the schema level.

---

## Query Efficiency

### 5. Mid-Layer Aggregation Tables

Never query raw event tables from dashboards or downstream pipelines. Pre-aggregate at the lowest useful grain and query the aggregate layer instead. See also: [dashboard/readme.md](../dashboard/readme.md#3-multi-level-aggregation--avoid-recomputing-shared-elements).

```
raw events (billions of rows)
    → daily aggregate (millions of rows)   ← pipeline reads from here
        → weekly rollup (thousands of rows) ← dashboard reads from here
```

```sql
-- Build the daily aggregate once
CREATE TABLE regrade_sessions_daily AS
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

-- All downstream queries hit the aggregate — not the raw table
SELECT
    product_id,
    SUM(score_sum) / SUM(n)        AS avg_score,
    SUM(acceptance_count) / SUM(n) AS acceptance_rate
FROM regrade_sessions_daily
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY product_id;
```

Store additive elements (`n`, `sum`, `sum_sq`, `count_positive`) — not derived metrics like `avg`. Averages cannot be re-aggregated across groups; sums can.

---

### 6. Delta Load — Only Process New Data

Never reprocess the full history on every pipeline run. Use a watermark to fetch only new records since the last run.

```sql
-- Fetch only new rows
INSERT INTO regrade_sessions_daily (date, product_id, n, score_sum)
SELECT
    DATE(session_date)   AS date,
    product_id,
    COUNT(*)             AS n,
    SUM(predicted_score) AS score_sum
FROM raw_regrade_sessions
WHERE session_date > (
    SELECT MAX(last_processed_at)
    FROM pipeline_watermarks
    WHERE pipeline = 'regrade_sessions_daily'
)
GROUP BY DATE(session_date), product_id;

-- Advance the watermark
UPDATE pipeline_watermarks
SET last_processed_at = CURRENT_TIMESTAMP()
WHERE pipeline = 'regrade_sessions_daily';
```

- Use `>` not `>=` on the watermark to avoid double-counting boundary records
- Backfill by resetting the watermark — no code changes needed
- Partition the raw table by date so the delta scan only touches the latest partition

---

### 7. Don't Read the Same Table Repeatedly

Every table reference in a query is a potential scan. If you need the same data in multiple places, read it once with a CTE.

```sql
-- ❌ Bad — raw_regrade_sessions scanned twice
SELECT a.customer_id, a.predicted_score, b.avg_score
FROM raw_regrade_sessions a
JOIN (
    SELECT product_id, AVG(predicted_score) AS avg_score
    FROM raw_regrade_sessions          -- second scan of the same table
    GROUP BY product_id
) b ON a.product_id = b.product_id

-- ✅ Good — read once, reference twice
WITH sessions AS (
    SELECT customer_id, product_id, predicted_score
    FROM raw_regrade_sessions
    WHERE session_date >= '2025-01-01'
),
product_avg AS (
    SELECT product_id, AVG(predicted_score) AS avg_score
    FROM sessions                      -- reuses the CTE, no second scan
    GROUP BY product_id
)
SELECT s.customer_id, s.predicted_score, p.avg_score
FROM sessions s
JOIN product_avg p ON s.product_id = p.product_id;
```

---

### 8. Prefer GROUP BY MAX over Joins on Record-Level Tables

A key-value record table (one row per entity per metric) is common in feature stores and metric pipelines. Pivoting it with a self-join reads the table once per metric — `GROUP BY MAX` with a `CASE` expression reads it once regardless of how many metrics you need.

```sql
-- ❌ Bad — self-join reads metric_record once per metric
SELECT a.id, a.value AS metric_a, b.value AS metric_b
FROM metric_record a
JOIN metric_record b
  ON a.id = b.id
  AND a.key = 'metric_a'
  AND b.key = 'metric_b'

-- ✅ Good — single scan, pivot with GROUP BY MAX
SELECT
    id,
    MAX(CASE WHEN key = 'metric_a' THEN value END) AS metric_a,
    MAX(CASE WHEN key = 'metric_b' THEN value END) AS metric_b
FROM metric_record
WHERE key IN ('metric_a', 'metric_b')
GROUP BY id
```

Adding a third metric to the join version requires another `JOIN` and another scan. The `GROUP BY MAX` version requires only one more `CASE` expression — the scan count stays at one.

---

### 9. Never ORDER BY in a Pipeline

`ORDER BY` forces a full sort of the result set — one of the most expensive operations in SQL. It has no effect on downstream aggregations or joins, and most warehouse storage formats (Parquet, ORC) do not preserve row order anyway.

```sql
-- ❌ Bad — sorts millions of rows for no downstream benefit
INSERT INTO prediction_features
SELECT customer_id, product_id, predicted_score
FROM raw_regrade_sessions
ORDER BY session_date DESC   -- wasted sort

-- ✅ Good — no ORDER BY in pipeline queries
INSERT INTO prediction_features (customer_id, product_id, predicted_score)
SELECT customer_id, product_id, predicted_score
FROM raw_regrade_sessions
```

Only use `ORDER BY` at the final query layer when the consumer genuinely requires ordered output (e.g. a report or a ranked list). Never in `INSERT INTO`, `CREATE TABLE AS`, or intermediate CTEs.

---

### 10. Don't Use RAND() — Use a Fingerprint Instead

`RAND()` generates a new random value on every row evaluation, which prevents deterministic sampling and makes results non-reproducible. A hash fingerprint on a stable column produces a deterministic, reproducible sample.

```sql
-- ❌ Bad — non-deterministic, different result every run
SELECT customer_id, product_id, predicted_score
FROM raw_regrade_sessions
WHERE RAND() < 0.1   -- 10% sample, but different rows each time

-- ✅ Good — deterministic fingerprint sample, same rows every run
SELECT customer_id, product_id, predicted_score
FROM raw_regrade_sessions
WHERE MOD(ABS(FARM_FINGERPRINT(CAST(customer_id AS STRING))), 10) = 0  -- BigQuery

-- Redshift / Postgres equivalent
WHERE MOD(ABS(HASHTEXT(customer_id::TEXT)), 10) = 0
```

Fingerprint sampling is also faster — it evaluates a hash function once per row rather than calling a random number generator, and the query engine can push the filter down to partition pruning.

---

## Code Reuse & Maintainability

### 11. Use Meaningful Aliases

Aliases are the variable names of SQL — a reader should understand what a column or subquery contains without tracing back through the query. Cryptic single-letter aliases and auto-named expressions make queries hard to review, debug, and maintain.

```sql
-- ❌ Bad — aliases reveal nothing about content
SELECT
    a.id,
    SUM(b.v)          AS s,
    COUNT(*)          AS c,
    SUM(b.v) / COUNT(*) AS x
FROM t1 a
JOIN t2 b ON a.id = b.id
GROUP BY a.id

-- ✅ Good — aliases describe what the value represents
SELECT
    sessions.customer_id,
    SUM(scores.predicted_score)                         AS score_sum,
    COUNT(*)                                            AS session_count,
    SUM(scores.predicted_score) / COUNT(*)              AS avg_predicted_score
FROM regrade_sessions    AS sessions
JOIN prediction_scores   AS scores ON sessions.customer_id = scores.customer_id
GROUP BY sessions.customer_id
```

### Rules

- Name every derived column with `AS <alias>` — never leave expressions unnamed
- Table aliases should be the table name shortened meaningfully (`sessions`, `scores`), not single letters (`a`, `b`)
- CTE names should describe what the CTE contains (`product_avg`, `daily_counts`), not how it is used (`tmp`, `cte1`)
- Aggregation aliases should reflect the aggregation: `score_sum`, `session_count`, `avg_predicted_score` — not `s`, `c`, `x`
- Window function aliases should describe the rank or partition context: `rank_by_date`, `row_num_per_customer`

```sql
-- ❌ Bad — CTE names reveal nothing
WITH tmp AS (...),
     cte2 AS (...)

-- ✅ Good — CTE names describe the data they hold
WITH daily_session_counts AS (...),
     product_acceptance_rates AS (...)
```

---

### 12. Build a Reusable SQL Library

Repeated logic scattered across queries is a maintenance problem — a change to a business rule requires updating every query that implements it. Centralise reusable logic as views, table functions, or scalar functions depending on what is being reused.

#### Table functions — reusable query logic with parameters

```sql
-- ❌ Bad — discount calculation duplicated across many queries
SELECT
    customer_id,
    (avg_recommended_price - predicted_price)
        / avg_recommended_price AS discount
FROM raw_regrade_sessions
WHERE session_date BETWEEN '2025-01-01' AND '2025-01-31'

-- ✅ Good — define once as a table function, call with any date range
CREATE OR REPLACE TABLE FUNCTION discount_sessions(start_date DATE, end_date DATE)
AS (
    SELECT
        customer_id,
        product_id,
        predicted_price,
        avg_recommended_price,
        (avg_recommended_price - predicted_price)
            / NULLIF(avg_recommended_price, 0)  AS discount
    FROM raw_regrade_sessions
    WHERE session_date BETWEEN start_date AND end_date
);

SELECT customer_id, product_id, discount
FROM discount_sessions(DATE '2025-01-01', DATE '2025-01-31')
WHERE discount > 0.1;
```

#### Scalar functions — reusable metric calculations

Statistical metrics that appear in multiple queries should be scalar functions. This ensures consistent calculation and makes queries readable.

```sql
-- Define once
CREATE OR REPLACE FUNCTION variance_from_agg(sum FLOAT64, sum_sq FLOAT64, n INT64)
RETURNS FLOAT64 AS (
    (sum_sq / n) - POW(sum / n, 2)   -- E[x²] - E[x]²
);

CREATE OR REPLACE FUNCTION stddev_from_agg(sum FLOAT64, sum_sq FLOAT64, n INT64)
RETURNS FLOAT64 AS (
    SQRT(variance_from_agg(sum, sum_sq, n))
);

CREATE OR REPLACE FUNCTION psi(actual_pct FLOAT64, expected_pct FLOAT64)
RETURNS FLOAT64 AS (
    (actual_pct - expected_pct) * LN(actual_pct / NULLIF(expected_pct, 0))
);

-- Use anywhere — consistent, readable
SELECT
    product_id,
    stddev_from_agg(score_sum, score_sum_sq, n)   AS score_stddev,
    variance_from_agg(score_sum, score_sum_sq, n) AS score_variance
FROM regrade_sessions_daily
GROUP BY product_id;
```

---

### 13. Dynamic SQL for Parameterised Pipelines

When pipeline logic varies by environment, date range, or configuration, use dynamic SQL inside stored procedures rather than duplicating queries per environment.

```sql
CREATE OR REPLACE PROCEDURE run_feature_pipeline(
    environment STRING,
    run_date    DATE
)
BEGIN
    DECLARE target_table STRING;
    DECLARE query        STRING;

    SET target_table = CONCAT('features_', environment, '_', FORMAT_DATE('%Y%m%d', run_date));

    SET query = CONCAT("""
        CREATE OR REPLACE TABLE `project.dataset.""", target_table, """` AS
        SELECT
            customer_id,
            product_id,
            AVG(predicted_score)    AS avg_score,
            COUNT(*)                AS session_count
        FROM `project.dataset.raw_regrade_sessions`
        WHERE DATE(session_date) = '""", run_date, """'
        GROUP BY customer_id, product_id
    """);

    EXECUTE IMMEDIATE query;
END;

-- Run for any environment and date without duplicating the query
CALL run_feature_pipeline('prod', DATE '2025-01-15');
CALL run_feature_pipeline('exp',  DATE '2025-01-15');
```

---

### 14. Zero-Downtime Table Updates

Never update a serving table in-place while it is being read. Build the new version in a staging table, validate it, then swap atomically. Readers always see either the old or the new table — never a partial state.

```sql
-- BigQuery — atomic swap inside a stored procedure
CREATE OR REPLACE PROCEDURE update_customer_features()
BEGIN
    -- 1. Build new version in staging
    CREATE OR REPLACE TABLE customer_features_staging AS
    SELECT
        customer_id,
        product_id,
        DATE(session_date)               AS feature_date,
        AVG(predicted_score)             AS avg_score,
        COUNTIF(accepted = 1) / COUNT(*) AS acceptance_rate
    FROM raw_regrade_sessions
    WHERE session_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
    GROUP BY customer_id, product_id, DATE(session_date);

    -- 2. Validate — abort if staging is empty
    IF (SELECT COUNT(*) FROM customer_features_staging) = 0 THEN
        RAISE USING MESSAGE = 'Staging table is empty — aborting swap';
    END IF;

    -- 3. Atomic swap
    ALTER TABLE customer_features         RENAME TO customer_features_old;
    ALTER TABLE customer_features_staging RENAME TO customer_features;

    -- 4. Drop old table after successful swap
    DROP TABLE IF EXISTS customer_features_old;
END;

CALL update_customer_features();
```

```sql
-- PostgreSQL equivalent
BEGIN;
    ALTER TABLE customer_features     RENAME TO customer_features_old;
    ALTER TABLE customer_features_new RENAME TO customer_features;
    DROP TABLE customer_features_old;
COMMIT;
```

- Always validate the staging table (row count, null rate, key metrics) before the swap
- The rename is near-instantaneous — downtime is effectively zero
- Keep the old table until the next successful run as a rollback target

---

## Anti-Patterns

| Anti-pattern | Why it hurts | Fix |
|-------------|-------------|-----|
| Positional column references (`ORDER BY 1`) | Breaks silently on schema change | Always use explicit column names |
| Querying raw tables from dashboards | Full scan on every load | Build mid-layer aggregate tables |
| Full history reprocess on every run | Wasteful, slow | Delta load with a watermark |
| Reading the same table twice in one query | Two scans where one would do | Use a CTE to read once |
| Self-join to pivot key-value metrics | One scan per metric | `GROUP BY MAX` with `CASE` expressions |
| `ORDER BY` in pipeline queries | Full sort with no downstream benefit | Remove; sort only at the final consumer |
| `RAND()` for sampling | Non-deterministic, no partition pruning | Use `FARM_FINGERPRINT` or `HASHTEXT` |
| Key-value feature table | Pivot requires full scan per feature | Use STRUCT schema |
| No partition on large tables | Full table scan on every query | Partition by date; always filter on partition column |
| No index on serving tables | Slow lookups at runtime | Add composite and partial indexes |
| In-place update of serving table | Readers see partial state | Build staging, validate, atomic rename |
| Duplicated business logic | One rule change requires N updates | Centralise in table functions or scalar functions |
| Cryptic aliases (`a`, `s`, `cte1`) | Reader must trace back through the query to understand values | Use descriptive aliases for columns, tables, and CTEs |
