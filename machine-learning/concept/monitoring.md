# Monitoring Best Practices

A model that is deployed but not monitored is not a production system — it is a time bomb. Models degrade silently: input distributions shift, upstream schemas change, and business conditions evolve. Monitoring is what turns a one-time deployment into a reliable, long-running system.

```
infrastructure health → model behaviour → business KPIs
```

All three layers must be monitored. A healthy service with a degraded model is a silent failure.

---

## 1. What to Monitor

### Layer 1 — Infrastructure

| Metric | Alert condition |
|--------|----------------|
| Request volume | Drop > 20% vs 7-day average |
| Error rate | > 1% of requests returning errors |
| P95 latency | Exceeds latency budget |
| Feature fetch failure rate | > 0% — any failure is significant |
| Cache hit rate | Drop > 20% — may indicate upstream change |

### Layer 2 — Model behaviour

| Metric | Alert condition |
|--------|----------------|
| Prediction score distribution (PSI) | PSI > 0.2 vs training baseline |
| Null / missing feature rate | Increase > threshold per feature |
| Feature distribution shift | PSI > 0.1 on key input features |
| Prediction confidence distribution | Shift in mean or variance |

### Layer 3 — Business KPIs

| Metric | Alert condition |
|--------|----------------|
| Acceptance rate | Drop > X% week-over-week |
| Revenue per session | Drop > X% week-over-week |
| Conversion rate | Drop > X% week-over-week |

Business KPIs are the ground truth. Infrastructure and model metrics are early warning signals — business KPIs confirm whether the degradation is real.

---

## 2. Data Drift Detection

Data drift occurs when the distribution of input features in production diverges from the training distribution. It is the most common cause of silent model degradation.

### Population Stability Index (PSI)

PSI measures how much a distribution has shifted. It is computed per feature by comparing production bucket proportions against training bucket proportions.

```python
import numpy as np

def compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Population Stability Index between reference and production distributions.

    Parameters
    ----------
    reference : np.ndarray
        Values from the training / reference period.
    production : np.ndarray
        Values from the production / current period.
    n_bins : int
        Number of equal-width bins.
    epsilon : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        PSI value. < 0.1: stable, 0.1–0.2: minor shift, > 0.2: significant drift.
    """
    bins          = np.linspace(
        min(reference.min(), production.min()),
        max(reference.max(), production.max()),
        n_bins + 1,
    )
    ref_counts, _ = np.histogram(reference,  bins=bins)
    prod_counts,_ = np.histogram(production, bins=bins)

    ref_pct  = ref_counts  / (ref_counts.sum()  + epsilon)
    prod_pct = prod_counts / (prod_counts.sum() + epsilon)

    ref_pct  = np.clip(ref_pct,  epsilon, None)
    prod_pct = np.clip(prod_pct, epsilon, None)

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


PSI_THRESHOLDS = {
    "stable":       0.1,
    "minor_shift":  0.2,
    # > 0.2: significant drift — investigate and consider retraining
}
```

### Two-sample KS test

The Kolmogorov-Smirnov test detects whether two samples come from the same distribution:

```python
from scipy import stats

def ks_drift_test(
    reference: np.ndarray,
    production: np.ndarray,
    significance: float = 0.05,
) -> dict[str, float | bool]:
    """
    Run a two-sample KS test for distribution drift.

    Returns
    -------
    dict
        statistic, p_value, and is_drifted flag.
    """
    statistic, p_value = stats.ks_2samp(reference, production)
    return {
        "statistic":  float(statistic),
        "p_value":    float(p_value),
        "is_drifted": bool(p_value < significance),
    }
```

### Monitor drift per feature daily

```python
def monitor_feature_drift(
    reference_features: np.ndarray,
    production_features: np.ndarray,
    feature_names: list[str],
    logger,
) -> dict[str, float]:
    psi_scores = {}
    for i, name in enumerate(feature_names):
        psi = compute_psi(reference_features[:, i], production_features[:, i])
        psi_scores[name] = psi

        if psi > PSI_THRESHOLDS["minor_shift"]:
            logger.warning(
                "WARNING: feature drift detected. feature=%s psi=%.4f", name, psi
            )

    return psi_scores
```

---

## 3. Model Output Drift

Monitor the distribution of model predictions independently of input features. A shift in prediction scores can indicate model degradation even when individual features appear stable.

```python
def monitor_prediction_drift(
    reference_scores: np.ndarray,
    production_scores: np.ndarray,
    logger,
    run_date: str,
) -> None:
    psi = compute_psi(reference_scores, production_scores)

    logger.info(
        "prediction_drift_check. psi=%.4f run_date=%s", psi, run_date
    )

    if psi > PSI_THRESHOLDS["minor_shift"]:
        logger.warning(
            "WARNING: prediction score drift detected. psi=%.4f run_date=%s",
            psi, run_date,
        )
```

---

## 4. Business KPI Monitoring

Business KPIs are the ground truth signal. Monitor them on a rolling window and alert on statistically significant drops.

### Week-over-week z-test on acceptance rate

```python
def monitor_acceptance_rate(
    current_accepted: int,
    current_total: int,
    previous_accepted: int,
    previous_total: int,
    logger,
    run_date: str,
) -> None:
    """
    Run a z-test for proportions comparing current vs previous week acceptance rate.
    Alert if the drop is statistically significant.
    """
    p_current  = current_accepted  / current_total
    p_previous = previous_accepted / previous_total
    p_pool     = (current_accepted + previous_accepted) / (current_total + previous_total)

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / current_total + 1 / previous_total))
    z  = (p_current - p_previous) / (se + 1e-10)

    logger.info(
        "acceptance_rate_check. current=%.4f previous=%.4f z=%.3f run_date=%s",
        p_current, p_previous, z, run_date,
    )

    if z < -1.96:   # statistically significant drop at 95% confidence
        logger.warning(
            "WARNING: acceptance rate drop is statistically significant. "
            "z=%.3f current=%.4f previous=%.4f run_date=%s",
            z, p_current, p_previous, run_date,
        )
```

---

## 5. Retraining Triggers

Define explicit, measurable conditions that trigger retraining. Ad-hoc retraining is unpredictable and hard to audit.

### Trigger types

| Trigger | Condition | Type |
|---------|-----------|------|
| Schedule | Retrain every N days regardless of performance | Time-based |
| Drift | PSI > 0.2 on a key feature | Data-based |
| Performance | Business KPI drops > X% week-over-week | Performance-based |
| Volume | New data volume exceeds N rows since last training | Volume-based |
| Manual | Triggered by a team member via CI variable | Ad-hoc |

```python
from dataclasses import dataclass

@dataclass
class RetrainingTrigger:
    max_days_since_training: int   = 30     # schedule trigger
    psi_threshold:           float = 0.2    # drift trigger
    kpi_drop_threshold:      float = 0.05   # 5% drop in business KPI
    min_new_rows:            int   = 50_000 # volume trigger

def should_retrain(
    days_since_training: int,
    max_feature_psi: float,
    kpi_change: float,
    new_rows_since_training: int,
    trigger: RetrainingTrigger,
    logger,
) -> bool:
    reasons = []

    if days_since_training >= trigger.max_days_since_training:
        reasons.append(f"schedule: {days_since_training} days since last training")

    if max_feature_psi >= trigger.psi_threshold:
        reasons.append(f"drift: max PSI={max_feature_psi:.4f}")

    if kpi_change <= -trigger.kpi_drop_threshold:
        reasons.append(f"kpi_drop: {kpi_change * 100:.1f}%")

    if new_rows_since_training >= trigger.min_new_rows:
        reasons.append(f"volume: {new_rows_since_training} new rows")

    if reasons:
        logger.info("retraining_triggered. reasons=%s", reasons)
        return True

    return False
```

---

## 6. Monitoring Infrastructure

### Structured log fields for monitoring

Every prediction log entry should include the fields needed to compute all monitoring metrics without joining to other tables:

| Field | Purpose |
|-------|---------|
| `model_version` | Attribute metrics to a specific model |
| `request_id` | Correlate across pipeline steps |
| `entity_id` | Aggregate per entity |
| `prediction_score` | Track score distribution |
| `features_available` | Track feature null rate |
| `latency_ms` | Track serving latency |
| `run_date` | Partition monitoring queries by date |
| `environment` | Separate prod from exp metrics |

### Tooling

| Stack | Metrics store | Dashboard |
|-------|--------------|-----------|
| AWS | CloudWatch + S3 | Grafana / CloudWatch Dashboards |
| GCP | Cloud Monitoring + BigQuery | Looker / Grafana |
| Self-hosted | Prometheus | Grafana |
| Any | Evidently AI | Evidently UI |

### Rules

- Monitor all three layers: infrastructure, model behaviour, and business KPIs
- Define alert thresholds before deployment — not after seeing the first degradation
- Alert on trends, not just on individual anomalies — a single bad day is noise; a week-over-week drop is a signal
- Log model version with every prediction — degradation must be attributable to a specific deployment
- Store reference distributions (training feature stats, score distribution) alongside the model artefact — they are needed to compute PSI at monitoring time
- Retraining triggers must be automated — manual monitoring is not reliable at scale
