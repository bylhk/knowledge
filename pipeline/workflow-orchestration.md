# Workflow Orchestration

A workflow orchestrator schedules, monitors, and retries pipeline runs. It is the control plane for the entire ML lifecycle — triggering training on a schedule, reacting to data arrival, and chaining pipelines across environments.

---

## Orchestrator Comparison

| Tool | Best for | Hosting | Key strength |
|------|---------|---------|-------------|
| **Apache Airflow** | Complex DAGs, large teams, mature ecosystem | Self-hosted or managed (MWAA, Cloud Composer) | Huge ecosystem, battle-tested |
| **Prefect** | Python-native, fast iteration, modern UX | Cloud or self-hosted | Minimal boilerplate, dynamic workflows |
| **Dagster** | Data-aware pipelines, asset-based thinking | Cloud or self-hosted | Asset lineage, strong typing, testability |
| **Metaflow** | Data science-first, notebook-friendly | AWS-native | Minimal infra, versioned runs |
| **Kubeflow Pipelines** | ML-specific DAGs on Kubernetes/Vertex AI | GCP Vertex AI or self-hosted | ML artefact tracking, GPU support |

---

## Apache Airflow

Airflow defines pipelines as Python DAGs. Each node in the DAG is a task; edges define execution order and dependencies.

### DAG definition

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    "owner":            "ml-team",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
    "email":            ["ml-alerts@company.com"],
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "training"],
) as dag:

    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=run_data_validation,
        op_kwargs={"date": "{{ ds }}"},   # Jinja template — injects execution date
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="python feature_pipeline.py --date {{ ds }}",
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_training,
        op_kwargs={"date": "{{ ds }}"},
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
    )

    deploy_model = PythonOperator(
        task_id="deploy_model",
        python_callable=run_deployment,
    )

    # Define execution order
    validate_data >> build_features >> train_model >> evaluate_model >> deploy_model
```

### Sensors — wait for data arrival

```python
from airflow.sensors.filesystem import FileSensor
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor

# Wait for a file to appear before proceeding
wait_for_data = GCSObjectExistenceSensor(
    task_id="wait_for_data",
    bucket="my-bucket",
    object="data/{{ ds }}/features.parquet",
    timeout=3600,          # fail after 1 hour
    poke_interval=60,      # check every 60 seconds
    mode="reschedule",     # release the worker slot while waiting
)

wait_for_data >> build_features
```

### XCom — pass data between tasks

```python
def extract_metrics(**context) -> dict:
    metrics = {"auc": 0.82, "n_samples": 50000}
    # Push to XCom
    context["ti"].xcom_push(key="metrics", value=metrics)
    return metrics

def decide_deployment(**context) -> str:
    # Pull from XCom
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    return "deploy" if metrics["auc"] >= 0.78 else "skip"
```

---

## Prefect

Prefect uses Python decorators to define flows and tasks. It is significantly less boilerplate than Airflow and supports dynamic workflows natively.

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    retries=2,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def validate_data(date: str) -> bool:
    """Validate input data for the given date."""
    # validation logic
    return True

@task(retries=2)
def build_features(date: str) -> str:
    """Build feature table and return output path."""
    output_path = f"s3://bucket/features/{date}/"
    # feature engineering logic
    return output_path

@task(retries=1)
def train_model(features_path: str, n_estimators: int) -> dict:
    """Train model and return metrics."""
    # training logic
    return {"auc": 0.82}

@task
def deploy_model(metrics: dict, min_auc: float) -> None:
    if metrics["auc"] < min_auc:
        raise ValueError(f"AUC {metrics['auc']} below threshold {min_auc}")
    # deployment logic

@flow(name="ml-training-pipeline", log_prints=True)
def training_pipeline(
    date: str,
    n_estimators: int = 200,
    min_auc: float = 0.78,
) -> None:
    """End-to-end ML training pipeline."""
    valid         = validate_data(date)
    features_path = build_features(date)
    metrics       = train_model(features_path, n_estimators)
    deploy_model(metrics, min_auc)


# Run locally
if __name__ == "__main__":
    training_pipeline(date="2025-01-15")
```

### Schedule and deploy

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="daily-training",
    schedule=CronSchedule(cron="0 2 * * *", timezone="UTC"),
    parameters={"n_estimators": 200, "min_auc": 0.78},
    work_queue_name="ml-workers",
)
deployment.apply()
```

---

## Dagster

Dagster is asset-centric — pipelines are defined as transformations between data assets rather than task sequences. This makes lineage, testing, and incremental computation natural.

```python
from dagster import asset, AssetIn, define_asset_job, ScheduleDefinition
import numpy as np

@asset(
    group_name="features",
    description="Daily customer feature table",
)
def customer_features(context) -> np.ndarray:
    """Compute customer features for the current partition date."""
    date = context.partition_key
    context.log.info(f"Building features for {date}")
    # feature computation
    return features_array

@asset(
    ins={"features": AssetIn("customer_features")},
    group_name="models",
    description="Trained churn model",
)
def churn_model(context, features: np.ndarray) -> dict:
    """Train churn model from customer features."""
    # training logic
    metrics = {"auc": 0.82}
    context.log.info(f"Training complete. AUC: {metrics['auc']}")
    return {"model": trained_model, "metrics": metrics}

@asset(
    ins={"model_bundle": AssetIn("churn_model")},
    group_name="serving",
)
def deployed_model(context, model_bundle: dict) -> None:
    """Deploy model if it meets the quality threshold."""
    if model_bundle["metrics"]["auc"] < 0.78:
        raise ValueError("Model did not meet quality threshold")
    # deployment logic

# Define a job that materialises all assets
training_job = define_asset_job(
    name="daily_training_job",
    selection=["customer_features", "churn_model", "deployed_model"],
)

# Schedule
daily_schedule = ScheduleDefinition(
    job=training_job,
    cron_schedule="0 2 * * *",
)
```

---

## Choosing an Orchestrator

| Requirement | Recommended |
|-------------|-------------|
| Large existing Airflow investment | Airflow |
| New project, Python-first, fast iteration | Prefect |
| Data lineage and asset tracking are important | Dagster |
| ML-specific DAGs on GCP | Kubeflow / Vertex AI Pipelines |
| Data science team, AWS, minimal infra | Metaflow |
| Complex dependencies, many teams | Airflow or Dagster |

---

## Rules

- Every pipeline run must be idempotent — re-running with the same inputs produces the same outputs
- Always parameterise the execution date — never hardcode `today()` inside a task
- Use sensors or data-aware triggers rather than fixed schedules where possible — run when data arrives, not on a timer
- Set retries and timeouts on every task — a task that hangs indefinitely blocks the entire pipeline
- Log row counts and key metrics at every step — silent data volume drops are harder to detect than errors
- Never deploy from a pipeline without a quality gate — always evaluate before deploying
