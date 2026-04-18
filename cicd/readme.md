# CI/CD

CI/CD automates the build, test, and deployment lifecycle — ensuring reproducible, auditable, and reliable releases. In ML projects, this extends beyond application code to cover tests, training, evaluation, deployment, monitoring and workflow.

---

## Pipeline Structure

```
lint&format → pre-tests → build → train → evaluate → deploy → smoke-test → monitor → workflow
```
\* All stages should contain shortcut CI rules, enabling the pipeline to be tested immediately.

\* Depending on company policy, the ML pipeline can run on CI/CD or a cloud workflow tool.

### Stage Breakdown

#### 1. lint&format
Static code quality checks — runs before anything else to catch style and syntax issues early.

- **Linting** — detect code smells, unused imports, undefined variables (tools: `flake8`, `pylint`, `ruff`)
- **Formatting** — enforce consistent code style automatically (tools: `black`, `isort`, `prettier`)
- **Type checking** — catch type errors statically (tools: `mypy`, `pyright`)

This stage is fast, cheap, and catches the most common issues. If it fails, nothing else runs.

#### 2. pre-tests
Unit tests and data validation — verify that code logic and data assumptions are correct before building.

- **Unit tests** — test individual functions and classes in isolation (`pytest`, `unittest`)
- **Data validation** — check input data schema, types, and distributions (Great Expectations, Pandera, TFX Data Validation)
- **Contract tests** — verify API request/response schemas match expectations

Runs before the build stage to fail fast on logic errors without wasting compute on packaging.

#### 3. build
Package the application and its dependencies into a deployable artefact.

- Install dependencies and compile the project
- Build distributable packages (wheel, sdist, container image)
- Run semantic versioning to tag the artefact (see [semantic-build/readme.md](../semantic-build/readme.md))
- Push the artefact to a registry (package registry, container registry, cloud storage)

The output of this stage is an immutable, versioned artefact that all downstream stages consume.

#### 4. train
Execute the model training pipeline using the built artefact and validated data.

- Load training data from the data warehouse or feature store
- Run hyperparameter tuning (grid search, Bayesian optimisation, random search)
- Train the model with the selected hyperparameters
- Log all parameters, metrics, and the data snapshot version for reproducibility
- Save the trained model artefact to cloud storage

This stage may run on specialised compute (GPU instances, training clusters) and can take minutes to hours.

#### 5. evaluate
Automated quality gate — decides whether the trained model is good enough to deploy.

- Compute evaluation metrics (accuracy, F1, AUC, RMSE, etc.) on a held-out test set
- Compare the candidate model against the currently deployed model (champion/challenger)
- Fail the pipeline if metrics fall below defined thresholds
- Generate evaluation reports and log them as pipeline artefacts
- Register the model in a model registry (MLflow, SageMaker, Vertex AI) with full lineage metadata

If this stage fails, the model is not deployed and the team is notified.

#### 6. deploy
Release the validated model to a target environment.

- Pull the registered model artefact from the model registry
- Deploy to the serving infrastructure (API endpoint, batch job, edge device)
- Apply the deployment pattern appropriate for the environment:

| Pattern | Description |
|---------|-------------|
| Blue/green | Swap traffic from old to new version instantly |
| Canary | Route a small % of traffic to the new model, then gradually increase |
| Shadow | Run new model in parallel without serving its output — compare offline |
| A/B test | Split traffic to measure business impact between versions |

- Update service configuration (autoscaling, resource limits, environment variables)

#### 7. smoke-test
Lightweight post-deployment check — confirms the service is alive and responding correctly.

- Send a sample request to the deployed endpoint
- Verify the response status code, schema, and latency are within bounds
- Does **not** validate model accuracy — only that the service is functional

If the smoke test fails, trigger an automatic rollback to the previous model version.

#### 8. monitor
Continuous post-deployment observability — detects degradation after the pipeline completes.

- **Performance monitoring** — track prediction latency, throughput, error rates
- **Data drift detection** — compare live input distributions against training data distributions
- **Model drift detection** — monitor prediction distributions and business KPIs over time
- **Alerting** — trigger notifications when metrics breach thresholds
- **Logging** — structured logs for every prediction request (input features, output, latency, request ID)

Tools: Prometheus, Grafana, Evidently AI, WhyLabs, SageMaker Model Monitor, Vertex AI Model Monitoring.

#### 9. workflow
Orchestrate downstream or cross-pipeline processes triggered after a successful deployment.

- Trigger retraining schedules (daily, weekly, on-drift)
- Kick off batch inference jobs on new data
- Notify downstream consumers that a new model version is live
- Promote the deployment to the next environment (e.g. `staging → prod`)
- Update dashboards, model cards, and documentation

This stage connects the CI/CD pipeline to the broader MLOps lifecycle.

---

## Key Principles

- **Fail fast** — run cheap checks (lint, tests) before expensive ones (train, deploy)
- **Idempotency** — every stage is safe to retry without side effects
- **Immutable artefacts** — build once, deploy the same artefact to every environment
- **Gated promotion** — never skip environments; always validate before promoting
- **Reproducibility** — pin dependencies, log seeds, snapshot data versions

---

## DevOps CI Variables

CI variables store credentials and environment-specific secrets that must never appear in code or logs. Grouping variables by environment enables clean multi-environment support.

### What belongs in CI variables

| Category | Examples |
|----------|----------|
| Credentials & tokens | API keys, Git tokens, cloud service account keys |
| Infrastructure endpoints | Database hosts, cache endpoints, model registry URLs |
| Environment selectors | `ENV=staging`, `CLOUD_PROJECT_ID` |
| Feature flags | `ENABLE_GPU=true`, `IS_DEPLOYMENT=true` |

### Rules

- Scope variables per environment (`dev`, `staging`, `prod`) using your CI platform's variable grouping
- Mark sensitive values as **masked** and **protected** — they should never appear in job logs
- Guard deployment jobs so they only execute in CI, never from local machines
- Rotate secrets on a regular schedule and audit access

---

## DevOps Config Files

Non-sensitive, application-level configuration belongs in version-controlled config files. Organise by environment using a folder-per-environment structure: `config/{env}/`.

### Folder structure

```
config/
├── dev/
│   ├── training_config.yaml
│   ├── serving_config.yaml
│   └── infra_config.yaml
├── staging/
│   └── ...
└── prod/
    └── ...
```

### What belongs in config files

| File | Contents |
|------|----------|
| `training_config.yaml` | Feature columns, hyperparameters, data split ratios, training resource specs |
| `serving_config.yaml` | Model endpoint settings, batch size, timeout, caching TTL |
| `infra_config.yaml` | Compute resources, autoscaling rules, storage paths |

### Rules

- Load the correct config at runtime using an environment variable:
  ```python
  env = os.getenv("ENV", "dev")
  config = load_config(f"config/{env}/training_config.yaml")
  ```
- Never hardcode environment names, bucket paths, or project IDs in application code
- Keep config files small and single-purpose — one file per concern
- Use `.env` files for local development, loaded via a dotenv library

---

## CI Variables vs Config Files

| | CI Variables | Config Files |
|---|---|---|
| Credentials & secrets | ✅ | ❌ |
| Infrastructure endpoints | ✅ | ❌ |
| Hyperparameters & feature lists | ❌ | ✅ |
| Business rules & thresholds | ❌ | ✅ |
| Version controlled & reviewable | ❌ | ✅ |

Rule of thumb: if it is sensitive or infrastructure-specific, use CI variables. If it benefits from code review and version history, use config files.

---

## Environment Promotion

Environments should follow a linear promotion path with isolated configs at each stage:

```
dev → staging → prod
```

- Each environment has its own CI variable group and config folder
- Promotion is triggered via CI — never manually deploy to production
- Never skip environments — always validate in lower environments first
- Use infrastructure-as-code (Terraform, CloudFormation, Pulumi) so environments are reproducible
