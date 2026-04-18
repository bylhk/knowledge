# CI/CD Configuration

Practical tips for managing the moving parts in an ML CI/CD pipeline — configs, rules, changes, images, and artefacts.

---

## 1. Config Management

### Use a layered config strategy

Separate configs by concern and override by environment:

```
config/
├── base.yaml            # shared defaults
├── dev/
│   └── override.yaml    # dev-specific overrides
├── staging/
│   └── override.yaml
└── prod/
    └── override.yaml
```

At runtime, merge base + environment override — so you only define what differs per environment.

Since YAML is language-agnostic, the same config files can be reused across modules written in different languages (Python, Java, Go, etc.) — each module simply loads the shared YAML with its own config parser.

### Rules

- Keep configs **declarative** — describe *what*, not *how*
- Validate configs at pipeline start with a schema (JSON Schema, Pydantic, Cerberus) — fail fast on typos or missing fields
- Never store secrets in config files — use CI variables for credentials, config files for everything else
- Version config changes alongside code — they should go through the same review process
- Use a config loader that supports environment variable interpolation:
  ```yaml
  database:
    host: ${DB_HOST}       # injected from CI variable
    pool_size: 10          # static, version-controlled
  ```

---

## 2. CI Rules & Triggers

### Only run what changed

Use path-based rules to avoid running the entire pipeline on every commit:

```yaml
# GitLab CI example
train_model:
  rules:
    - changes:
        - "src/training/**"
        - "config/**/training_config.yaml"

deploy_service:
  rules:
    - changes:
        - "src/serving/**"
        - "Dockerfile"
```

### Condition rules with CI variables

Use `if` conditions with CI variables to control pipeline behaviour — especially useful for ad-hoc shortcuts that let you trigger specific stages on demand:

```yaml
# Ad-hoc shortcut: set RUN_TRAINING=true in the CI UI to trigger training on any branch
train_model:
  rules:
    - if: '$RUN_TRAINING == "true"'
      when: always
    - changes:
        - "src/training/**"

# Ad-hoc shortcut: skip expensive stages during development
deploy_service:
  rules:
    - if: '$SKIP_DEPLOY == "true"'
      when: never
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: on_success

# Shortcut: force full pipeline run
lint:
  rules:
    - if: '$RUN_ALL == "true"'
      when: always
    - changes:
        - "src/**"
```

This allows developers to set ad-hoc CI variables in the pipeline UI (or via API) to override default trigger behaviour without changing the pipeline config.

### Environment-based rules

Use branch or variable conditions to control which stages run per environment:

```yaml
# Dev: run on feature branches — lint, test, build only
lint:
  rules:
    - if: '$CI_COMMIT_BRANCH != "main" && $CI_COMMIT_BRANCH != "staging"'

# Staging: run on staging branch — full pipeline including deploy
deploy_staging:
  rules:
    - if: '$CI_COMMIT_BRANCH == "staging"'
  environment: staging

# Prod: run on main branch — full pipeline with manual approval gate
deploy_prod:
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
  environment: prod
```

| Environment | Branch / Trigger | Stages run |
|-------------|-----------------|------------|
| Dev | Feature branches | lint, pre-tests, build |
| Staging | `staging` branch | All stages, auto-deploy |
| Prod | `main` branch | All stages, manual approval before deploy |

### Rules

- Define **shortcut rules** per component — each component only triggers when its own files change
- Use `if` conditions with ad-hoc CI variables (`RUN_TRAINING`, `SKIP_DEPLOY`, `RUN_ALL`) as shortcuts to override default triggers from the pipeline UI
- Use branch rules to control what runs where: feature branches run lint + tests, main branch runs the full pipeline
- Tag-based triggers for releases: `v*` tags trigger build + deploy
- Avoid `when: always` on expensive stages (training, deployment) — gate them behind rules or manual approval
- Document your trigger rules in the repo — future contributors need to understand what triggers what

---

## 3. Change Management

### Use conventional commits to track intent

```
feat: add batch inference endpoint
fix: correct feature scaling in preprocessing
chore: update training dependencies
refactor!: change prediction response schema
```

This enables automated changelogs, version bumping, and clear audit trails. See [semantic-build/readme.md](../semantic-build/readme.md) for details on how conventional commits drive automated versioning and artefact tagging.

### Rules

- Require merge/pull requests for all changes to protected branches — no direct pushes
- Use branch protection rules: require CI to pass + at least one approval before merge
- Keep changes small and focused — one concern per merge request
- Tag every release with a semantic version — makes rollback targets explicit
- Maintain a `CHANGELOG.md` (auto-generated from commits or manually curated)
- For ML-specific changes, log what changed and why:

| Change type | What to record |
|-------------|---------------|
| Data change | Data snapshot ID, row count delta, schema diff |
| Model change | Metrics before/after, hyperparameter diff |
| Code change | Git diff, linked ticket/issue |
| Config change | YAML diff, affected environments |

---

## 4. Image Management

### Tag strategy

Use multiple tags per image for different purposes:

```
registry/model-serve:abc123f          # Git SHA — exact traceability
registry/model-serve:1.2.0            # SemVer — human-readable release
registry/model-serve:1.2.0-staging    # SemVer + env — promotion tracking
```

### Rules

- Never rely solely on `latest` — it is mutable and ambiguous
- Use immutable tags (Git SHA or SemVer) for anything deployed to staging or prod
- Set up a retention policy to auto-delete old images — registries grow fast
  - Keep the last N versions per environment
  - Keep all images tagged with a SemVer release
  - Delete untagged and feature-branch images after 7–14 days
- Scan images for vulnerabilities in CI before pushing (`trivy`, `grype`, `docker scout`)
- Use a base image registry for shared base images — rebuild downstream images when the base updates
- Separate training images from serving images — training images are large (GPU libs, data tools), serving images should be minimal

---

## 5. Artefact Management

ML projects produce more artefact types than traditional software:

| Artefact | Examples | Storage |
|----------|----------|---------|
| Model files | `.pkl`, `.onnx`, `.pt`, `.joblib` | Model registry, cloud storage (S3, GCS) |
| Data snapshots | Parquet, CSV, Feather | Data lake, versioned bucket |
| Evaluation reports | Metrics JSON, confusion matrix plots | CI artefacts, model registry |
| Built packages | Wheels, container images | Package registry, container registry |
| Config snapshots | Frozen YAML used for a training run | Logged with model in registry |

### Rules

- **Immutability** — once published, an artefact version is never overwritten
- **Lineage** — every model artefact should record:
  - Git commit SHA
  - Data snapshot ID
  - Training config hash
  - Evaluation metrics
  - Parent model version (if fine-tuned)
- **Retention policy** — define how long artefacts are kept:
  - Production models: retained indefinitely (or until decommissioned)
  - Experiment models: auto-delete after 30–90 days
  - CI build artefacts: auto-delete after 7–14 days
- **Access control** — production artefact buckets should be read-only for most users; only CI can write
- **Naming convention** — use a consistent pattern:
  ```
  {project}/{model_name}/v{version}/{filename}
  e.g. pricing/xgboost/v12/model.pkl
  ```
- **Download at runtime, not at build time** — keep model files out of container images where possible; pull from storage at startup

---

## 6. General Rules

- **Audit everything** — CI logs, who triggered what, which artefact was deployed where
- **Automate cleanup** — stale branches, old images, expired artefacts; don't rely on manual housekeeping
- **Pin versions everywhere** — dependencies, base images, tool versions; avoid `latest` in any context
- **Centralise shared logic** — reusable CI templates, shared base images, common config schemas
- **Document the pipeline** — a diagram or table showing stages, triggers, and artefact flow saves hours of onboarding time
