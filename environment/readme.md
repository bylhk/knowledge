# Environment Best Practices

A well-structured local environment setup reduces the gap between development and production, catches bugs before they reach CI, and lets every team member run the same code regardless of their operating system.

---

## Core Principle — Test Locally First

Never use cloud resources as a development sandbox. Cloud runs are slow to iterate, expensive, and produce noisy logs that are hard to debug. Every change should be validated locally before it is pushed to CI or deployed to a cloud environment.

```
local test → CI pipeline → exp → stag → prod
```

- If it fails locally, it will fail in CI — fix it before pushing
- Cloud environments are for validation and production, not for debugging
- A bug found locally costs seconds to fix; the same bug found in CI costs minutes; in production it costs hours

---

## 1. Cross-Platform Runner

A cross-platform runner is a single entry point script that abstracts where the code runs. Instead of maintaining separate instructions for local, Docker, and cloud execution, one script handles all targets via a flag.

```
python runner.py --[local|docker|gcp|aws|azure]
```

This pattern improves development speed because:
- Developers switch between environments with one flag change — no rewriting commands
- Bugs are reproducible across environments — the same runner logic executes everywhere
- Onboarding is reduced to one command — new team members do not need to learn environment-specific setup

### Example runner structure

```python
"""
runner.py — Cross-platform pipeline entry point.

Executes the training or inference pipeline in the target environment.
Use --local for development, --docker for container validation,
and --gcp / --aws for cloud execution.

Functions / Routines
--------------------
main
    Parse arguments and dispatch to the appropriate executor.

Notes
-----
All environment-specific config is loaded via TemplateConfigs.
Never hardcode environment names or cloud project IDs here.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform pipeline runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local",  action="store_true", help="Run in local Python environment")
    group.add_argument("--docker", action="store_true", help="Run inside Docker container")
    group.add_argument("--gcp",    action="store_true", help="Submit to Vertex AI")
    group.add_argument("--aws",    action="store_true", help="Submit to AWS ECS / Batch")
    group.add_argument("--azure",  action="store_true", help="Submit to Azure ML")
    args = parser.parse_args()

    if args.local:
        from executors.local import run
    elif args.docker:
        from executors.docker import run
    elif args.gcp:
        from executors.gcp import run
    elif args.aws:
        from executors.aws import run
    elif args.azure:
        from executors.azure import run

    run()


if __name__ == "__main__":
    main()
```

### Rules

- All targets must accept the same input interface — if `--local` requires a config path, so does `--gcp`
- Cloud targets should guard against local execution: check for `CI_COMMIT_REF_NAME` or an `IS_DEPLOYMENT` env var before submitting jobs
- Log the active target at startup so pipeline logs are self-identifying

---

## 2. Local Environment with venv / uv

Use a virtual environment for every project — never install project dependencies into the system Python. This prevents version conflicts between projects and makes the environment reproducible.

### venv (standard library)

```bash
# Create and activate
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Install the local package in editable mode
pip install -e .

# Deactivate
deactivate
```

### uv (recommended — significantly faster)

`uv` is a drop-in replacement for `pip` and `venv` written in Rust. It resolves and installs dependencies 10–100x faster than pip and produces a lockfile for reproducible installs.

```bash
# Install uv
pip install uv

# Create environment and install in one step
uv venv .venv
source .venv/bin/activate

# Install from requirements (fast)
uv pip install -r requirements.txt

# Install with lockfile for reproducible environments
uv pip compile requirements.in -o requirements.txt   # generate pinned lockfile
uv pip sync requirements.txt                          # install exactly what is in the lockfile
```

### pyproject.toml — pin your tools

```toml
[tool.black]
line-length = 120

[tool.isort]
profile = "black"
line_length = 120

[project]
name = "my-ml-project"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "scikit-learn>=1.6",
    "fastapi>=0.115",
]
```

### Rules

- Always use a virtual environment — never `pip install` into system Python
- Pin exact versions in `requirements.txt` for reproducibility — `numpy==1.26.4`, not `numpy>=1.22`
- Commit `requirements.txt` (pinned lockfile) to version control — not just `requirements.in`
- Use `uv pip sync` rather than `pip install -r` in CI — it removes packages not in the lockfile, preventing environment drift
- Keep `.venv/` in `.gitignore` — never commit the virtual environment directory

---

## 3. Local Environment with Docker

Running locally inside the same Docker image used in production eliminates the "works on my machine" class of bugs. The container is the environment — OS, Python version, system libraries, and all.

### When to use Docker locally

| Scenario | Use venv | Use Docker |
|----------|----------|------------|
| Fast iteration on Python logic | ✅ | — |
| Testing the full service (API + Redis + DB) | — | ✅ |
| Validating the production image before CI | — | ✅ |
| Reproducing a production bug | — | ✅ |
| Running on a different OS than production | — | ✅ |

### docker-compose for local development

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        RUN_ENVIRONMENT: local
    ports:
      - "8080:8080"
    env_file: env/local.env
    volumes:
      - ./src:/app/src        # mount source for hot reload without rebuilding
    depends_on:
      - redis
      - db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: features
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    ports:
      - "5432:5432"
```

```bash
# Start the full local stack
docker compose up

# Rebuild after Dockerfile or dependency changes
docker compose up --build

# Run a one-off command inside the container
docker compose run app python runner.py --local

# Tear down
docker compose down
```

### Matching the production image locally

```bash
# Build with the same args used in CI
docker build \
  --build-arg RUN_ENVIRONMENT=exp \
  --build-arg BASE_IMAGE=python:3.12-slim \
  -t my-ml-project:local .

# Run interactively to inspect the environment
docker run --rm -it my-ml-project:local bash
```

### Rules

- Use `volumes` to mount source code into the container during development — avoids rebuilding the image on every code change
- Use `env_file` to load local credentials — never hardcode them in `docker-compose.yml`
- Keep a `local.env` file (gitignored) for local overrides — mirror the structure of CI variables
- Always test with `--build` before pushing — a cached layer may hide a broken dependency install
- Use the same base image tag locally as in CI — `python:3.12-slim`, not `python:latest`

---

## 4. Environment Variable Management

Environment variables are the boundary between code and configuration. They split into two categories — see [cicd/readme.md](../cicd/readme.md#devops-ci-variables) and [cicd/configuration.md](../cicd/configuration.md#1-config-management) for the full treatment.

| | CI Variables | Config Files |
|---|---|---|
| Credentials & secrets | ✅ | ❌ |
| Infrastructure endpoints | ✅ | ❌ |
| Hyperparameters & feature lists | ❌ | ✅ |
| Business rules & thresholds | ❌ | ✅ |
| Version controlled & reviewable | ❌ | ✅ |

Rule of thumb: if it is sensitive or infrastructure-specific, use CI variables. If it benefits from code review and version history, use config files.

### Inject CI variables directly into the local environment

Rather than loading a `.env` file in application code, export CI variables directly in your shell profile or use `direnv` for per-project overrides. This keeps local and CI execution identical — the same `os.getenv()` calls work in both contexts without any file loading in the application.

```bash
# ~/.zshrc or ~/.bashrc — export once, available in every session
export RUN_ENVIRONMENT=exp
export IS_DEPLOYMENT=false
export RDS_HOST=localhost
export ARTIFACT_LOCATION=./artifacts
export PYTHONUNBUFFERED=1
export PORT=8080
```

For per-project overrides without polluting the shell profile, use `direnv` — it automatically exports variables from a `.envrc` file when you enter the project directory and unsets them when you leave:

```bash
# .envrc (gitignored) — activated automatically by direnv
export RUN_ENVIRONMENT=exp
export ARTIFACT_LOCATION=./artifacts
```

```bash
# Install direnv and hook into shell
brew install direnv
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Allow the .envrc in the project directory
direnv allow .
```

As noted in [cicd/configuration.md](../cicd/configuration.md#1-config-management), config files support environment variable interpolation — so CI variables flow directly into YAML config at runtime:

```yaml
# config/exp/serving_config.yaml
database:
  host: ${RDS_HOST}       # injected from CI variable or shell export
  pool_size: 10           # static, version-controlled
```

```python
import os

run_environment = os.getenv("RUN_ENVIRONMENT", "exp")
is_deployment   = os.getenv("IS_DEPLOYMENT", "false").lower() == "true"
artifact_root   = os.getenv("ARTIFACT_LOCATION", "./artifacts")
```

### Validate required variables at startup

Fail fast with a clear error rather than a cryptic runtime failure later:

```python
REQUIRED_ENV_VARS = ["RUN_ENVIRONMENT", "ARTIFACT_LOCATION"]

missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {missing}")
```

### Rules

- Export variables in your shell profile or via `direnv` — no `.env` file loading in application code
- Commit `env/example.env` with placeholder values as the onboarding reference — new team members copy it and export the values themselves
- Use `os.getenv("VAR", default)` with a sensible local default — code must run locally without every production variable being set
- In CI, scope variables per environment group (`dev`, `staging`, `prod`) and mark sensitive values as **masked** and **protected** — see [cicd/readme.md](../cicd/readme.md#devops-ci-variables)
- Non-sensitive config (hyperparameters, feature lists, thresholds) belongs in version-controlled YAML config files, not environment variables — see [cicd/configuration.md](../cicd/configuration.md#1-config-management)

---

## Anti-Patterns

| Anti-pattern | Why it hurts | Fix |
|-------------|-------------|-----|
| Debugging directly in cloud | Slow iteration, expensive, noisy logs | Reproduce locally first; push only when it works |
| System Python for project dependencies | Version conflicts across projects | Always use venv or uv per project |
| `pip install` without pinned versions | Environment drifts between machines and CI | Pin exact versions in a lockfile |
| Hardcoded environment names in code | Breaks when environment changes | Use `os.getenv()` with defaults |
| Committing `.env` files | Credentials leak into version control | Gitignore all `.env` files; use `example.env` as template |
| Different base image locally vs CI | "Works on my machine" bugs | Pin the same image tag in both places |
| No cross-platform runner | Every developer has their own run instructions | Single `runner.py --[local\|docker\|cloud]` entry point |
