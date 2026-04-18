# Git

Git is the foundation of every CI/CD pipeline. In ML projects, repositories contain not just code but also configs, notebooks, and references to large artefacts — making good Git hygiene especially important.

---

## .gitignore

ML repositories generate large binary files and sensitive data that must never be committed. A well-configured `.gitignore` is the first line of defence:

```gitignore
# Model artefacts
*.pkl
*.joblib
*.onnx
*.pt
*.pth
*.h5
*.pb
*.safetensors
*.bin

# Data files
*.csv
*.parquet
*.feather
*.arrow
*.tfrecord
*.npy
*.npz
data/

# Credentials & secrets
.env
*.pem
*.key

# Python
__pycache__/
*.pyc
*.egg-info/
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Notebook outputs
*.ipynb_checkpoints/
```

If large files must be tracked, use **Git LFS** rather than committing them directly.

---

## Pre-Commit Hooks

Pre-commit hooks run automatically before every commit — catching issues before they enter the repository. Use the `pre-commit` framework to manage hooks declaratively:

```yaml
# .pre-commit-config.yaml
repos:
  # Formatting
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        args: [--line-length=120]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=120]

  # Linting
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff

  # Secrets detection
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets

  # General checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: [--maxkb=500]
      - id: check-merge-conflict
      - id: trailing-whitespace
      - id: end-of-file-fixer
```

Install with:

```bash
pip install pre-commit
pre-commit install
```

| Hook | Purpose |
|------|---------|
| `black` | Enforces consistent code formatting |
| `isort` | Sorts imports into the correct group order |
| `ruff` | Fast linting — catches unused imports, undefined variables, code smells |
| `detect-secrets` | Scans for API keys, tokens, passwords, and other secrets |
| `check-added-large-files` | Blocks commits containing large files (model weights, datasets) |
| `check-merge-conflict` | Prevents committing unresolved merge conflict markers |

---

## PII & Sensitive Data Prevention

Personally identifiable information (PII) and credentials must never enter the repository — once committed, they persist in Git history even after deletion.

- Use `detect-secrets` in pre-commit to catch API keys, tokens, and passwords automatically
- Add a custom regex hook to catch common PII patterns (emails, phone numbers):
  ```yaml
  - repo: local
    hooks:
      - id: no-pii
        name: Check for PII patterns
        entry: bash -c 'grep -rPn "(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)" "$@" && exit 1 || exit 0' --
        language: system
        types: [python]
  ```
- Use placeholders in code and examples: `<email>`, `<phone>`, `<api_key>`, `<name>`
- Exclude `.env` files, service account keys, and data directories via `.gitignore`
- If PII is accidentally committed, purge it from history immediately using `git filter-repo` or `BFG Repo-Cleaner`

---

## Commit Messages

Use **conventional commits** to structure every commit message. This enables automated version bumping and changelog generation — see [semantic_build](../semantic-build/readme.md) for the full versioning workflow.

### Format

```
<type>[optional scope][!]: <short description>

[optional body]

[optional footer]
```

### Types

| Type | When to use | Version bump |
|------|-------------|-------------|
| `feat:` | New feature or capability | MINOR |
| `fix:` | Bug fix | PATCH |
| `feat!:` or `BREAKING CHANGE:` | Breaking change | MAJOR |
| `chore:` | Dependency updates, maintenance | PATCH or none |
| `docs:` | Documentation only | None |
| `ci:` | CI/CD pipeline changes | None |
| `refactor:` | Code restructure, no behaviour change | None |
| `test:` | Adding or updating tests | None |

### Examples

```bash
feat: add batch inference endpoint
fix: correct null handling in feature preprocessing
feat(serving)!: change predict response schema to v2
chore: bump xgboost to 2.1.0
docs: update API usage examples
ci: add staging deployment rule
test: add unit tests for ladder enforcement
```

### Tips

- Write in imperative mood: *"add feature"* not *"added feature"*
- Keep the first line under 72 characters
- Use the body to explain *why*, not *what* — the diff shows what changed
- Reference issue/ticket IDs in the footer: `Closes: #123`
- Enforce conventional commits with `commitlint` or `commitizen` — reject non-conforming messages in CI or pre-commit

---

## Branch Strategy

| Branch | Purpose | Merges to |
|--------|---------|-----------|
| `main` | Production-ready code | — |
| `staging` | Pre-production validation | `main` |
| `feature/*` | New features and experiments | `staging` or `main` |
| `fix/*` | Bug fixes | `staging` or `main` |

- Protect `main` and `staging` — require CI to pass + peer approval before merge
- Delete feature branches after merge to keep the repository clean
- Use squash merges for feature branches to keep history readable
