# Semantic Build

Semantic build combines **semantic versioning** with **automated build processes** to produce traceable, consistently versioned artefacts from CI/CD pipelines.

---

## Semantic Versioning (SemVer)

Every release is tagged with a version in the format:

```
MAJOR.MINOR.PATCH
```

| Component | When to increment | Example |
|-----------|-------------------|---------|
| **MAJOR** | Breaking changes — API contracts change, model input/output schema changes, incompatible config format | `1.0.0 → 2.0.0` |
| **MINOR** | New features or improvements — new endpoint, new model version, added fields (backward compatible) | `1.0.0 → 1.1.0` |
| **PATCH** | Bug fixes, performance tweaks, dependency updates (no behaviour change) | `1.0.0 → 1.0.1` |

### Pre-release and build metadata

SemVer supports optional suffixes:

```
1.2.0-alpha.1       # pre-release: unstable, for testing
1.2.0-rc.1          # release candidate: feature-complete, final validation
1.2.0+build.abc123  # build metadata: Git SHA, CI job ID (ignored in precedence)
```

---

## Why It Matters for ML Projects

ML projects have **two versioned artefacts** that evolve independently:

| Artefact | What changes | Versioned how |
|----------|-------------|---------------|
| Application code | API logic, preprocessing, serving infrastructure | SemVer (`1.2.0`) |
| Model | Retrained weights, new features, new data | Model version (`v12`, registry ID) |

Semantic versioning on the application ensures:
- Consumers know if an update is safe to adopt (PATCH) or requires migration (MAJOR)
- Rollbacks target a specific, immutable version
- Model registry entries link to the exact application version they were trained/served with

---

## Automated Version Bumping

### Conventional Commits

Use structured commit messages to determine the version bump automatically:

```
feat: add new feature endpoint          → MINOR bump
fix: correct preprocessing null check   → PATCH bump
feat!: change predict response schema   → MAJOR bump
chore: update dependencies              → PATCH bump (or no bump)
```

Format: `<type>[optional scope][!]: <description>`

| Prefix | Bump | Description |
|--------|------|-------------|
| `fix:` | PATCH | Bug fix |
| `feat:` | MINOR | New feature |
| `feat!:` or `BREAKING CHANGE:` | MAJOR | Breaking change |
| `chore:`, `docs:`, `ci:`, `refactor:` | PATCH or none | Maintenance |

### CI automation tools

| Tool | Platform | How it works |
|------|----------|-------------|
| `semantic-release` | GitHub/GitLab | Reads commit history, bumps version, creates tag and changelog |
| `commitizen` | Any | Python-native, enforces conventional commits, bumps version |
| `release-please` | GitHub | Google's tool, creates release PRs with changelog |
| `bump2version` | Any | Config-driven version bumping across files |

Example CI stage:

```yaml
semantic_build:
  stage: build
  script:
    - npx semantic-release  # or: cz bump --changelog
  only:
    - main
```

---

## What a Semantic Build Stage Does

In the CI/CD pipeline, the build stage with semantic versioning performs:

1. **Analyse commits** — read commit messages since the last tag to determine the bump type
2. **Bump version** — update version in `pyproject.toml`, `setup.cfg`, `__version__`, or equivalent
3. **Tag the commit** — create a Git tag (`v1.2.0`)
4. **Generate changelog** — auto-generate `CHANGELOG.md` from commit messages
5. **Build the artefact** — package the code (wheel, container image, zip)
6. **Label the artefact** — tag with the semantic version and Git SHA for traceability
7. **Publish** — push to the appropriate registry (PyPI, container registry, cloud storage)

```
commit history → version bump → git tag → build artefact → publish
```

---

## Versioning Strategy for ML

### Application + Model versioning

Track both independently but link them:

```
Application: v2.1.0
Model:       model-v15 (trained with app v2.1.0, data snapshot 2024-12-01)
```

The model registry entry should record:
- Model version
- Application version it was trained with
- Data snapshot identifier
- Training config hash
- Evaluation metrics

### When to bump what

| Change | Application version | Model version |
|--------|-------------------|---------------|
| Fix a bug in preprocessing | PATCH bump | Retrain recommended |
| Add a new API field | MINOR bump | No change |
| Change model input schema | MAJOR bump | New model version |
| Retrain on new data (same code) | No change | New model version |
| New model architecture | MINOR or MAJOR bump | New model version |

---

## Best Practices

- **Automate** — never manually edit version numbers; let CI derive them from commits
- **Immutable artefacts** — once a version is published, it is never overwritten
- **Traceability** — every deployed artefact links back to a Git tag, commit SHA, and CI job
- **Changelog** — auto-generated from commits; reviewable in merge requests
- **Branch strategy** — only bump versions on the main/release branch, not on feature branches
- **Lock dependencies** — pin exact versions in lockfiles so builds are reproducible across time
