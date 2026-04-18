# Kaniko Docker Build

Kaniko is a tool for building Docker images inside containers or CI environments **without requiring a Docker daemon**. This makes it the standard choice for building images in CI/CD pipelines where Docker-in-Docker is unavailable or a security concern.

---

## Why Kaniko?

| Concern | Docker-in-Docker | Kaniko |
|---------|-----------------|--------|
| Requires Docker daemon | ✅ Yes — needs privileged mode | ❌ No — runs in userspace |
| Security risk | High — privileged containers can escape isolation | Low — no elevated privileges needed |
| CI compatibility | Limited — not all CI runners support privileged mode | Universal — runs as a regular container |
| Layer caching | Local daemon cache only | Supports remote cache (registry-based) |
| Kubernetes-native | Requires DinD sidecar | Runs as a standard pod |

Kaniko is the recommended approach for building images in GitLab CI, GitHub Actions, and Kubernetes-based CI systems.

---

## How Kaniko Works

```
Dockerfile + build context → Kaniko executor → image layers → push to registry
```

1. Kaniko reads the Dockerfile and build context (source files)
2. Executes each Dockerfile instruction in userspace (no daemon)
3. Creates image layers and snapshots the filesystem after each step
4. Pushes the final image directly to a container registry

---

## Basic Usage

### GitLab CI

```yaml
build_image:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.23.0-debug
    entrypoint: [""]
  script:
    - >
      /kaniko/executor
      --context "${CI_PROJECT_DIR}/src"
      --dockerfile "${CI_PROJECT_DIR}/src/Dockerfile"
      --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHORT_SHA}"
      --destination "${CI_REGISTRY_IMAGE}:latest"
```

### GitHub Actions

```yaml
- name: Build and push with Kaniko
  uses: aevea/action-kaniko@v0.12.0
  with:
    image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
    tag: ${{ github.sha }}
    path: ./src
    build_file: ./src/Dockerfile
```

### Kubernetes Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kaniko-build
spec:
  template:
    spec:
      containers:
        - name: kaniko
          image: gcr.io/kaniko-project/executor:v1.23.0
          args:
            - --context=s3://my-bucket/build-context.tar.gz
            - --dockerfile=Dockerfile
            - --destination=registry.example.com/model-serve:v1.2.0
          volumeMounts:
            - name: docker-config
              mountPath: /kaniko/.docker
      volumes:
        - name: docker-config
          secret:
            secretName: registry-credentials
      restartPolicy: Never
```

---

## Secret Handling

Kaniko supports `--mount=type=secret` in Dockerfiles (BuildKit syntax). Pass secrets securely without baking them into image layers:

```dockerfile
RUN --mount=type=secret,id=git_token \
    GIT_TOKEN=$(cat /run/secrets/git_token) && \
    pip install git+https://${GIT_TOKEN}@github.com/org/package.git && \
    unset GIT_TOKEN
```

In GitLab CI, mount the secret via a file:

```yaml
build_image:
  script:
    - echo "${GIT_TOKEN}" > /kaniko/git_token
    - >
      /kaniko/executor
      --context "${CI_PROJECT_DIR}"
      --dockerfile Dockerfile
      --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHORT_SHA}"
      --build-arg GIT_TOKEN_FILE=/kaniko/git_token
```

---

## Layer Caching

Kaniko does not have a local daemon cache, so builds can be slow without remote caching. Use registry-based caching to speed up rebuilds:

```yaml
/kaniko/executor
  --context .
  --dockerfile Dockerfile
  --destination registry.example.com/model-serve:v1.2.0
  --cache=true
  --cache-repo=registry.example.com/model-serve/cache
```

| Flag | Purpose |
|------|---------|
| `--cache=true` | Enable layer caching |
| `--cache-repo` | Registry path to store/retrieve cached layers |
| `--cache-ttl` | Cache expiry duration (default: 336h / 14 days) |

### How it works

1. Before building each layer, Kaniko checks the cache repo for a matching layer digest
2. If found, it pulls the cached layer instead of rebuilding
3. After building, new layers are pushed to the cache repo

This is especially important for ML images where `pip install` layers (PyTorch, TensorFlow, XGBoost) can take minutes to rebuild.

---

## Build Args

Pass build-time variables for environment-specific builds:

```yaml
/kaniko/executor
  --context .
  --dockerfile Dockerfile
  --destination registry.example.com/model-serve:v1.2.0
  --build-arg RUN_ENVIRONMENT=prod
  --build-arg BASE_IMAGE=python:3.12-slim
```

```dockerfile
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG RUN_ENVIRONMENT=dev
ENV RUN_ENVIRONMENT=${RUN_ENVIRONMENT}
```

---

## Multi-Destination Tagging

Push the same image with multiple tags in a single build:

```yaml
/kaniko/executor
  --context .
  --dockerfile Dockerfile
  --destination registry.example.com/model-serve:${CI_COMMIT_SHORT_SHA}
  --destination registry.example.com/model-serve:1.2.0
  --destination registry.example.com/model-serve:latest
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `error building image: could not parse reference` | Invalid image name or tag | Check registry URL and tag format |
| `unauthorized: authentication required` | Missing or expired registry credentials | Mount Docker config with valid credentials |
| `error resolving dockerfile` | Wrong `--dockerfile` path | Use absolute path relative to `--context` |
| Slow builds | No layer caching | Enable `--cache=true` with `--cache-repo` |
| Large build context | `.git/`, `data/`, `notebooks/` included | Add `.dockerignore` to the context root |

---

## Best Practices

- Always use a pinned Kaniko version (`v1.23.0`), not `latest`
- Enable `--cache=true` with a `--cache-repo` — uncached ML image builds are painfully slow
- Use `--context` to limit the build context to only what the Dockerfile needs
- Add `.dockerignore` in the context root to exclude data files, notebooks, and `.git/`
- Use the `-debug` image variant (`executor:v1.23.0-debug`) in CI — it includes a shell for troubleshooting
- Tag images with both Git SHA and semantic version for traceability
- Store registry credentials as CI secrets, mounted into the Kaniko container — never hardcode
