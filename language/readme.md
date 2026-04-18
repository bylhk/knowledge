# Choosing the Right Language for Each Module

ML projects span data processing, model training, serving, infrastructure, and orchestration. No single language is optimal for all of these — picking the right language per module reduces complexity, improves performance, and leverages each ecosystem's strengths.

---

## Language Speed Comparison

![Language Speed Comparison](language_speed.gif)

---

## Language-to-Module Mapping

| Module | Recommended language | Why |
|--------|---------------------|-----|
| Data processing & feature engineering | SQL | Runs where the data lives — no data movement, optimised by the query engine |
| Model training & experimentation | Python | Richest ML ecosystem (scikit-learn, XGBoost, PyTorch, TensorFlow) |
| Model serving (lightweight) | Python | FastAPI/Flask, quick to build, matches training code |
| Model serving (high-throughput) | Go / Rust / C++ | Lower latency, smaller memory footprint, better concurrency |
| Infrastructure provisioning | HCL (Terraform) / YAML (CloudFormation) | Declarative IaC, cloud-native tooling, drift detection |
| CI/CD pipelines | YAML + Bash | Native to CI platforms (GitLab CI, GitHub Actions), simple glue logic |
| CLI tools & scripts | Python / Bash | Python for complex logic, Bash for simple glue and file operations |
| Frontend / dashboards | JavaScript / TypeScript | Browser-native, rich visualisation libraries (React, D3.js) |
| Mobile / edge inference | Swift / Kotlin / C++ | Platform-native performance, ONNX/TFLite runtime support |

---

## Language Breakdown

### SQL

Best for: data processing, feature engineering, aggregation, data validation.

| Pros | Cons |
|------|------|
| Runs where the data lives (BigQuery, Redshift, Snowflake) — no data transfer overhead | Limited expressiveness for complex logic (loops, custom algorithms) |
| Query engine optimises execution (parallelism, partitioning, caching) automatically | Hard to unit test compared to Python |
| Declarative — describe *what* you want, not *how* to compute it | Dialect differences across platforms (BigQuery vs Postgres vs Spark SQL) |
| Handles terabyte-scale data without memory constraints | Version control and code review of SQL can be awkward |
| Widely understood across data, analytics, and engineering teams | No package ecosystem — logic must be self-contained or use UDFs |

Use SQL when: the data is already in a warehouse, the logic is aggregation/filtering/joining, and the scale exceeds what fits in memory.

### Python

Best for: model training, experimentation, serving, scripting, orchestration.

| Pros | Cons |
|------|------|
| Dominant ML ecosystem — scikit-learn, XGBoost, PyTorch, TensorFlow, Hugging Face | Slow for CPU-bound computation without NumPy/C extensions |
| Rapid prototyping — notebooks, REPL, dynamic typing | GIL limits true multi-threading for CPU tasks |
| Huge package ecosystem for every ML task | Dependency management can be painful (version conflicts) |
| Same language for training and serving — no translation layer | Higher memory footprint than compiled languages |
| Strong community, documentation, and hiring pool | Type safety is opt-in (mypy), not enforced |

Use Python when: you need ML libraries, rapid iteration, or the team is Python-native. Avoid for latency-critical hot paths where compiled languages excel.

### Go

Best for: high-throughput model serving, microservices, CLI tools.

| Pros | Cons |
|------|------|
| Fast compilation, single static binary — simple deployment | Small ML ecosystem — no native training libraries |
| Excellent concurrency model (goroutines) — handles thousands of concurrent requests | Verbose error handling |
| Low memory footprint and predictable latency | Interop with Python ML models requires ONNX/gRPC bridge |
| Strong standard library for HTTP, JSON, networking | Less familiar to data science teams |

Use Go when: serving pre-trained models (via ONNX Runtime) at high throughput with strict latency requirements.

### Rust

Best for: performance-critical inference, edge deployment, data pipelines.

| Pros | Cons |
|------|------|
| Near-C performance with memory safety guarantees | Steep learning curve |
| No garbage collector — predictable latency | Slower development velocity than Python or Go |
| Growing ML ecosystem (candle, burn, ort) | Smaller community for ML-specific tasks |
| Excellent for WASM targets (edge/browser inference) | Compile times can be long |

Use Rust when: you need maximum performance and memory safety — edge inference, high-frequency serving, or data-intensive pipelines (Polars, DataFusion).

### C++

Best for: inference engines, custom operators, embedded/edge deployment.

| Pros | Cons |
|------|------|
| Maximum performance — most inference runtimes are written in C++ | Complex build systems, manual memory management |
| Direct access to hardware (GPU, TPU, custom accelerators) | Slow development cycle |
| Required for custom TensorFlow/PyTorch operators | Hard to hire for compared to Python |
| ONNX Runtime, TensorRT, TFLite all have C++ APIs | Security risks from memory bugs |

Use C++ when: you are writing custom inference kernels, optimising for specific hardware, or deploying to resource-constrained devices.

### HCL (Terraform) / YAML (CloudFormation, Pulumi YAML)

Best for: infrastructure provisioning, cloud resource management.

| Pros | Cons |
|------|------|
| Declarative — define desired state, tool handles the diff | Limited logic (conditionals, loops are awkward in HCL) |
| Drift detection — detects manual changes to infrastructure | State management adds complexity (Terraform state files) |
| Cloud-native — first-class support for AWS, GCP, Azure resources | Learning curve for each IaC tool |
| Reproducible environments — same config produces same infrastructure | Debugging plan/apply failures can be opaque |

Use HCL/YAML when: provisioning cloud resources (compute, networking, storage, IAM). Never manage infrastructure with ad-hoc scripts.

### Bash

Best for: simple glue scripts, CI/CD steps, file operations.

| Pros | Cons |
|------|------|
| Available everywhere — no installation needed | Fragile — poor error handling, no type safety |
| Fast for simple tasks (file moves, env setup, curl calls) | Unreadable beyond ~50 lines |
| Native to CI/CD runners | Hard to test and debug |
| Good for chaining existing CLI tools | Cross-platform inconsistencies (macOS vs Linux) |

Use Bash when: the task is simple glue logic (< 50 lines). Switch to Python for anything with conditionals, loops, or error handling.

### JavaScript / TypeScript

Best for: dashboards, monitoring UIs, lightweight API gateways.

| Pros | Cons |
|------|------|
| Browser-native — required for web frontends | Not suited for numerical computation |
| Rich visualisation ecosystem (D3.js, Plotly.js, React) | ML ecosystem is immature compared to Python |
| TypeScript adds type safety | Context switching for ML teams who primarily use Python |
| Node.js can serve lightweight APIs | Dependency bloat (node_modules) |

Use JS/TS when: building monitoring dashboards, model result visualisations, or web-based annotation tools.

---

## Decision Framework

When choosing a language for a new module, ask:

1. **Where does the data live?** → If in a warehouse, use SQL. Don't move data unnecessarily.
2. **Does it need ML libraries?** → Python. No other language matches the ecosystem.
3. **Is latency critical?** → Go or Rust for serving. Python adds overhead.
4. **Is it infrastructure?** → HCL/Terraform or CloudFormation YAML. Keep infra declarative.
5. **Is it glue logic under 50 lines?** → Bash. Over 50 lines → Python.
6. **Does the team know it?** → Prefer languages the team already uses. A fast language nobody can maintain is worse than a slower one everyone understands.

```
Data in warehouse?  ──yes──→  SQL
        │ no
Need ML libraries?  ──yes──→  Python
        │ no
Latency critical?   ──yes──→  Go / Rust / C++
        │ no
Infrastructure?     ──yes──→  HCL / YAML
        │ no
Simple glue?        ──yes──→  Bash
        │ no
Web frontend?       ──yes──→  TypeScript
        │ no
Default             ────────→  Python
```

---

## Anti-Patterns

- **Python for terabyte-scale data processing** — use SQL in the warehouse or Spark; don't pull data into memory
- **Bash for complex logic** — if it has functions, arrays, or error handling, rewrite in Python
- **SQL for ML model logic** — UDFs exist but are painful to maintain and test; keep ML in Python
- **Go/Rust for prototyping** — slower iteration; prototype in Python first, then port the hot path if needed
- **Multiple languages for the same concern** — don't split model serving across Python and Go unless there is a clear performance boundary
- **Choosing a language for its novelty** — team familiarity and hiring pool matter more than benchmarks for most ML projects
