# AI Observability & LLM Footprint

## Overview

Observability for LLM applications means tracking every interaction — prompts, completions, latency, token usage, costs, errors, and quality scores — so you can debug, optimise, and monitor production systems.

## Why It Matters

- Debug why a specific response was bad (trace the full chain)
- Track cost per user/feature/query type
- Detect quality degradation over time
- Identify slow steps in multi-step agent workflows
- A/B test prompts and models with real usage data
- Compliance and audit trails

---

## Langfuse (Open Source, Self-Hostable)

The most popular open-source LLM observability platform. Tracks traces, generations, scores, and costs.

### Install & Setup

```bash
pip install langfuse
```

```python
import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"  # or self-hosted URL
```

### Basic Tracing (Low-Level)

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create a trace (represents one user interaction)
trace = langfuse.trace(
    name="user-query",
    user_id="user_123",
    metadata={"source": "chat", "model": "gemini-2.0-flash"},
)

# Log a generation (LLM call)
generation = trace.generation(
    name="answer-generation",
    model="gemini-2.0-flash",
    input=[{"role": "user", "content": "How do I reset my password?"}],
    output="To reset your password, go to Settings > Security...",
    usage={"input": 150, "output": 89, "total": 239},
    metadata={"temperature": 0.7},
)

# Log a span (retrieval, processing, etc.)
span = trace.span(
    name="vector-search",
    input={"query": "password reset"},
    output={"n_results": 5, "top_score": 0.87},
)

# Score the trace (quality feedback)
trace.score(name="relevance", value=0.9, comment="Good answer")
trace.score(name="faithfulness", value=1.0)

# Flush (important at end of request)
langfuse.flush()
```

### LangChain Integration (Automatic Tracing)

```python
from langfuse.callback import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Langfuse callback handler — traces everything automatically
langfuse_handler = CallbackHandler(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com",
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | llm

# Pass handler — all chain steps are traced
response = chain.invoke(
    {"question": "What is gradient boosting?"},
    config={"callbacks": [langfuse_handler]},
)
```

### Decorator-Based Tracing

```python
from langfuse.decorators import observe, langfuse_context

@observe()
def answer_question(question: str) -> str:
    # Retrieval step (auto-traced)
    context = retrieve_context(question)

    # Generation step (auto-traced)
    response = generate_answer(question, context)

    # Add score
    langfuse_context.update_current_observation(
        metadata={"n_chunks": len(context)},
    )
    langfuse_context.score_current_trace(name="user_feedback", value=1.0)

    return response

@observe()
def retrieve_context(query: str) -> list[str]:
    results = vectorstore.similarity_search(query, k=5)
    return [doc.page_content for doc in results]

@observe()
def generate_answer(question: str, context: list[str]) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content
```

### Prompt Management with Langfuse

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Fetch versioned prompt from Langfuse
prompt = langfuse.get_prompt("support-qa", version=2)

# Use in your chain
compiled = prompt.compile(question="What features matter?", context="...")
response = llm.invoke(compiled)
```

### Evaluation Datasets & Scoring

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create a dataset for evaluation
dataset = langfuse.create_dataset(name="qa-eval-v1")

# Add items
langfuse.create_dataset_item(
    dataset_name="qa-eval-v1",
    input={"question": "How do I reset my password?"},
    expected_output="Navigate to Settings, click Security, use Reset Password",
)

# Run evaluation
dataset = langfuse.get_dataset("qa-eval-v1")
for item in dataset.items:
    # Run your system
    response = chain.invoke(item.input)

    # Link run to dataset item
    item.link(
        trace_id=langfuse_handler.get_trace_id(),
        run_name="experiment-v2",
    )
```

---

## LangSmith (LangChain Native)

Built by LangChain team — deep integration with LangChain/LangGraph.

### Setup

```bash
pip install langsmith
```

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "support-assistant"
```

### Automatic Tracing (Zero Config)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# With LANGCHAIN_TRACING_V2=true, all chains are auto-traced
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | llm

# This is automatically traced to LangSmith
response = chain.invoke({"question": "How does password recovery work?"})
```

### Custom Runs & Feedback

```python
from langsmith import Client

client = Client()

# Log feedback on a specific run
client.create_feedback(
    run_id="run-uuid-here",
    key="correctness",
    score=1.0,
    comment="Accurate answer",
)

# Create evaluation dataset
dataset = client.create_dataset("support-eval")
client.create_examples(
    inputs=[{"question": "What is two-factor authentication?"}],
    outputs=[{"answer": "A security method requiring two forms of verification."}],
    dataset_id=dataset.id,
)
```

### Running Evaluations

```python
from langsmith import evaluate

def predict(inputs: dict) -> dict:
    response = chain.invoke(inputs)
    return {"answer": response.content}

def correctness(run, example) -> dict:
    prediction = run.outputs["answer"]
    reference = example.outputs["answer"]
    # Simple check — use LLM judge for production
    score = 1.0 if reference.lower() in prediction.lower() else 0.0
    return {"key": "correctness", "score": score}

results = evaluate(
    predict,
    data="support-eval",
    evaluators=[correctness],
    experiment_prefix="v2-gemini-flash",
)
```

---

## OpenTelemetry for LLM (OpenLLMetry)

Open standard instrumentation — vendor-neutral.

```bash
pip install opentelemetry-instrumentation-langchain traceloop-sdk
```

```python
from traceloop.sdk import Traceloop

# Initialize (sends to any OTLP-compatible backend)
Traceloop.init(
    app_name="support-assistant",
    api_endpoint="http://localhost:4318",  # Jaeger, Grafana Tempo, etc.
)

# All LangChain calls are auto-instrumented
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
response = llm.invoke("What is caching?")  # Automatically traced
```

### Custom Spans

```python
from traceloop.sdk.decorators import workflow, task

@workflow(name="support_query")
def handle_query(question: str) -> str:
    context = retrieve(question)
    answer = generate(question, context)
    return answer

@task(name="retrieval")
def retrieve(query: str) -> list[str]:
    return vectorstore.similarity_search(query, k=5)

@task(name="generation")
def generate(question: str, context: list[str]) -> str:
    return llm.invoke(f"{context}\n\n{question}").content
```

---

## Phoenix (Arize — Local Observability)

Run locally, no cloud dependency. Good for development and debugging.

```bash
pip install arize-phoenix
```

```python
import phoenix as px

# Launch local UI
session = px.launch_app()
print(f"Phoenix UI: {session.url}")  # http://localhost:6006

# Instrument LangChain
from phoenix.otel import register
tracer_provider = register(project_name="support-assistant")

from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Now all LangChain calls appear in Phoenix UI
response = chain.invoke({"question": "What features matter?"})
```

---

## Key Metrics to Track

### Cost & Usage

```python
from dataclasses import dataclass

@dataclass
class LLMMetrics:
    total_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost per call."""
    pricing = {
        "gemini-2.0-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "claude-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    }
    rates = pricing.get(model, {"input": 0, "output": 0})
    return input_tokens * rates["input"] + output_tokens * rates["output"]
```

### Dashboard Metrics

| Metric | What to Track | Alert When |
|--------|--------------|------------|
| Latency (p50, p95, p99) | Response time distribution | p95 > 5s |
| Token usage per request | Input + output tokens | Avg > 2x baseline |
| Cost per user/day | Cumulative spend | Daily cost spike |
| Error rate | Failed LLM calls | > 1% |
| Hallucination rate | Scored faithfulness | < 0.8 average |
| Retrieval hit rate | Queries with relevant context | < 90% |
| User feedback | Thumbs up/down ratio | Negative > 20% |
| Cache hit rate | Semantic cache utilisation | < expected |

---

## Custom Logging (Lightweight)

For projects that don't need a full platform:

```python
import json
import time
from datetime import datetime
from pathlib import Path

class LLMLogger:
    def __init__(self, log_path: str = "./llm_logs.jsonl"):
        self.log_path = Path(log_path)

    def log(self, trace_id: str, **kwargs):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace_id,
            **kwargs,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_llm_call(self, trace_id: str, model: str, input_text: str, output_text: str,
                     input_tokens: int, output_tokens: int, latency_ms: float):
        self.log(
            trace_id=trace_id,
            type="generation",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            input_preview=input_text[:200],
            output_preview=output_text[:200],
        )

logger = LLMLogger()

# Usage
start = time.perf_counter()
response = llm.invoke(prompt)
latency = (time.perf_counter() - start) * 1000

logger.log_llm_call(
    trace_id="req_001",
    model="gemini-2.0-flash",
    input_text=prompt,
    output_text=response.content,
    input_tokens=response.usage_metadata["input_tokens"],
    output_tokens=response.usage_metadata["output_tokens"],
    latency_ms=latency,
)
```

---

## Platform Comparison

| Platform | Type | Self-Host | LangChain Integration | Cost |
|----------|------|-----------|----------------------|------|
| Langfuse | Open source | Yes | Callback handler | Free (self-host) / paid cloud |
| LangSmith | Managed | No | Native (auto) | Free tier + paid |
| Phoenix (Arize) | Local/open | Yes | OpenInference | Free |
| OpenLLMetry | Open standard | Yes (OTLP) | Auto-instrumentation | Free |
| Weights & Biases | Managed | No | W&B Tracer | Free tier + paid |
| Helicone | Proxy-based | No | Proxy header | Free tier + paid |

## When to Use What

| Scenario | Recommended |
|----------|-------------|
| Development & debugging | Phoenix (local, instant) |
| Production monitoring | Langfuse (self-host) or LangSmith |
| Cost tracking | Langfuse or Helicone |
| Vendor-neutral | OpenTelemetry + OpenLLMetry |
| LangChain/LangGraph heavy | LangSmith (deepest integration) |
| Privacy-sensitive (on-prem) | Langfuse self-hosted or Phoenix |
| Quick prototype | LangSmith (zero config with env var) |

## Best Practices

- Trace everything in development, sample in production (10-100% based on volume)
- Track cost per query type — identifies expensive patterns
- Log retrieval context alongside generations — enables faithfulness debugging
- Score traces (automated + user feedback) to track quality over time
- Set up alerts on latency spikes, error rates, and cost anomalies
- Use evaluation datasets to regression-test prompt changes before deployment
- Tag traces with metadata (user_id, feature, experiment) for slicing
- Review low-scored traces regularly — they reveal systematic failures
- Keep prompt versions tracked — enables rollback and A/B testing
