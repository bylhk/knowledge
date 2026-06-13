# Cost Management & Token Economics

## Overview

LLM costs scale with usage — every token processed costs money. Without active management, costs can spiral unexpectedly. This card covers strategies for tracking, reducing, and optimising LLM spend across development and production.

## Cost Anatomy

```
Total Cost = (Input Tokens × Input Rate) + (Output Tokens × Output Rate)
```

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Context Window |
|-------|----------------------|------------------------|----------------|
| GPT-4o | $2.50 | $10.00 | 128k |
| GPT-4o-mini | $0.15 | $0.60 | 128k |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 200k |
| Claude 3.5 Haiku | $0.80 | $4.00 | 200k |
| Gemini 2.0 Flash | $0.075 | $0.30 | 1M |
| Gemini 2.0 Flash Lite | $0.02 | $0.08 | 1M |

---

## Strategy 1: Model Routing (Cheapest Model That Works)

Route queries to the cheapest model capable of handling them.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

class QueryComplexity(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    reasoning: str

# Cheap classifier to route queries
classifier = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
complexity_checker = classifier.with_structured_output(QueryComplexity)

# Model pool
MODELS = {
    "simple": ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite"),    # $0.02/$0.08
    "moderate": ChatGoogleGenerativeAI(model="gemini-2.0-flash"),       # $0.075/$0.30
    "complex": ChatGoogleGenerativeAI(model="gemini-2.5-pro"),          # Higher cost
}

def route_query(query: str) -> str:
    """Route to cheapest sufficient model."""
    assessment = complexity_checker.invoke(
        f"Rate complexity (simple/moderate/complex):\n{query}"
    )
    model = MODELS[assessment.complexity]
    response = model.invoke(query)
    return response.content

# Cost savings: 60-80% of queries are "simple" → use cheapest model
```

### Rule-Based Routing (No Classifier Cost)

```python
def route_by_heuristic(query: str) -> str:
    """Route based on query characteristics — zero classifier cost."""
    query_lower = query.lower()

    # Simple: short, factual, lookup-style
    if len(query) < 100 and not any(w in query_lower for w in ["explain", "compare", "analyse", "why"]):
        return "simple"

    # Complex: long, reasoning-heavy, multi-part
    if len(query) > 500 or query.count("?") > 2 or any(w in query_lower for w in ["step by step", "trade-offs", "design"]):
        return "complex"

    return "moderate"
```

---

## Strategy 2: Caching

### Exact Cache (Redis)

```python
import hashlib
import json
import redis

r = redis.Redis(host="localhost", port=6379)

def cached_llm_call(prompt: str, model: str, ttl: int = 3600) -> str:
    """Cache exact prompt matches."""
    cache_key = f"llm:{model}:{hashlib.sha256(prompt.encode()).hexdigest()}"

    # Check cache
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)["response"]

    # Call LLM
    response = llm.invoke(prompt)

    # Store in cache
    r.setex(cache_key, ttl, json.dumps({"response": response.content}))
    return response.content
```

### Semantic Cache

```python
from langchain_community.cache import RedisSemanticCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import langchain

# Cache semantically similar queries (not just exact matches)
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    score_threshold=0.95,  # How similar queries must be to hit cache
)

# "What is machine learning?" → cache miss (first call)
# "Can you explain what ML is?" → cache HIT (semantically similar)
```

### Cache Hit Rate Monitoring

```python
from dataclasses import dataclass, field

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    estimated_savings: float = 0.0  # USD saved

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def record_hit(self, estimated_tokens: int, model_rate: float):
        self.hits += 1
        self.estimated_savings += estimated_tokens * model_rate / 1_000_000

    def record_miss(self):
        self.misses += 1

    def report(self):
        print(f"Cache hit rate: {self.hit_rate:.1%}")
        print(f"Estimated savings: ${self.estimated_savings:.2f}")
```

---

## Strategy 3: Token Reduction

### Prompt Compression

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device_map="cpu",
)

def compress_context(context: list[str], question: str, rate: float = 0.5) -> str:
    """Compress RAG context to reduce tokens."""
    result = compressor.compress_prompt(
        context=context,
        instruction="Answer based on the context.",
        question=question,
        rate=rate,  # Keep 50% of tokens
    )
    print(f"Compressed: {result['origin_tokens']} → {result['compressed_tokens']} tokens "
          f"({result['ratio']:.1f}x reduction)")
    return result["compressed_prompt"]
```

### Efficient Prompt Design

```python
# BAD: Verbose system prompt (wastes tokens every call)
bad_prompt = """
You are a highly knowledgeable and experienced senior software engineer 
with deep expertise in Python programming, distributed systems architecture, 
cloud-native development practices, and modern software engineering principles. 
Your role is to provide comprehensive, detailed, and actionable advice to 
developers who seek your guidance on technical matters. When responding, 
you should consider best practices, performance implications, security 
considerations, and maintainability aspects of any solution you propose.
"""  # ~80 tokens

# GOOD: Concise equivalent
good_prompt = """Senior Python/distributed systems engineer. 
Be concise. Consider: performance, security, maintainability."""  # ~20 tokens

# Savings: 60 tokens × $2.50/1M × 10,000 requests/day = $1.50/day
```

### Context Window Management

```python
def budget_context(
    query: str,
    retrieved_docs: list[str],
    max_context_tokens: int = 3000,
    max_output_tokens: int = 1000,
) -> list[str]:
    """Fit retrieved documents within token budget."""
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")

    system_tokens = 100  # Estimated system prompt size
    query_tokens = len(enc.encode(query))
    budget = max_context_tokens - system_tokens - query_tokens

    selected = []
    used_tokens = 0

    for doc in retrieved_docs:
        doc_tokens = len(enc.encode(doc))
        if used_tokens + doc_tokens <= budget:
            selected.append(doc)
            used_tokens += doc_tokens
        else:
            # Truncate last doc to fit
            remaining = budget - used_tokens
            if remaining > 50:  # Only include if meaningful
                truncated = enc.decode(enc.encode(doc)[:remaining])
                selected.append(truncated)
            break

    return selected
```

---

## Strategy 4: Batch Processing

Process multiple requests together for efficiency.

```python
from langchain_core.runnables import RunnableParallel

async def batch_process(queries: list[str], batch_size: int = 10) -> list[str]:
    """Process queries in parallel batches."""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        # Parallel execution within batch
        batch_results = await llm.abatch(batch, config={"max_concurrency": batch_size})
        results.extend([r.content for r in batch_results])

    return results

# Batch processing reduces overhead vs sequential calls
# Also enables better rate limit utilisation
```

---

## Strategy 5: Cost Tracking & Budgets

### Per-Request Cost Tracking

```python
from dataclasses import dataclass
from datetime import datetime, date
from collections import defaultdict

@dataclass
class UsageRecord:
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: str | None = None
    feature: str | None = None

class CostTracker:
    PRICING = {
        "gemini-2.0-flash": {"input": 0.075e-6, "output": 0.30e-6},
        "gemini-2.0-flash-lite": {"input": 0.02e-6, "output": 0.08e-6},
        "gpt-4o": {"input": 2.50e-6, "output": 10.0e-6},
        "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
    }

    def __init__(self, daily_budget: float = 50.0):
        self.records: list[UsageRecord] = []
        self.daily_budget = daily_budget

    def record(self, model: str, input_tokens: int, output_tokens: int,
               user_id: str = None, feature: str = None):
        rates = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = input_tokens * rates["input"] + output_tokens * rates["output"]

        self.records.append(UsageRecord(
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            user_id=user_id,
            feature=feature,
        ))
        return cost

    def daily_spend(self, day: date = None) -> float:
        day = day or date.today()
        return sum(r.cost_usd for r in self.records if r.timestamp.date() == day)

    def is_over_budget(self) -> bool:
        return self.daily_spend() >= self.daily_budget

    def spend_by_feature(self) -> dict[str, float]:
        by_feature = defaultdict(float)
        for r in self.records:
            by_feature[r.feature or "unknown"] += r.cost_usd
        return dict(by_feature)

    def spend_by_model(self) -> dict[str, float]:
        by_model = defaultdict(float)
        for r in self.records:
            by_model[r.model] += r.cost_usd
        return dict(by_model)

    def report(self):
        print(f"Daily spend: ${self.daily_spend():.4f} / ${self.daily_budget:.2f}")
        print(f"By model: {self.spend_by_model()}")
        print(f"By feature: {self.spend_by_feature()}")
```

### Budget Enforcement

```python
class BudgetGuard:
    """Prevent exceeding cost limits."""

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def check_budget(self, estimated_tokens: int, model: str) -> bool:
        """Pre-flight check before making an LLM call."""
        rates = self.tracker.PRICING.get(model, {"input": 0, "output": 0})
        estimated_cost = estimated_tokens * (rates["input"] + rates["output"]) / 2

        remaining = self.tracker.daily_budget - self.tracker.daily_spend()

        if estimated_cost > remaining:
            raise BudgetExceededError(
                f"Estimated cost ${estimated_cost:.4f} exceeds remaining budget ${remaining:.4f}"
            )
        return True

    def fallback_model(self, preferred: str) -> str:
        """Switch to cheaper model if budget is tight."""
        remaining = self.tracker.daily_budget - self.tracker.daily_spend()

        if remaining < self.tracker.daily_budget * 0.2:  # Under 20% budget remaining
            return "gemini-2.0-flash-lite"  # Cheapest option
        return preferred
```

---

## Strategy 6: Development vs Production Cost Control

```python
import os

class CostConfig:
    """Different cost strategies for different environments."""

    @staticmethod
    def get_config():
        env = os.getenv("ENVIRONMENT", "development")

        if env == "development":
            return {
                "default_model": "gemini-2.0-flash-lite",  # Cheapest
                "max_tokens": 500,
                "cache_enabled": True,
                "cache_ttl": 86400,  # 24h (aggressive caching for dev)
                "daily_budget": 5.0,
            }
        elif env == "staging":
            return {
                "default_model": "gemini-2.0-flash",
                "max_tokens": 1000,
                "cache_enabled": True,
                "cache_ttl": 3600,
                "daily_budget": 20.0,
            }
        else:  # production
            return {
                "default_model": "gemini-2.0-flash",
                "max_tokens": 2000,
                "cache_enabled": True,
                "cache_ttl": 300,  # 5min (fresher responses)
                "daily_budget": 100.0,
            }
```

---

## Cost Optimisation Quick Wins

| Technique | Savings | Effort | Risk |
|-----------|---------|--------|------|
| Switch to cheaper model | 50-95% | Low | Quality may drop |
| Exact response caching | 30-70% | Low | Stale responses |
| Semantic caching | 20-50% | Medium | Incorrect cache hits |
| Shorter system prompts | 5-20% | Low | None |
| Reduce retrieved context (top-3 vs top-5) | 10-30% | Low | Recall may drop |
| Prompt compression (LLMLingua) | 30-50% | Medium | Slight quality drop |
| Model routing | 40-70% | Medium | Routing errors |
| Batch processing | Throughput gain | Low | Added latency |
| Output length limits | 10-30% | Low | Truncation risk |
| Self-hosted models | 70-90% long-term | High | Infra management |

## Monitoring Dashboard Metrics

| Metric | Track | Alert When |
|--------|-------|------------|
| Daily spend (USD) | Running total per day | > 80% of budget |
| Cost per request | Average and P95 | > 2x normal |
| Tokens per request | Input + output | Sudden spike |
| Cache hit rate | Hits / total | < 30% (expected higher) |
| Cost by user | Per user-id | Single user > 10% of budget |
| Cost by feature | Per endpoint/feature | Unexpected feature dominates |
| Model distribution | % traffic per model | Routing drift |

## Best Practices

- Set daily/monthly budgets with hard caps and alerts at 80%
- Track cost per feature/user — identifies expensive patterns
- Cache aggressively in development (day-level TTL)
- Use the cheapest model that meets quality requirements
- Compress RAG context before injecting into prompts
- Keep system prompts short — they're repeated every single call
- Route simple queries to small models automatically
- Monitor cost trends — gradual increases indicate prompt/usage drift
- Consider self-hosting for high-volume, predictable workloads
- Log token counts alongside responses for cost attribution
