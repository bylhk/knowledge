# Retrieval Performance Evaluation

## Overview

Evaluating retrieval quality is essential before optimising your RAG pipeline. Poor retrieval = poor generation, regardless of how good your LLM is. Measure retrieval independently from generation to isolate issues.

## Where This Fits

```
[Retrieval System] → ⭐ EVALUATION ⭐ → Iterate on: chunking, embedding, retrieval method
```

---

## Core Metrics

### Precision@k

Of the k documents retrieved, how many are actually relevant?

```python
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of retrieved docs are relevant?"""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k

# Example: retrieved 5 docs, 3 are relevant
# Precision@5 = 3/5 = 0.6
```

### Recall@k

Of all relevant documents, how many did we retrieve in the top k?

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """What fraction of all relevant docs did we find?"""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0.0

# Example: 4 relevant docs exist, we found 3 in top 5
# Recall@5 = 3/4 = 0.75
```

### Hit Rate (Success@k)

Did at least one relevant document appear in the top k? Binary yes/no per query.

```python
def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Did we find ANY relevant doc in top k?"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0
```

### Mean Reciprocal Rank (MRR)

How high does the first relevant result appear? Higher = better.

```python
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/position of first relevant result."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0

def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    """Average RR across all queries."""
    rrs = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in queries_results]
    return sum(rrs) / len(rrs) if rrs else 0.0
```

### Normalised Discounted Cumulative Gain (NDCG@k)

Accounts for position and graded relevance (not just binary relevant/irrelevant).

```python
import numpy as np

def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    scores = relevance_scores[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))

def ndcg_at_k(retrieved_ids: list[str], relevance_map: dict[str, float], k: int) -> float:
    """Normalised DCG — compares actual ranking to ideal ranking."""
    # Actual relevance scores in retrieved order
    actual_scores = [relevance_map.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]

    # Ideal: sort all relevant docs by score descending
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]

    actual_dcg = dcg_at_k(actual_scores, k)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# Example with graded relevance: 3=highly relevant, 2=relevant, 1=marginal, 0=irrelevant
relevance_map = {"doc_a": 3, "doc_b": 2, "doc_c": 1}
retrieved = ["doc_x", "doc_a", "doc_c", "doc_b", "doc_y"]
print(ndcg_at_k(retrieved, relevance_map, k=5))
```

### Mean Average Precision (MAP)

Average of precision at each relevant position.

```python
def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Average precision for a single query."""
    hits = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            hits += 1
            sum_precision += hits / i

    return sum_precision / len(relevant_ids) if relevant_ids else 0.0

def mean_average_precision(queries_results: list[tuple[list[str], set[str]]]) -> float:
    aps = [average_precision(retrieved, relevant) for retrieved, relevant in queries_results]
    return sum(aps) / len(aps) if aps else 0.0
```

---

## Building an Evaluation Dataset

### Manual Annotation

```python
# Evaluation dataset structure
eval_dataset = [
    {
        "query": "What machine learning algorithms are used for classification?",
        "relevant_chunk_ids": ["chunk_042", "chunk_043", "chunk_108"],
    },
    {
        "query": "How is cross-validation used in model training?",
        "relevant_chunk_ids": ["chunk_071", "chunk_072"],
    },
    {
        "query": "What hyperparameters were tuned with Hyperopt?",
        "relevant_chunk_ids": ["chunk_089"],
    },
]
```

### Synthetic Evaluation Set (LLM-Generated)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import random

class SyntheticQuery(BaseModel):
    query: str = Field(description="Natural question this chunk answers")
    difficulty: str = Field(description="easy, medium, or hard")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
generator = llm.with_structured_output(SyntheticQuery)

def generate_eval_dataset(chunks: list[dict], n_queries: int = 50) -> list[dict]:
    """Generate synthetic query-relevance pairs from chunks."""
    sample = random.sample(chunks, min(n_queries, len(chunks)))
    eval_data = []

    for chunk in sample:
        result = generator.invoke(
            f"Generate a natural question that this text chunk would answer. "
            f"Vary difficulty.\n\nChunk:\n{chunk['content'][:500]}"
        )
        eval_data.append({
            "query": result.query,
            "relevant_chunk_ids": [chunk["id"]],
            "difficulty": result.difficulty,
        })

    return eval_data
```

---

## Running Evaluation

### Full Evaluation Harness

```python
import numpy as np

def evaluate_retriever(retriever, eval_dataset: list[dict], k: int = 5) -> dict:
    """Evaluate a retriever against a labelled dataset."""
    metrics = {
        "precision_at_k": [],
        "recall_at_k": [],
        "hit_rate": [],
        "mrr": [],
        "ndcg_at_k": [],
    }

    for item in eval_dataset:
        query = item["query"]
        relevant_ids = set(item["relevant_chunk_ids"])

        # Retrieve
        results = retriever.invoke(query)
        retrieved_ids = [doc.metadata.get("id", str(i)) for i, doc in enumerate(results)]

        # Calculate metrics
        metrics["precision_at_k"].append(precision_at_k(retrieved_ids, relevant_ids, k))
        metrics["recall_at_k"].append(recall_at_k(retrieved_ids, relevant_ids, k))
        metrics["hit_rate"].append(hit_rate_at_k(retrieved_ids, relevant_ids, k))
        metrics["mrr"].append(reciprocal_rank(retrieved_ids, relevant_ids))

    # Aggregate
    report = {name: np.mean(values) for name, values in metrics.items()}
    report["n_queries"] = len(eval_dataset)
    report["k"] = k

    print(f"Retrieval Evaluation (k={k}, n={len(eval_dataset)} queries)")
    print(f"  Precision@{k}: {report['precision_at_k']:.3f}")
    print(f"  Recall@{k}:    {report['recall_at_k']:.3f}")
    print(f"  Hit Rate@{k}:  {report['hit_rate']:.3f}")
    print(f"  MRR:           {report['mrr']:.3f}")

    return report
```

### Comparing Retrievers

```python
def compare_retrievers(retrievers: dict[str, any], eval_dataset: list[dict], k: int = 5):
    """Compare multiple retrieval configurations side by side."""
    results = {}

    for name, retriever in retrievers.items():
        print(f"\n--- {name} ---")
        results[name] = evaluate_retriever(retriever, eval_dataset, k)

    # Summary table
    print("\n\n=== COMPARISON ===")
    print(f"{'Method':<30} {'Precision':<12} {'Recall':<12} {'Hit Rate':<12} {'MRR':<12}")
    print("-" * 78)
    for name, metrics in results.items():
        print(
            f"{name:<30} "
            f"{metrics['precision_at_k']:.3f}       "
            f"{metrics['recall_at_k']:.3f}       "
            f"{metrics['hit_rate']:.3f}       "
            f"{metrics['mrr']:.3f}"
        )

    return results

# Usage
retrievers = {
    "simple_top5": vectorstore.as_retriever(search_kwargs={"k": 5}),
    "mmr": vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}),
    "reranked": reranking_retriever,
    "hybrid": ensemble_retriever,
}

compare_retrievers(retrievers, eval_dataset, k=5)
```

---

## Framework-Based Evaluation

### RAGAS (RAG Assessment)

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What algorithms work best for text classification?", "How is model selection optimised?"],
    "answer": ["Algorithms include random forests, gradient boosting...", "Hyperparameter tuning uses Bayesian optimisation..."],
    "contexts": [
        ["chunk about algorithms...", "another relevant chunk..."],
        ["chunk about optimisation..."],
    ],
    "ground_truth": ["The best algorithms are...", "Model selection uses cross-validation..."],
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
)

print(results)
# {'context_precision': 0.85, 'context_recall': 0.78, 'faithfulness': 0.92, ...}
```

### DeepEval

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

test_cases = [
    LLMTestCase(
        input="What algorithms work best for text classification?",
        actual_output="The best approaches are random forests and neural networks.",
        expected_output="Key approaches include random forests, gradient boosting, and neural networks.",
        retrieval_context=["chunk 1 content...", "chunk 2 content..."],
    ),
]

metrics = [
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7),
]

evaluate(test_cases, metrics)
```

---

## LLM-as-Judge for Relevance

When you don't have labelled data, use an LLM to judge relevance.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class RelevanceJudgment(BaseModel):
    is_relevant: bool = Field(description="Is this chunk relevant to the query?")
    relevance_score: float = Field(ge=0, le=1, description="How relevant (0-1)")
    reasoning: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
judge = llm.with_structured_output(RelevanceJudgment)

def judge_relevance(query: str, chunk: str) -> RelevanceJudgment:
    return judge.invoke(
        f"Judge whether this retrieved chunk is relevant to the query.\n\n"
        f"Query: {query}\n\nChunk:\n{chunk}"
    )

def evaluate_with_llm_judge(retriever, queries: list[str], k: int = 5) -> dict:
    """Evaluate retrieval using LLM-as-judge (no ground truth needed)."""
    all_scores = []

    for query in queries:
        results = retriever.invoke(query)
        query_scores = []

        for doc in results[:k]:
            judgment = judge_relevance(query, doc.page_content)
            query_scores.append(judgment.relevance_score)

        all_scores.append(np.mean(query_scores))

    return {
        "avg_relevance": np.mean(all_scores),
        "min_relevance": np.min(all_scores),
        "per_query": all_scores,
    }
```

---

## Metric Interpretation Guide

| Metric | Good | Acceptable | Poor | Meaning |
|--------|------|-----------|------|---------|
| Precision@5 | > 0.8 | 0.5–0.8 | < 0.5 | Most retrieved docs are relevant |
| Recall@5 | > 0.8 | 0.5–0.8 | < 0.5 | Found most of the relevant docs |
| Hit Rate@5 | > 0.95 | 0.8–0.95 | < 0.8 | At least one relevant doc found |
| MRR | > 0.8 | 0.5–0.8 | < 0.5 | First relevant result appears early |
| NDCG@5 | > 0.8 | 0.5–0.8 | < 0.5 | Relevant docs ranked higher |

## What to Improve Based on Metrics

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Low recall, ok precision | Chunking too narrow, query mismatch | Multi-query, HyDE, hybrid search |
| Ok recall, low precision | Too many irrelevant results | Reranking, metadata filtering |
| Low hit rate | Chunks missing, bad embeddings | Check ingestion coverage, try different model |
| Low MRR, ok recall | Relevant docs exist but ranked low | Reranking, better embedding model |
| All metrics low | Fundamental pipeline issue | Re-evaluate chunking + embedding choice |

## Best Practices

- Evaluate retrieval separately from generation — isolate the problem
- Build a labelled eval set of 50-100 queries minimum
- Use synthetic queries for initial eval, manual annotations for production
- Compare retrieval methods on the same eval set before choosing
- Track metrics over time — degradation signals stale content or drift
- Test with hard queries (ambiguous, multi-hop, jargon-heavy)
- LLM-as-judge is a good proxy when you lack ground truth labels
- Report recall@k for RAG (did we find the info?) and precision@k for UX (is the context clean?)
