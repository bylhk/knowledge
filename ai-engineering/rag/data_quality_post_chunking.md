# Data Quality: Post-Chunking Evaluation

## Overview

After chunking but before embedding and indexing, validate that chunks are well-formed, semantically coherent, and suitable for retrieval. Catching issues here avoids polluting your vector store with noise.

## Where This Fits

```
[Document] → [Ingestion] → [Quality] → [Chunking] → ⭐ CHUNK QUALITY ⭐ → [Embedding] → [Vector DB]
```

---

## Chunk-Level Validation

### Basic Quality Filters

```python
import re
from dataclasses import dataclass

@dataclass
class ChunkQualityResult:
    is_valid: bool
    score: float
    issues: list[str]

def validate_chunk(text: str, metadata: dict = None) -> ChunkQualityResult:
    issues = []
    metadata = metadata or {}

    # Length checks
    if len(text.strip()) < 50:
        issues.append("too_short")
    if len(text) > 10000:
        issues.append("too_long")

    # Token estimate (rough: 1 token ≈ 4 chars)
    estimated_tokens = len(text) / 4
    if estimated_tokens < 20:
        issues.append("too_few_tokens")

    # Encoding artefacts
    if "\x00" in text or "â€" in text or "Ã" in text:
        issues.append("encoding_artefacts")

    # Whitespace ratio
    whitespace_ratio = text.count(" ") / max(len(text), 1)
    if whitespace_ratio > 0.6:
        issues.append("excessive_whitespace")

    # Repetition detection
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append("highly_repetitive")

    # Incomplete sentences (starts mid-word or ends abruptly)
    if text and text[0].islower() and not text.startswith(("def ", "class ", "import ")):
        issues.append("starts_mid_sentence")

    # Metadata completeness
    required_keys = ["source"]
    for key in required_keys:
        if key not in metadata:
            issues.append(f"missing_metadata:{key}")

    score = max(0, 1.0 - len(issues) * 0.2)
    return ChunkQualityResult(is_valid=len(issues) == 0, score=score, issues=issues)
```

---

## Deduplication

### Exact Deduplication

```python
import hashlib

def deduplicate_exact(chunks):
    seen = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    print(f"Removed {len(chunks) - len(unique)} exact duplicates")
    return unique
```

### Fuzzy Deduplication (Cosine Similarity)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_fuzzy(chunks, threshold=0.92):
    texts = [c.page_content for c in chunks]
    tfidf = TfidfVectorizer(max_features=5000).fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)

    to_remove = set()
    for i in range(len(chunks)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(chunks)):
            if j not in to_remove and sim_matrix[i, j] > threshold:
                to_remove.add(j)

    unique = [c for i, c in enumerate(chunks) if i not in to_remove]
    print(f"Removed {len(to_remove)} near-duplicates")
    return unique
```

---

## Semantic Coherence Check

### Embedding Variance (Low Information Detection)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def check_chunk_coherence(chunks, low_percentile=5):
    texts = [c.page_content for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Distance from centroid — low distance = generic/boilerplate
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    threshold = np.percentile(distances, low_percentile)
    low_info = [(c, d) for c, d in zip(chunks, distances) if d < threshold]

    print(f"Flagged {len(low_info)} low-information chunks (< {low_percentile}th percentile)")
    return low_info
```

### Self-Containedness Score (LLM)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ChunkAssessment(BaseModel):
    self_contained: bool = Field(description="Can this chunk be understood without surrounding context?")
    coherent: bool = Field(description="Does this chunk cover a single coherent topic?")
    quality_score: float = Field(ge=0, le=1)
    suggestion: str = Field(description="How to improve this chunk, or 'none'")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
assessor = llm.with_structured_output(ChunkAssessment)

def assess_chunk(text: str) -> ChunkAssessment:
    return assessor.invoke(
        f"Evaluate this text chunk for a RAG knowledge base:\n\n{text}"
    )

# Sample assessment (don't assess every chunk — expensive)
import random
sample = random.sample(chunks, min(30, len(chunks)))
results = [assess_chunk(c.page_content) for c in sample]
avg_quality = sum(r.quality_score for r in results) / len(results)
print(f"Average chunk quality: {avg_quality:.2f}")
```

---

## Distribution Analysis

```python
import numpy as np

def chunk_distribution_report(chunks):
    lengths = [len(c.page_content) for c in chunks]
    token_estimates = [l / 4 for l in lengths]

    report = {
        "total_chunks": len(chunks),
        "char_length": {
            "min": min(lengths), "max": max(lengths),
            "mean": np.mean(lengths), "median": np.median(lengths),
            "std": np.std(lengths),
        },
        "token_estimate": {
            "min": min(token_estimates), "max": max(token_estimates),
            "mean": np.mean(token_estimates), "median": np.median(token_estimates),
        },
        "very_short": sum(1 for l in lengths if l < 100),
        "very_long": sum(1 for l in lengths if l > 5000),
    }

    print(f"Chunks: {report['total_chunks']}")
    print(f"Char length: {report['char_length']['mean']:.0f} avg, {report['char_length']['median']:.0f} median")
    print(f"Token estimate: {report['token_estimate']['mean']:.0f} avg")
    print(f"Very short (<100 chars): {report['very_short']}")
    print(f"Very long (>5000 chars): {report['very_long']}")
    return report
```

---

## Full Post-Chunking Pipeline

```python
def post_chunking_quality_pipeline(chunks):
    print(f"Input: {len(chunks)} chunks")

    # 1. Basic validation
    valid = [c for c in chunks if validate_chunk(c.page_content, c.metadata).is_valid]
    print(f"After validation: {len(valid)}")

    # 2. Exact deduplication
    valid = deduplicate_exact(valid)

    # 3. Fuzzy deduplication
    valid = deduplicate_fuzzy(valid, threshold=0.92)

    # 4. Distribution report
    chunk_distribution_report(valid)

    print(f"Output: {len(valid)} quality chunks")
    return valid
```

## Best Practices

- Validate chunk boundaries — check for mid-sentence starts/ends
- Deduplicate before embedding (embeddings are expensive)
- Monitor chunk size distribution — bimodal distributions suggest chunking issues
- Sample-based LLM assessment for coherence (10-30 chunks is enough)
- Track quality metrics per source document to identify bad ingestion
- Set alerts on chunk count changes between re-indexing runs
