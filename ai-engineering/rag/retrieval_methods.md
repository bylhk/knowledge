# Retrieval Methods

## Overview

Retrieval is how you fetch relevant chunks from your vector store to feed into an LLM. Simple similarity search is the baseline, but production systems use multi-stage pipelines with reranking and compression to improve relevance and reduce token usage.

## Where This Fits

```
[User Query] → ⭐ RETRIEVAL ⭐ → [Context] → [LLM] → [Response]
```

---

## Stage 1: Simple Retrieval Methods

### Similarity Search (Baseline)

```python
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Top-k nearest neighbours
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("What are the best practices for API design?")
```

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity — avoids returning 5 near-identical chunks.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,            # Return 5 results
        "fetch_k": 20,     # Fetch 20 candidates first
        "lambda_mult": 0.7, # 1.0 = pure relevance, 0.0 = pure diversity
    },
)

results = retriever.invoke("microservices communication patterns")
```

### Similarity Score Threshold

Only return results above a minimum relevance score.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 10},
)

# Returns 0-10 results depending on how many exceed the threshold
results = retriever.invoke("model calibration techniques")
```

### Metadata Filtering

Narrow the search space before computing similarity.

```python
# ChromaDB
results = vectorstore.similarity_search(
    "feature importance",
    k=5,
    filter={"source": "model_report.pdf"},
)

# Qdrant (via LangChain)
from langchain_qdrant import QdrantVectorStore
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"must": [{"key": "type", "match": {"value": "technical"}}]},
    },
)
```

### Multi-Query Retrieval

Generate multiple query variations to capture different phrasings, then merge results.

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# LLM generates 3+ query variations, retrieves for each, deduplicates
results = multi_retriever.invoke("How does the system handle rate limiting?")
# Internally generates:
# - "rate limiting in distributed systems"
# - "how rate limiting affects system throughput"
# - "rate limiting algorithms for API gateways"
```

### Hybrid Retrieval (Dense + Sparse)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Dense (semantic)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse (keyword)
sparse_retriever = BM25Retriever.from_texts(chunks, k=5)

# Combine with Reciprocal Rank Fusion
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4],
)

results = ensemble.invoke("XGBoost hyperopt bayesian tuning")
```

---

## Stage 2: Reranking (Two-Stage Retrieval)

Retrieve a broad set of candidates (Stage 1), then rerank with a more powerful model (Stage 2). Cross-encoders score query-document pairs jointly, which is more accurate than bi-encoder similarity.

### Why Two Stages?

```
Stage 1: Retrieve 20-50 candidates (fast, approximate — bi-encoder)
Stage 2: Rerank top candidates (slow, accurate — cross-encoder)
Return: Top 5 after reranking
```

### Cross-Encoder Reranking (Local)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Cross-encoder model
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

# Base retriever (Stage 1: fetch many candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Compression retriever (Stage 2: rerank to top 5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

results = reranking_retriever.invoke("What patterns are most important for scalability?")
# Returns top 5 from 20 candidates, reranked by cross-encoder
```

### Cohere Rerank (API)

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5,
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)

results = reranking_retriever.invoke("database query optimisation")
```

### Jina Reranker (API)

```python
from langchain_community.document_compressors import JinaRerank

reranker = JinaRerank(
    model="jina-reranker-v2-base-multilingual",
    top_n=5,
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
```

### FlashRank (Local, Fast)

```python
from langchain_community.document_compressors import FlashrankRerank

reranker = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",
    top_n=5,
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
```

### Manual Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[str, float]]:
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

# Usage
query = "How is cross-validation used?"
candidates = [doc.page_content for doc in base_results]
top_results = rerank(query, candidates, top_n=5)

for text, score in top_results:
    print(f"Score: {score:.3f} | {text[:80]}")
```

---

## Stage 3: Context Compression

After retrieval (and optionally reranking), compress the retrieved documents to only include relevant parts — reduces token usage and noise.

### LLM-Based Extraction

Extracts only the relevant sentences from each retrieved chunk.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Extracts relevant portions from each document
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

results = compression_retriever.invoke("What hyperparameters were tuned?")
# Each result contains only the sentences relevant to the query
```

### LLM-Based Filtering

Removes entire documents that aren't relevant (binary keep/drop).

```python
from langchain.retrievers.document_compressors import LLMChainFilter

filter_compressor = LLMChainFilter.from_llm(llm)

filter_retriever = ContextualCompressionRetriever(
    base_compressor=filter_compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

results = filter_retriever.invoke("model performance benchmarking")
# Returns only chunks the LLM deems relevant
```

### Embedding-Based Filtering (No LLM Cost)

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Filter chunks below similarity threshold
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.75,
)

filter_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

### Chaining Compressors (Pipeline)

Combine multiple compression steps.

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_text_splitters import CharacterTextSplitter

# Pipeline: split long chunks → filter by embedding → extract with LLM
pipeline = DocumentCompressorPipeline(
    transformers=[
        # Step 1: Split any oversized chunks
        CharacterTextSplitter(chunk_size=500, chunk_overlap=0),
        # Step 2: Filter by embedding similarity
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7),
        # Step 3: Extract relevant sentences
        LLMChainExtractor.from_llm(llm),
    ]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

---

## Advanced: Query Transformation

Transform the user query before retrieval to improve results.

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, embed that instead of the query.

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def hyde_retrieval(query: str, vectorstore, k: int = 5):
    # Generate hypothetical answer
    prompt = ChatPromptTemplate.from_template(
        "Write a short passage that would answer this question:\n{query}"
    )
    chain = prompt | llm
    hypothetical_doc = chain.invoke({"query": query}).content

    # Embed the hypothetical answer (not the query)
    results = vectorstore.similarity_search(hypothetical_doc, k=k)
    return results
```

### Step-Back Prompting

Abstract the query to retrieve broader context.

```python
def step_back_retrieval(query: str, vectorstore, llm, k: int = 5):
    # Generate a more general version of the query
    step_back = llm.invoke(
        f"Given this specific question, generate a broader question that would help "
        f"find relevant background information:\n\nSpecific: {query}\nBroader:"
    ).content

    # Retrieve for both original and step-back query
    original_results = vectorstore.similarity_search(query, k=k)
    broad_results = vectorstore.similarity_search(step_back, k=k)

    # Combine and deduplicate
    all_results = original_results + broad_results
    seen = set()
    unique = []
    for doc in all_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique[:k]
```

---

## Full Two-Stage + Compression Pipeline

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors import LLMChainExtractor

# Stage 1: Broad retrieval (20 candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Stage 2: Rerank to top 8
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=8)

# Stage 3: Compress — extract only relevant sentences
extractor = LLMChainExtractor.from_llm(llm)

# Combine into pipeline
pipeline = DocumentCompressorPipeline(transformers=[reranker, extractor])

final_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever,
)

# Result: highly relevant, compressed context
results = final_retriever.invoke("What design patterns improve system reliability?")
```

---

## Comparison

| Method | Precision | Latency | Cost | Complexity |
|--------|-----------|---------|------|------------|
| Simple top-k | Baseline | Fast | Free | Low |
| MMR | Better (diverse) | Fast | Free | Low |
| Multi-query | Good | Moderate | LLM call | Low |
| Hybrid (dense+sparse) | Good | Fast | Free | Moderate |
| Two-stage reranking | Very good | Moderate | Free (local) / API | Moderate |
| LLM compression | Excellent | Slow | LLM calls | Moderate |
| Full pipeline | Best | Slowest | Moderate | High |

## When to Use What

| Scenario | Recommended |
|----------|-------------|
| Quick prototype | Simple top-k (k=5) |
| Diverse results needed | MMR |
| Technical jargon queries | Hybrid (dense + BM25) |
| High precision required | Two-stage reranking |
| Long context window limit | Compression after reranking |
| Complex/ambiguous queries | Multi-query + reranking |
| Maximum quality | Full pipeline (retrieve → rerank → compress) |

## Best Practices

- Always over-retrieve then filter down (fetch 20, return 5)
- Use reranking when precision matters more than speed
- Cross-encoders are much more accurate than bi-encoders for relevance scoring
- Compression saves tokens but adds latency — use when context window is tight
- Multi-query catches missed phrasings but costs one extra LLM call
- HyDE works well for questions where the answer's embedding space differs from the query's
- Monitor which retrieved docs actually get used by the LLM (attribution tracking)
- Test retrieval independently before adding generation — bad retrieval = bad answers
