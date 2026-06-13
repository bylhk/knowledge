# Vector Databases: Insert & Retrieval

## Overview

Vector databases store embedding vectors alongside metadata, enabling fast approximate nearest-neighbour (ANN) search. They're the backbone of RAG systems — you insert embedded chunks and retrieve the most semantically similar ones at query time.

## Where This Fits

```
[Document] → [Ingestion] → [Quality] → [Chunking] → [Embedding] → ⭐ VECTOR DB ⭐ → [Retrieval]
```

---

## ChromaDB (Local / Embedded)

Lightweight, Python-native. Great for prototyping and small-medium collections.

### Setup & Insert

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Embedding function (auto-embeds on add/query)
ef = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",
    normalize_embeddings=True,
)

# Create collection
collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},  # similarity metric
)

# Insert with auto-embedding
collection.add(
    documents=["chunk 1 text", "chunk 2 text", "chunk 3 text"],
    ids=["doc_001", "doc_002", "doc_003"],
    metadatas=[
        {"source": "report.pdf", "page": 1, "type": "text"},
        {"source": "report.pdf", "page": 2, "type": "text"},
        {"source": "report.pdf", "page": 3, "type": "table"},
    ],
)

# Insert with pre-computed embeddings
collection.add(
    documents=chunks,
    embeddings=precomputed_vectors,  # list of lists
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=metadata_list,
)
```

### Retrieval

```python
# Basic similarity search
results = collection.query(
    query_texts=["How do I implement authentication?"],
    n_results=5,
)
# results["documents"][0] — list of matching texts
# results["distances"][0] — similarity scores
# results["metadatas"][0] — metadata for each result

# With metadata filter
results = collection.query(
    query_texts=["authentication methods"],
    n_results=5,
    where={"source": "api_docs.pdf"},         # exact match
    where_document={"$contains": "discount"},       # content filter
)

# Filter with operators
results = collection.query(
    query_texts=["feature importance"],
    n_results=10,
    where={
        "$and": [
            {"type": {"$eq": "text"}},
            {"page": {"$gte": 5}},
        ]
    },
)
```

### Update & Delete

```python
# Update documents
collection.update(
    ids=["doc_001"],
    documents=["updated chunk text"],
    metadatas=[{"source": "report_v2.pdf", "page": 1}],
)

# Upsert (insert or update)
collection.upsert(
    ids=["doc_001", "doc_004"],
    documents=["updated text", "new chunk"],
    metadatas=[{"source": "v2.pdf"}, {"source": "new.pdf"}],
)

# Delete
collection.delete(ids=["doc_002", "doc_003"])
collection.delete(where={"source": "old_report.pdf"})
```

---

## Qdrant (Production-Grade)

High-performance, supports hybrid search, filtering, and multi-vector storage.

### Setup & Insert

```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# Connect (local or cloud)
client = QdrantClient(":memory:")  # or url="http://localhost:6333"
# client = QdrantClient(url="https://xxx.qdrant.io", api_key="...")

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Create collection
client.create_collection(
    collection_name="knowledge",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
    ),
)

# Insert
vectors = model.encode(chunks, normalize_embeddings=True)

client.upsert(
    collection_name="knowledge",
    points=[
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "text": chunks[i],
                "source": "report.pdf",
                "page": i,
                "type": "text",
            },
        )
        for i in range(len(chunks))
    ],
)
```

### Retrieval

```python
query_vector = model.encode("What is the authentication flow?", normalize_embeddings=True)

# Basic search
results = client.query_points(
    collection_name="knowledge",
    query=query_vector.tolist(),
    limit=5,
)

for point in results.points:
    print(f"Score: {point.score:.3f} | {point.payload['text'][:80]}")

# With filtering
results = client.query_points(
    collection_name="knowledge",
    query=query_vector.tolist(),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value="report.pdf")),
            models.FieldCondition(key="page", range=models.Range(gte=5, lte=20)),
        ]
    ),
    limit=5,
)

# Hybrid search (dense + sparse with RRF)
results = client.query_points(
    collection_name="hybrid_collection",
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=20),
        models.Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

---

## Pinecone (Managed Cloud)

Fully managed, serverless option. No infrastructure to maintain.

### Setup & Insert

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-key")

# Create index
pc.create_index(
    name="knowledge-base",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("knowledge-base")

# Insert (batch)
vectors_to_upsert = [
    {
        "id": f"chunk_{i}",
        "values": embedding_vectors[i],
        "metadata": {"source": "report.pdf", "text": chunks[i], "page": i},
    }
    for i in range(len(chunks))
]

# Upsert in batches of 100
for i in range(0, len(vectors_to_upsert), 100):
    batch = vectors_to_upsert[i : i + 100]
    index.upsert(vectors=batch)
```

### Retrieval

```python
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True,
    filter={"source": {"$eq": "report.pdf"}},
)

for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | {match['metadata']['text'][:80]}")
```

---

## FAISS (Meta — In-Memory, High Performance)

Extremely fast for large-scale similarity search. No metadata filtering natively — pair with a document store.

### Setup & Insert

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
vectors = model.encode(chunks, normalize_embeddings=True).astype("float32")

# Create index
dimension = vectors.shape[1]  # 768

# Flat index (exact search, small datasets)
index = faiss.IndexFlatIP(dimension)  # Inner Product (= cosine for normalised vectors)

# IVF index (approximate, large datasets)
# nlist = number of clusters
nlist = 100
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
index.train(vectors)  # Required for IVF

# Add vectors
index.add(vectors)
print(f"Index contains {index.ntotal} vectors")

# Save / load
faiss.write_index(index, "knowledge.faiss")
index = faiss.read_index("knowledge.faiss")
```

### Retrieval

```python
query_vector = model.encode("authentication best practices", normalize_embeddings=True).astype("float32")
query_vector = query_vector.reshape(1, -1)

# Search
k = 5
distances, indices = index.search(query_vector, k)

for dist, idx in zip(distances[0], indices[0]):
    print(f"Score: {dist:.3f} | Chunk: {chunks[idx][:80]}")
```

### With LangChain

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create from documents
vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadata_list)

# Save / load
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# Search
results = vectorstore.similarity_search_with_score("authentication flow", k=5)
for doc, score in results:
    print(f"Score: {score:.3f} | {doc.page_content[:80]}")
```

---

## Weaviate (GraphQL-Based)

Supports hybrid search, generative search, and multi-tenancy.

### Setup & Insert

```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType

client = weaviate.connect_to_local()  # or connect_to_wcs for cloud

# Create collection with vectorizer
collection = client.collections.create(
    name="KnowledgeBase",
    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
    properties=[
        Property(name="content", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="page", data_type=DataType.INT),
    ],
)

# Insert (auto-vectorised)
with collection.batch.dynamic() as batch:
    for i, chunk in enumerate(chunks):
        batch.add_object(
            properties={"content": chunk, "source": "report.pdf", "page": i},
        )
```

### Retrieval

```python
# Semantic search
results = collection.query.near_text(
    query="user authentication methods",
    limit=5,
    filters=weaviate.classes.query.Filter.by_property("source").equal("report.pdf"),
)

for obj in results.objects:
    print(f"{obj.properties['content'][:80]}")

# Hybrid search (BM25 + vector)
results = collection.query.hybrid(
    query="XGBoost feature importance",
    limit=5,
    alpha=0.7,  # 0 = pure BM25, 1 = pure vector
)
```

---

## LangChain Unified Interface

Works with any supported vector store — swap backends without changing retrieval code.

```python
from langchain_community.vectorstores import Chroma  # or FAISS, Qdrant, Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Create and populate
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadata_list,
    collection_name="knowledge",
    persist_directory="./chroma_db",
)

# As retriever (for chains)
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5},
)

# MMR (Maximum Marginal Relevance) — diverse results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
)

# With score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 10},
)

# Use in a chain
results = retriever.invoke("How do I implement OAuth2?")
```

---

## Comparison

| Database | Type | Hybrid | Filtering | Scaling | Best For |
|----------|------|--------|-----------|---------|----------|
| ChromaDB | Embedded/local | No | Basic | Small-medium | Prototyping, single-user |
| Qdrant | Self-hosted/cloud | Yes | Advanced | Large | Production, hybrid search |
| Pinecone | Managed cloud | Yes | Good | Auto-scale | Serverless, zero-ops |
| FAISS | In-memory | No | None (external) | Very large | Speed-critical, batch |
| Weaviate | Self-hosted/cloud | Yes | GraphQL | Large | Multi-tenant, generative |
| pgvector | Postgres extension | No | SQL | Medium | Already using Postgres |

## Index Types & Tradeoffs

| Index | Search Type | Speed | Accuracy | Memory | Build Time |
|-------|-------------|-------|----------|--------|------------|
| Flat | Exact | Slow (O(n)) | 100% | Low | Instant |
| IVF | Approximate | Fast | ~95% | Low | Minutes |
| HNSW | Approximate | Very fast | ~99% | High | Minutes |
| PQ (Product Quantisation) | Approximate | Fast | ~90% | Very low | Minutes |
| IVF-PQ | Approximate | Very fast | ~92% | Very low | Minutes |

## Best Practices

- Start with ChromaDB for prototyping, migrate to Qdrant/Pinecone for production
- Use HNSW index for most use cases (best speed/accuracy tradeoff)
- Always store raw text in metadata — enables debugging and reranking
- Batch inserts (100-1000 at a time) for throughput
- Use metadata filtering to narrow search scope before vector similarity
- MMR retrieval reduces redundancy in results
- Don't mix embedding models in a single collection
- Monitor index size and query latency as collections grow
- Use namespaces/collections to separate document types or tenants
