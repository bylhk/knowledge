# Embedding Methods for Vector Databases

## Overview

Embeddings convert text into dense numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together in vector space, enabling semantic search over a vector database.

## Where This Fits

```
[Document] → [Ingestion] → [Data Quality] → [Chunking] → ⭐ EMBEDDING ⭐ → [Vector DB]
```

## Key Concepts

- **Dimensionality**: number of values per vector (e.g. 768, 1024, 1536, 3072)
- **Max tokens**: maximum input length the model can embed at once
- **Similarity metric**: cosine similarity, dot product, or euclidean distance
- **Symmetric vs asymmetric**: whether query and document use the same embedding approach

---

## Method 1: Cloud API Embeddings

### Google Text Embedding

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_document",  # or "retrieval_query" for queries
)

# Embed documents
doc_vectors = embeddings.embed_documents(["chunk 1 text", "chunk 2 text"])

# Embed a query (different task type optimises for search)
query_vector = embeddings.embed_query("What is the deployment strategy?")
```

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # Can reduce dimensions for cost/speed tradeoff
)

doc_vectors = embeddings.embed_documents(chunks)
query_vector = embeddings.embed_query("How does the model handle churn?")
```

### OpenAI with Matryoshka Dimensionality

```python
from openai import OpenAI

client = OpenAI()

# text-embedding-3 models support dimension reduction without retraining
response = client.embeddings.create(
    model="text-embedding-3-large",  # native: 3072 dims
    input=["Your text here"],
    dimensions=256,  # Reduce to 256 dims — faster search, less storage
)

# Quality degrades gracefully: 3072 > 1024 > 512 > 256
```

### Cohere Embeddings (with Input Types)

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    input_type="search_document",  # For indexing
)

# At query time, switch input type
query_embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    input_type="search_query",  # For searching
)
```

---

## Method 2: Local / Self-Hosted Embeddings

### Sentence Transformers (Hugging Face)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Embed documents
doc_vectors = model.encode(
    ["chunk 1", "chunk 2"],
    normalize_embeddings=True,  # For cosine similarity
    show_progress_bar=True,
    batch_size=32,
)

# Embed query (BGE models use instruction prefix for queries)
query_vector = model.encode(
    "Represent this sentence for searching relevant passages: What is gradient boosting?",
    normalize_embeddings=True,
)
```

### With LangChain Integration

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},  # or "cpu"
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

vectors = embeddings.embed_documents(chunks)
```

### Jina Embeddings (Long Context)

```python
from sentence_transformers import SentenceTransformer

# 8192 token context — good for longer chunks
model = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")

vectors = model.encode(long_chunks, normalize_embeddings=True)
```

### Nomic Embed (Open Source, Matryoshka)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Supports Matryoshka — truncate dimensions at retrieval time
vectors = model.encode(chunks, normalize_embeddings=True)
# Full: 768 dims. Can truncate to 512, 256, 128 with minimal quality loss.
```

---

## Method 3: Code-Specific Embeddings

For embedding source code, SQL, or technical content where general-purpose models underperform.

### Jina Code Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code")

# Handles code + natural language in same embedding space
code_chunks = [
    "def calculate_price(base, discount): return base * (1 - discount)",
    "SELECT customer_id, SUM(revenue) FROM orders GROUP BY 1",
]
nl_query = "function to compute discounted price"

code_vectors = model.encode(code_chunks, normalize_embeddings=True)
query_vector = model.encode(nl_query, normalize_embeddings=True)
```

### Voyage Code Embeddings (API)

```python
from langchain_voyageai import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(
    model="voyage-code-3",
    input_type="document",  # or "query"
)

vectors = embeddings.embed_documents(code_chunks)
```

---

## Method 4: Multi-Task / Instruction-Based Embeddings

Models that accept a task instruction to optimise the embedding for different use cases.

### GTE with Instructions

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

# Different instructions for different tasks
doc_instruction = "Represent this document for retrieval: "
query_instruction = "Represent this query for searching relevant documents: "

doc_vectors = model.encode([doc_instruction + chunk for chunk in chunks])
query_vector = model.encode(query_instruction + "What algorithms work best for classification?")
```

### E5 Instruct (Microsoft)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# E5 format: "Instruct: {task}\nQuery: {text}"
task = "Given a technical question, retrieve relevant documentation"
query = f"Instruct: {task}\nQuery: How is cross-validation used in model training?"
query_vector = model.encode(query, normalize_embeddings=True)

# Documents use "passage: " prefix
doc_vectors = model.encode(
    [f"passage: {chunk}" for chunk in chunks],
    normalize_embeddings=True,
)
```

---

## Method 5: Sparse Embeddings (Keyword-Aware)

Sparse vectors capture exact keyword matches — complementary to dense semantic embeddings.

### BM25 (Traditional Sparse)

```python
from langchain_community.retrievers import BM25Retriever

# BM25 doesn't produce embeddings — it's a retriever directly
retriever = BM25Retriever.from_texts(chunks, k=5)
results = retriever.invoke("XGBoost feature importance")
```

### SPLADE (Learned Sparse)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

def encode_splade(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**inputs)

    # SPLADE: max pooling over token logits → sparse vector
    logits = output.logits
    sparse_vector = torch.max(torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1), dim=1)[0]
    sparse_vector = sparse_vector.squeeze()

    # Convert to sparse format (non-zero indices and values)
    non_zero = sparse_vector.nonzero().squeeze()
    values = sparse_vector[non_zero]

    tokens = tokenizer.convert_ids_to_tokens(non_zero.tolist())
    return {token: value.item() for token, value in zip(tokens, values)}
```

---

## Method 6: Hybrid Embeddings (Dense + Sparse)

Combines semantic understanding (dense) with keyword precision (sparse).

### With Qdrant

```python
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

client = QdrantClient(":memory:")
dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="hybrid",
    vectors_config={
        "dense": models.VectorParams(size=768, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(),
    },
)

# Search with both (Reciprocal Rank Fusion)
results = client.query_points(
    collection_name="hybrid",
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=20),
        models.Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
    limit=10,
)
```

### With LangChain Ensemble Retriever

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Dense retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma.from_texts(chunks, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Sparse retriever
sparse_retriever = BM25Retriever.from_texts(chunks, k=5)

# Hybrid: weighted combination
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4],  # 60% semantic, 40% keyword
)

results = ensemble_retriever.invoke("XGBoost hyperparameter tuning")
```

---

## Method 7: Fine-Tuned Embeddings

Train or fine-tune an embedding model on your domain data for better retrieval.

### Synthetic Training Data Generation

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class SyntheticPairs(BaseModel):
    pairs: list[dict] = Field(description="List of {query, positive} pairs")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
generator = llm.with_structured_output(SyntheticPairs)

def generate_training_pairs(chunks: list[str], n_per_chunk: int = 3) -> list[dict]:
    """Generate (query, positive_passage) pairs for embedding fine-tuning."""
    all_pairs = []

    for chunk in chunks:
        result = generator.invoke(
            f"Generate {n_per_chunk} natural questions that this text answers. "
            f"Return as query-positive pairs.\n\nText:\n{chunk}"
        )
        for pair in result.pairs:
            pair["positive"] = chunk
            all_pairs.append(pair)

    return all_pairs
```

### Fine-Tune with Sentence Transformers

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load base model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Prepare training data
train_examples = [
    InputExample(texts=[pair["query"], pair["positive"]])
    for pair in training_pairs
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# MultipleNegativesRankingLoss — only needs positive pairs
loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./finetuned-embeddings",
)
```

---

## Method 8: Multi-Modal Embeddings

Embed text and images into the same vector space.

### CLIP (Text + Image)

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer("clip-ViT-B-32")

# Embed text
text_vector = model.encode("A diagram showing the authentication flow")

# Embed image
image = Image.open("auth_flow_diagram.png")
image_vector = model.encode(image)

# Both vectors are in the same space — can search images with text queries
```

### Nomic Embed Vision

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Unified text + vision embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# Text chunks
text_vectors = model.encode(["system architecture document content..."])

# Image (diagrams, charts extracted from PDFs)
image = Image.open("chart.png")
image_vector = model.encode(image)
```

---

## Model Comparison

| Model | Dims | Max Tokens | Type | Cost | Best For |
|-------|------|------------|------|------|----------|
| text-embedding-004 (Google) | 768 | 2048 | Cloud API | ~$0.01/1M tokens | General RAG |
| text-embedding-3-large (OpenAI) | 3072* | 8191 | Cloud API | $0.13/1M tokens | High quality |
| text-embedding-3-small (OpenAI) | 1536* | 8191 | Cloud API | $0.02/1M tokens | Budget |
| embed-english-v3 (Cohere) | 1024 | 512 | Cloud API | $0.10/1M tokens | Multilingual |
| voyage-code-3 (Voyage) | 1024 | 16000 | Cloud API | $0.06/1M tokens | Code |
| bge-base-en-v1.5 | 768 | 512 | Local | Free | General (local) |
| bge-large-en-v1.5 | 1024 | 512 | Local | Free | Higher quality (local) |
| jina-embeddings-v2-base-en | 768 | 8192 | Local | Free | Long chunks |
| jina-embeddings-v2-base-code | 768 | 8192 | Local | Free | Code + text |
| nomic-embed-text-v1.5 | 768* | 8192 | Local | Free | Matryoshka, open |
| gte-large-en-v1.5 | 1024 | 8192 | Local | Free | Instruction-based |
| e5-large-instruct | 1024 | 512 | Local | Free | Task-specific |

*Supports Matryoshka dimension reduction

---

## Choosing a Similarity Metric

| Metric | When to Use | Notes |
|--------|-------------|-------|
| Cosine similarity | Default choice, normalised vectors | Most common, direction-only |
| Dot product | When magnitude matters, Matryoshka | Faster, assumes normalisation |
| Euclidean (L2) | Clustering use cases | Less common for retrieval |

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# With normalised vectors, cosine = dot product
# Always normalise embeddings for consistent similarity scores
```

---

## Embedding with ChromaDB

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Option 1: Built-in embedding function
ef = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    device="cuda",
    normalize_embeddings=True,
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="knowledge",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)

# Add documents (auto-embeds)
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "report.pdf", "page": i} for i in range(len(chunks))],
)

# Query (auto-embeds the query)
results = collection.query(query_texts=["What are the best practices?"], n_results=5)
```

### Pre-Computed Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Pre-compute (useful for batch processing / caching)
embeddings = model.encode(chunks, normalize_embeddings=True, batch_size=64)

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
)
```

---

## Best Practices

- Match embedding model to your content type (code model for code, multilingual for mixed languages)
- Use asymmetric models (separate query/document embeddings) for retrieval tasks
- Normalise embeddings for cosine similarity — avoids magnitude bias
- Batch embed documents (32-64 batch size) for throughput
- Consider Matryoshka models if you need to trade off quality vs speed/storage
- Fine-tune on domain data when off-the-shelf retrieval recall is below 80%
- Use hybrid (dense + sparse) for queries mixing technical terms and natural language
- Don't mix embedding models in the same collection — vectors aren't comparable
- Keep the same model for indexing and querying — different models = different spaces
- Test retrieval quality with representative queries before committing to a model
