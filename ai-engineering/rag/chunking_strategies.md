# Advanced Chunking Strategies

## Overview

Chunking splits ingested text into smaller pieces for embedding and retrieval. Naive chunking (fixed character splits) loses context and breaks semantic units. Advanced methods preserve meaning, structure, and relationships between chunks.

## Where This Fits

```
[Document] → [Ingestion] → [Data Quality] → ⭐ CHUNKING ⭐ → [Embedding] → [Vector DB]
```

## Why Chunking Matters

- Too small: chunks lack context, embeddings are shallow
- Too large: embeddings dilute meaning, exceed model context
- Bad boundaries: sentences split mid-thought, tables broken apart
- No metadata: can't trace chunks back to source or section

---

## Method 1: Token Text Splitting (Baseline)

Splits by token count rather than character count — aligns with how models actually consume text. More accurate than character-based splitting where 1 token ≈ 4 chars varies wildly.

### TokenTextSplitter (LangChain)

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,             # 500 tokens per chunk
    chunk_overlap=50,           # 50 token overlap
    encoding_name="cl100k_base",  # GPT-4/4o tokenizer
)

chunks = splitter.split_text(document_text)
```

### Recursive Splitting with Token-Based Length

```python
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

enc = tiktoken.encoding_for_model("gpt-4o")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,               # 500 tokens
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda t: len(enc.encode(t)),  # measure in tokens
)

chunks = splitter.split_text(document_text)
```

### Character vs Token Splitting

| Aspect | Character (`len`) | Token (`tiktoken`) |
|--------|-------------------|-------------------|
| Speed | Faster | Slightly slower |
| Accuracy | Approximate (~4 chars/token) | Exact token count |
| Use when | Prototyping, speed matters | Production, precision matters |

- ✅ Accurate to what the model sees
- ✅ Predictable context window usage
- ❌ Slightly slower than character counting
- ❌ Still no semantic awareness — splits mid-paragraph if size limit hit

### Recursive Character Splitting (Character-Based Variant)

For quick prototyping where token precision isn't needed:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # characters, not tokens
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,    # default: character count
)

chunks = splitter.split_text(document_text)
```

---

## Method 2: Semantic Chunking

Splits based on embedding similarity between sentences. Adjacent sentences with similar meaning stay together; a drop in similarity triggers a new chunk.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Breakpoint methods: percentile, standard_deviation, interquartile
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=75,  # Split at 75th percentile of distance
)

chunks = chunker.split_text(document_text)
```

### How It Works

1. Split text into sentences
2. Embed each sentence
3. Calculate cosine distance between adjacent sentence embeddings
4. Where distance exceeds threshold → insert chunk boundary

```python
# Manual implementation for understanding
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def semantic_chunk(text: str, threshold: float = 0.3) -> list[str]:
    # Split into sentences
    sentences = [s.strip() for s in text.split(". ") if s.strip()]

    # Embed all sentences
    embeddings = model.encode(sentences)

    # Calculate distances between adjacent sentences
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
        )
        distance = 1 - sim

        if distance > threshold:
            # Topic shift — start new chunk
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(". ".join(current_chunk) + ".")
    return chunks
```

- ✅ Respects topic boundaries naturally
- ✅ Variable chunk sizes based on content
- ❌ Slower (requires embedding every sentence)
- ❌ Threshold tuning needed per domain

---

## Method 3: Agentic Chunking (LLM-Driven)

Uses an LLM to decide chunk boundaries based on semantic understanding.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ChunkBoundaries(BaseModel):
    chunks: list[str] = Field(description="List of semantically coherent text chunks")
    reasoning: list[str] = Field(description="Why each boundary was chosen")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
structured_llm = llm.with_structured_output(ChunkBoundaries)

def agentic_chunk(text: str, context: str = "") -> list[str]:
    result = structured_llm.invoke(
        f"Split this text into semantically coherent chunks for a RAG knowledge base. "
        f"Each chunk should be self-contained and cover one topic or concept. "
        f"Target 200-500 words per chunk. Keep tables and lists intact.\n\n"
        f"Context: {context}\n\n"
        f"Text:\n{text}"
    )
    return result.chunks
```

- ✅ Best semantic understanding
- ✅ Handles complex document structures
- ❌ Expensive (LLM call per document)
- ❌ Slow, non-deterministic

---

## Method 4: Document Structure-Aware Chunking

Respects the document's inherent structure (headings, sections, code blocks).

### Markdown Header Splitting

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# First: split by headers (preserves section context)
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
header_chunks = header_splitter.split_text(markdown_text)

# Second: split large sections further with recursive splitter
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

final_chunks = []
for chunk in header_chunks:
    if len(chunk.page_content) > 1000:
        sub_chunks = recursive_splitter.split_text(chunk.page_content)
        for sc in sub_chunks:
            final_chunks.append({"content": sc, "metadata": chunk.metadata})
    else:
        final_chunks.append({"content": chunk.page_content, "metadata": chunk.metadata})
```

### HTML Section Splitting

```python
from langchain_text_splitters import HTMLSectionSplitter

splitter = HTMLSectionSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
)

chunks = splitter.split_text(html_content)
# Each chunk retains header hierarchy in metadata
```

### Code-Aware Splitting

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Python-aware splitting (respects class/function boundaries)
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)

chunks = python_splitter.split_text(python_code)

# Supported: PYTHON, JS, TS, GO, RUST, JAVA, CPP, SQL, MARKDOWN, and more
```

---

## Method 5: Parent-Child (Hierarchical) Chunking

Store small chunks for precise retrieval but return larger parent chunks for context.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Child splitter: small chunks for embedding precision
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Parent splitter: larger chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma(embedding_function=embeddings, collection_name="children")
docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents — automatically creates parent-child relationships
retriever.add_documents(documents)

# Query retrieves child (precise match) but returns parent (full context)
results = retriever.invoke("What is the deployment strategy?")
# Returns 2000-char parent chunks that contain the matching 400-char children
```

### How It Works

```
┌─────────────────────────────── Parent Chunk (2000 chars) ──────────────────────────────┐
│  ┌─── Child 1 (400) ───┐  ┌─── Child 2 (400) ───┐  ┌─── Child 3 (400) ───┐          │
│  │  embedded & searched │  │  ← match found here │  │                     │          │
│  └──────────────────────┘  └──────────────────────┘  └─────────────────────┘          │
└────────────────────────────── this entire block returned ──────────────────────────────┘
```

- ✅ Best of both worlds: precise retrieval + rich context
- ✅ LLM sees full surrounding context
- ❌ More complex infrastructure (two stores)
- ❌ Higher storage requirements

---

## Method 6: Proposition-Based Chunking

Decompose text into atomic propositions (single facts), then group related ones.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class Propositions(BaseModel):
    propositions: list[str] = Field(
        description="List of atomic, self-contained factual statements"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
extractor = llm.with_structured_output(Propositions)

def extract_propositions(text: str) -> list[str]:
    result = extractor.invoke(
        "Decompose this text into simple, atomic propositions. "
        "Each proposition should:\n"
        "- Be a single, self-contained fact\n"
        "- Include necessary context to stand alone\n"
        "- Be concise but complete\n\n"
        f"Text:\n{text}"
    )
    return result.propositions

# Example
text = """
XGBoost was chosen for the classification model because it handles 
tabular data well and provides feature importance. The model was 
trained on 18 months of historical data with 47 features.
"""

props = extract_propositions(text)
# [
#   "XGBoost was chosen for the classification model.",
#   "XGBoost handles tabular data well.",
#   "XGBoost provides feature importance.",
#   "The classification model was trained on 18 months of historical data.",
#   "The classification model uses 47 features."
# ]
```

- ✅ Maximum retrieval precision
- ✅ Each chunk is a complete, searchable fact
- ❌ Very expensive (LLM call per passage)
- ❌ Loses narrative flow and context

---

## Method 7: Late Chunking (Contextual Embeddings)

Embed the full document first, then split into chunks — each chunk's embedding retains full-document context.

```python
# Concept: use a long-context embedding model
# Embed full document → segment embeddings by chunk boundaries

from transformers import AutoModel, AutoTokenizer
import torch

model_name = "jinaai/jina-embeddings-v2-base-en"  # 8192 token context
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def late_chunking(text: str, chunk_size: int = 200) -> list[tuple[str, list[float]]]:
    """Embed full text, then extract per-chunk embeddings from token representations."""
    # Tokenize full document
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)

    # Split tokens into chunks
    tokens = tokenizer.tokenize(text)
    chunk_boundaries = range(0, len(tokens), chunk_size)

    chunks_with_embeddings = []
    for start in chunk_boundaries:
        end = min(start + chunk_size, len(tokens))
        # Mean pool the token embeddings for this chunk
        chunk_embedding = token_embeddings[start:end].mean(dim=0).numpy()
        chunk_text = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks_with_embeddings.append((chunk_text, chunk_embedding.tolist()))

    return chunks_with_embeddings
```

- ✅ Chunks retain awareness of surrounding document context
- ✅ Resolves pronouns and references better than naive chunking
- ❌ Requires long-context embedding model
- ❌ More complex implementation

---

## Method 8: Contextual Chunk Headers (Metadata Enrichment)

Prepend section/document context to each chunk before embedding.

```python
def add_contextual_headers(chunks: list[dict]) -> list[dict]:
    """Prepend hierarchical context to each chunk for better embeddings."""
    enriched = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        header_parts = []

        # Build context hierarchy
        if metadata.get("document_title"):
            header_parts.append(f"Document: {metadata['document_title']}")
        if metadata.get("h1"):
            header_parts.append(f"Section: {metadata['h1']}")
        if metadata.get("h2"):
            header_parts.append(f"Subsection: {metadata['h2']}")

        # Prepend context to chunk content
        context_header = " > ".join(header_parts)
        enriched_content = f"{context_header}\n\n{chunk['content']}" if header_parts else chunk["content"]

        enriched.append({
            "content": enriched_content,
            "content_raw": chunk["content"],  # Keep original for display
            "metadata": metadata,
        })

    return enriched

# Example result:
# "Document: Technical Report 2024 > Section: Model Architecture > Subsection: Feature Engineering
#
#  The top 5 features by importance are..."
```

- ✅ Simple to implement, big impact on retrieval quality
- ✅ Chunks embed with awareness of where they sit in the document
- ✅ No extra infrastructure needed
- ❌ Increases chunk size (and embedding cost)

---

## Method 9: Multi-Vector Chunking

Generate multiple representations per chunk — raw text, summary, questions it answers.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ChunkRepresentations(BaseModel):
    summary: str = Field(description="One-sentence summary of the chunk")
    questions: list[str] = Field(description="Questions this chunk can answer")
    keywords: list[str] = Field(description="Key terms and concepts")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
generator = llm.with_structured_output(ChunkRepresentations)

def generate_multi_vectors(chunk_text: str) -> dict:
    reps = generator.invoke(
        f"Analyse this text chunk and provide:\n"
        f"1. A one-sentence summary\n"
        f"2. Questions this chunk answers\n"
        f"3. Key terms\n\n"
        f"Chunk:\n{chunk_text}"
    )
    return {
        "original": chunk_text,
        "summary": reps.summary,
        "questions": reps.questions,
        "keywords": reps.keywords,
    }

# Store all representations — search across all, return original
# This catches queries phrased as questions, keyword searches, and summaries
```

- ✅ Catches diverse query styles (questions, keywords, descriptions)
- ✅ Dramatically improves retrieval recall
- ❌ 3-4x storage and embedding cost
- ❌ LLM cost per chunk for generation

---

## Comparison

| Method | Retrieval Quality | Cost | Complexity | Best For |
|--------|------------------|------|------------|----------|
| Recursive character | Baseline | Free | Low | Quick prototypes |
| Semantic | Good | Moderate | Low | Topic-diverse documents |
| Agentic (LLM) | Excellent | High | Moderate | High-value documents |
| Structure-aware | Good | Free | Low | Markdown, code, HTML |
| Parent-child | Very good | Low | Moderate | Long documents |
| Proposition-based | Excellent precision | High | Moderate | Factual/reference docs |
| Late chunking | Very good | Moderate | High | Pronoun-heavy text |
| Contextual headers | Good (easy win) | Free | Low | Any document |
| Multi-vector | Excellent recall | High | Moderate | Diverse query patterns |

## Recommended Combinations

| Use Case | Strategy |
|----------|----------|
| General RAG | Structure-aware + contextual headers |
| Technical docs / code | Code-aware splitting + parent-child |
| Enterprise knowledge base | Semantic chunking + multi-vector |
| High-precision Q&A | Proposition-based + contextual headers |
| Large document corpus (budget) | Recursive + parent-child |
| Maximum quality (no budget limit) | Agentic chunking + multi-vector |

## Best Practices

- Start with structure-aware splitting + contextual headers — high impact, low cost
- Add parent-child retrieval when users complain about missing context
- Use semantic chunking when documents mix multiple topics in flowing prose
- Never split tables, code blocks, or lists — keep them as single chunks
- Overlap is less important with semantic splitting (boundaries are natural)
- Evaluate chunking quality with retrieval tests, not just eyeballing
- Target 200–1000 tokens per chunk for most embedding models
- Store chunk boundaries as metadata — enables re-chunking without re-ingesting
