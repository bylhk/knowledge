# LangChain Stack Architecture

## Selected Stack

| # | Category | Tools |
|---|----------|-------|
| 1 | AI Framework | LangChain, LangGraph |
| 2 | Visual AI Workflow | Langflow |
| 3 | Prompt Optimisation | DSPy |
| 4 | Fine-Tuning | LangChain + sub-packages |
| 5 | LLM Structured Output | Guidance, Outlines, LangChain |
| 6 | Ingestion (Document Loading & OCR) | LangChain + LlamaParse / sub-packages |
| 7 | RAG System (Two-Stage Retrieval & Re-Ranking) | LangChain + sub-packages |
| 8 | Evaluation | LangChain + agentevals |
| 9 | LLM Efficiency | LangChain + sub-packages |
| 10 | Guardrails | LangChain + sub-packages |
| 11 | AI Footprint / Observability | LangFuse |
| 12 | AI UI | Chainlit |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                            ┌─────────────┐                                  │
│                            │  Chainlit   │                                  │
│                            │  (Chat UI)  │                                  │
│                            └──────┬──────┘                                  │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────┐
│                         AGENT ORCHESTRATION                                 │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         LangGraph                                      │ │
│  │  (Stateful agent graphs · Checkpointing · Human-in-the-loop · HITL)    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         LangChain (LCEL)                               │ │
│  │  (Chains · Retrievers · Tools · Memory · Prompt Templates)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  STRUCTURED     │    │     RAG PIPELINE     │    │    GUARDRAILS       │
│  OUTPUT         │    │                      │    │                     │
│                 │    │  ┌────────────────┐  │    │  Input validation   │
│  Guidance       │    │  │ Stage 1:       │  │    │  Output validation  │
│  (constrained   │    │  │ Vector Search  │  │    │  Moderation chain   │
│   generation)   │    │  │ + BM25 Hybrid  │  │    │  PII detection      │
│                 │    │  │ (EnsembleRetr) │  │    │  Tool arg schemas   │
│  Outlines       │    │  └───────┬────────┘  │    │                     │
│  (FSM-guided    │    │          │           │    │  (LangChain native  │
│   JSON/regex)   │    │          ▼           │    │   + Pydantic)       │
│                 │    │  ┌────────────────┐  │    └─────────────────────┘
│  LangChain      │    │  │ Stage 2:       │  │
│  (.with_struct  │    │  │ Re-Ranking     │  │
│   _output())    │    │  │ (CrossEncoder/ │  │
│                 │    │  │  CohereRerank) │  │
└─────────────────┘    │  └────────────────┘  │
                       └──────────┬───────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  INGESTION      │    │   VECTOR STORE       │    │  LLM PROVIDERS  │
│                 │    │                      │    │                 │
│  LangChain      │    │  ChromaDB / Qdrant   │    │  OpenAI         │
│  Doc Loaders    │    │  / Pinecone /        │    │  Anthropic      │
│  (PDF, DOCX,    │    │  pgvector            │    │  Google Gemini  │
│   HTML, etc.)   │    │                      │    │  Ollama (local) │
│                 │    │  + Embeddings:       │    │  Bedrock        │
│  LlamaParse     │    │  sentence-           │    │                 │
│  (complex PDFs, │    │  transformers /      │    └─────────────────┘
│   tables, OCR)  │    │  OpenAI / Cohere     │
│                 │    └──────────────────────┘
│  Text Splitters │
│  (Recursive,    │
│   Semantic,     │
│   Code-aware)   │
└─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         CROSS-CUTTING CONCERNS                              │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ PROMPT           │  │ EVALUATION       │  │ OBSERVABILITY             │  │
│  │ OPTIMISATION     │  │                  │  │                           │  │
│  │                  │  │ LangSmith        │  │ LangFuse                  │  │
│  │ DSPy             │  │ (datasets,       │  │ (tracing, cost analytics, │  │
│  │ (declarative     │  │  evaluators,     │  │  prompt management,       │  │
│  │  signatures,     │  │  LLM-as-judge,   │  │  scoring, self-hosted)    │  │
│  │  teleprompters,  │  │  RAG metrics,    │  │                           │  │
│  │  auto-compiled   │  │  online eval)    │  │ Callbacks:                │  │
│  │  few-shot)       │  │                  │  │ LangfuseCallbackHandler   │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────────────┘  │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐  │
│  │ FINE-TUNING      │  │ LLM EFFICIENCY   │  │ VISUAL WORKFLOW           │  │
│  │                  │  │                  │  │                           │  │
│  │ LangSmith →      │  │ Caching          │  │ Langflow                  │  │
│  │  export traces   │  │ (Redis/SQLite)   │  │ (drag-and-drop LangChain  │  │
│  │  as SFT data     │  │                  │  │  graph builder, export    │  │
│  │                  │  │ Rate limiting    │  │  as Python code or API)   │  │
│  │ langchain-openai │  │ Streaming        │  │                           │  │
│  │  (ft: models)    │  │ Batch parallel   │  │                           │  │
│  │                  │  │ Fallbacks        │  │                           │  │
│  │ langchain-       │  │ Token tracking   │  │                           │  │
│  │  huggingface     │  │                  │  │                           │  │
│  │  (LoRA models)   │  │ (all native      │  │                           │  │
│  │                  │  │  LangChain)      │  │                           │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Document/Image                    User Query
      │                                │
      ▼                                ▼
┌───────────┐                   ┌─────────────┐
│ Ingestion │                   │  Chainlit   │
│           │                   │  (Chat UI)  │
│ LlamaParse│                   └──────┬──────┘
│ + LC Doc  │                          │
│ Loaders   │                          ▼
└─────┬─────┘                   ┌─────────────┐
      │                         │  LangGraph  │
      ▼                         │  Agent      │
┌───────────┐                   └──────┬──────┘
│ Text      │                          │
│ Splitters │              ┌───────────┼───────────┐
└─────┬─────┘              │           │           │
      │                    ▼           ▼           ▼
      ▼              ┌─────────┐ ┌──────────┐ ┌────────┐
┌───────────┐        │Retrieval│ │  Tools   │ │  LLM   │
│ Embedding │        │(Hybrid) │ │(custom)  │ │  Call  │
│ Model     │        └────┬────┘ └──────────┘ └───┬────┘
└─────┬─────┘             │                       │
      │                   ▼                       │
      ▼              ┌─────────┐                  │
┌───────────┐        │Re-Ranker│                  │
│ Vector    │        │(Stage 2)│                  │
│ Store     │◄───────└────┬────┘                  │
└───────────┘             │                       │
                          ▼                       ▼
                   ┌─────────────────────────────────┐
                   │  Structured Output (Guidance/   │
                   │  Outlines / .with_structured)   │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │  Guardrails (validation/PII)    │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │  Response → Chainlit            │
                   └─────────────────────────────────┘

                   ═══════════════════════════════════
                   Observability: LangFuse traces
                   every step above automatically
                   ═══════════════════════════════════
```
