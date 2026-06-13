# LLM Customisation Strategies

## Comparison Matrix

| | Prompting | RAG | Fine-tuning | Compound |
|---|---|---|---|---|
| Training required | No | No | Yes | Depends |
| External data at inference | No | Yes | No | Yes |
| Iteration speed | Fast | Medium | Slow | Medium |
| Initial cost | Low | Medium | High | High |
| Running cost | Medium | High | Low | High |
| Knowledge freshness | Static | Dynamic | Static | Dynamic |
| Environment | Cloud | Cloud | Cloud/Edge | Cloud |
| Best for | Formatting, persona | Knowledge Q&A | Style, structure | Production systems |

---

## Decision Flowchart

```
Can prompting alone solve it?
  → Yes: Use system prompting
  → No: Does the model need access to external/changing knowledge?
      → Yes: Add RAG
      → No: Does the model need to learn new behaviour/style?
          → Yes: Fine-tune (LoRA/QLoRA for efficiency)
          → No: Consider compound approach with tool-use
```


## 1. System Prompting

**What:** Steer model behaviour at inference time using natural language instructions. No training, no external data retrieval — just carefully crafted prompts.

**When to use:**
- Quick prototyping and iteration
- Task-specific formatting or persona
- Guardrails and output constraints

**Limitations:**
- Bounded by context window size
- No access to knowledge beyond training data
- Can be brittle with complex instructions

---

## 2. Retrieval-Augmented Generation (RAG)

**What:** Augment the LLM's context at inference time by retrieving relevant documents from an external knowledge base. The model generates answers grounded in retrieved content.

**When to use:**
- Domain-specific Q&A over private/changing data
- Reducing hallucination with source attribution
- Knowledge that updates frequently (docs, policies, code)

**Limitations:**
- Quality depends heavily on retrieval step
- Adds latency (retrieval + generation)
- Chunk size and embedding quality matter a lot

**Architecture:**
```
Query → Embedding → Vector Search → Top-K Chunks → LLM (query + chunks) → Answer
```

---

## 3. Fine-Tuning / Adapters

**What:** Modify model weights to encode new knowledge or behaviour. Adapters (LoRA, QLoRA) are parameter-efficient methods that train a small number of additional weights rather than the full model.

**When to use:**
- Teaching the model a specific style, format, or domain language
- When prompting alone can't capture the desired behaviour
- Consistent structured outputs (JSON schemas, code patterns)

**Limitations:**
- Requires labelled training data
- Risk of catastrophic forgetting
- More expensive to iterate than prompting/RAG

**Techniques:**
| Technique | Description |
|-----------|-------------|
| Full fine-tuning | Update all model weights (expensive, needs large GPU) |
| LoRA | Low-Rank Adaptation — trains small rank-decomposition matrices |
| QLoRA | LoRA on quantised (4-bit) base model — fits on consumer GPUs |
| DPO / RLHF | Preference-based alignment (human feedback) |
| Prefix Tuning | Learnable soft prefixes prepended to layers |


---

## 4. Compound AI Systems (Combination)

**What:** Combine multiple techniques together. A fine-tuned model can use RAG for dynamic knowledge and system prompts for behavioural guardrails. This is how most production systems work.

**When to use:**
- Production-grade applications
- Complex multi-step reasoning (agentic workflows)
- Need both learned behaviour AND dynamic knowledge

**Example Architectures:**

### Fine-tuned + RAG + System Prompt
```
System Prompt (persona, guardrails)
    ↓
Fine-tuned Model (domain style/format)
    ↓
RAG Pipeline (current knowledge retrieval)
    ↓
Generated Response
```

### Agentic System (Tool-Use)
```
System Prompt + Few-shot examples
    ↓
LLM (routing/planning)
    ↓
Tools: [RAG search, API calls, Code execution, DB queries]
    ↓
LLM (synthesis)
    ↓
Final Answer
```

