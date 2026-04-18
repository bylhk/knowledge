# Generative AI

Generative AI models produce new content — text, images, code, audio, or video — rather than classifying or regressing on existing data. They are typically large pretrained models accessed via API or fine-tuned for specific tasks.

---

## Text Generation

Large language models (LLMs) generate text autoregressively — one token at a time, conditioned on all previous tokens.

| Model | Organisation | Key characteristics |
|-------|-------------|-------------------|
| GPT-4o / GPT-4.1 | OpenAI | Multimodal (vision + text + audio), strong reasoning, widely used via API |
| o3 / o4-mini | OpenAI | Reasoning models — extended chain-of-thought, strong on maths and coding |
| Gemini 2.0 / 2.5 Pro | Google DeepMind | Multimodal, very long context (1M tokens), integrated with Google services |
| Claude 3.5 / 3.7 Sonnet | Anthropic | Strong at long documents and coding, safety-focused, extended thinking mode |
| Grok 3 | xAI | Real-time web access, strong reasoning, DeepSearch mode |
| LLaMA 3.3 | Meta | Open weights — fine-tuneable, self-hostable, 8B–405B |
| DeepSeek-V3 / R1 | DeepSeek | MoE architecture, open weights, R1 is a strong reasoning model |
| Mistral Large | Mistral AI | Open and API, strong multilingual, efficient |
| Qwen 2.5 | Alibaba | Strong multilingual and coding, open weights |

### Key generation parameters

| Parameter | Effect |
|-----------|--------|
| Temperature | Controls randomness. 0 = deterministic (greedy), 1 = standard sampling, > 1 = more random |
| Top-p (nucleus sampling) | Sample from the smallest set of tokens whose cumulative probability ≥ p |
| Top-k | Sample from the k most probable tokens only |
| Max tokens | Maximum length of the generated response |
| System prompt | Persistent instruction that shapes model behaviour across the conversation |
| Thinking / reasoning budget | For reasoning models — controls how many tokens the model spends "thinking" before answering |

### RAG — Retrieval-Augmented Generation

LLMs have a fixed knowledge cutoff and cannot access private data. RAG augments generation by retrieving relevant documents at query time and injecting them into the prompt.

```
User question
  → Embed question → Vector similarity search → Retrieve top-K documents
  → Construct prompt: [system] + [retrieved docs] + [question]
  → LLM generates answer grounded in retrieved context
  → Output
```

**Components:**
- **Embedding model** — encodes documents and queries into dense vectors (e.g. `text-embedding-3-small`, `bge-large`, `nomic-embed-text`)
- **Vector database** — stores and retrieves embeddings by similarity (Pinecone, Weaviate, pgvector, FAISS, Qdrant)
- **LLM** — generates the final answer conditioned on retrieved context

**Advanced RAG patterns:**

| Pattern | What it adds |
|---------|-------------|
| Hybrid search | Combine dense vector search with BM25 keyword search — better recall |
| Re-ranking | Use a cross-encoder to re-rank retrieved chunks before passing to LLM |
| HyDE | Generate a hypothetical answer first, then retrieve using that as the query |
| Agentic RAG | LLM decides when and what to retrieve — multi-step retrieval for complex questions |

**When to use RAG:**
- Private or proprietary knowledge not in the model's training data
- Knowledge that changes frequently (product catalogues, documentation)
- When citations and source attribution are required

---

## Agentic AI

LLM agents extend generation with the ability to use tools, plan multi-step tasks, and act on the environment.

```
User goal → LLM plans steps → calls tools (search, code, APIs) → observes results → continues until done
```

| Framework | Key idea |
|-----------|----------|
| ReAct | Interleave reasoning (Thought) and action (Act) steps — LLM reasons about what tool to call |
| Function calling | LLM outputs structured JSON to call predefined functions — native in GPT-4, Claude, Gemini |
| LangChain / LlamaIndex | Orchestration frameworks for building RAG pipelines and agents |
| AutoGen | Multi-agent framework — multiple LLMs collaborate and critique each other |
| MCP (Model Context Protocol) | Standardised protocol for connecting LLMs to external tools and data sources |

---

## Image Generation

### DALL-E 3 (OpenAI)

Text-to-image model based on diffusion with strong prompt adherence. DALL-E 3 is integrated into ChatGPT and uses GPT-4 to automatically rewrite prompts for better results.

### Midjourney v6

Diffusion-based text-to-image model known for high aesthetic quality and photorealism. Accessed via Discord or web UI. Less programmable than DALL-E — primarily a creative tool.

### Stable Diffusion / FLUX

Open-weights latent diffusion models. FLUX.1 (Black Forest Labs, 2024) is the successor to Stable Diffusion from the original creators — significantly better prompt adherence and image quality.

**How latent diffusion works:**
```
Text prompt → text encoder → conditioning signal
Gaussian noise (latent space)
  → [Denoise with U-Net/DiT conditioned on text] × T steps
  → Clean latent
  → VAE decoder → Image
```

FLUX uses a Diffusion Transformer (DiT) instead of U-Net — a pure Transformer architecture for the denoising network, enabling better scaling.

| Model | Architecture | Open weights | Key strength |
|-------|-------------|-------------|-------------|
| Stable Diffusion XL | U-Net latent diffusion | ✅ | Large ecosystem, ControlNet |
| FLUX.1 [dev] | DiT latent diffusion | ✅ | Best open-weight quality |
| DALL-E 3 | Diffusion | ❌ | Strong prompt adherence |
| Midjourney v6 | Diffusion | ❌ | Aesthetic quality |
| Imagen 3 | Cascaded diffusion | ❌ | Google, photorealism |

### ControlNet

Adds spatial conditioning to diffusion models — control generation with edge maps, depth maps, pose skeletons, or segmentation masks. Enables precise layout control without changing the base model.

---

## Code Generation

| Tool | Underlying model | Integration |
|------|-----------------|-------------|
| GitHub Copilot | OpenAI Codex / GPT-4 | IDE plugin (VS Code, JetBrains) |
| Amazon Q Developer | Amazon internal models | AWS IDE plugin, CLI |
| Cursor | GPT-4 / Claude | IDE built around AI pair programming |
| Codeium | Codeium models | Free IDE plugin |

---

## Audio Generation

| Tool | Use case |
|------|---------|
| Murf.ai | Text-to-speech with natural-sounding voices |
| Amper Music | AI-generated background music from style/mood prompts |
| ElevenLabs | Voice cloning and realistic TTS |
| Suno | Full song generation from text prompts |

---

## Video Generation

| Tool | Organisation | Key capability |
|------|-------------|----------------|
| Sora | OpenAI | High-fidelity text-to-video, long clips, strong physics understanding |
| Runway Gen-3 Alpha | Runway | Text-to-video and image-to-video, fine-grained motion control |
| Kling | Kuaishou | High-quality video generation, strong motion consistency |
| Veo 2 | Google DeepMind | Cinematic quality, camera control, long duration |
| Pika 2.0 | Pika | Short video generation and editing, scene transitions |
| Wan 2.1 | Alibaba | Open-weight video generation model |

---

## Evaluation of Generative Models

| Task | Metric |
|------|--------|
| Text generation quality | Perplexity, BLEU, ROUGE, human evaluation |
| Image generation quality | FID (Fréchet Inception Distance), CLIP score |
| Image diversity | Precision/Recall in feature space |
| RAG retrieval quality | Recall@K, MRR |
| RAG answer quality | Faithfulness, answer relevance (RAGAS framework) |
