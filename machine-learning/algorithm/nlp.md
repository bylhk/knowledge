# NLP Algorithms

---

## Word Embeddings

### Word2Vec

**Core idea:** train a shallow neural network to predict either a word from its context (CBOW) or context words from a target word (Skip-gram). The learned weight matrix becomes the word embeddings — words with similar meanings end up close in vector space.

**Two architectures:**

```
CBOW:     [context words] → predict → [target word]
Skip-gram: [target word]  → predict → [context words]
```

Skip-gram works better for rare words; CBOW is faster to train.

**Key property:** linear relationships in embedding space encode semantic relationships:
```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

| Pros | Cons |
|------|------|
| Fast to train, small model | Static embeddings — one vector per word regardless of context |
| Captures semantic similarity | Out-of-vocabulary words have no representation |
| Useful for downstream similarity tasks | Replaced by contextual embeddings (BERT, GPT) for most tasks |

---

### GloVe (Global Vectors)

**Core idea:** factorise the global word co-occurrence matrix. Unlike Word2Vec which uses local context windows, GloVe uses statistics from the entire corpus — how often word `i` appears in the context of word `j` across all documents.

**Objective:** learn vectors `w_i`, `w_j` such that `w_i · w_j ≈ log P(i|j)` (log co-occurrence probability).

| Pros | Cons |
|------|------|
| Uses global corpus statistics — better for rare words | Still static — no context sensitivity |
| Often outperforms Word2Vec on analogy tasks | Requires full co-occurrence matrix — memory intensive for large corpora |

**When to use Word2Vec vs GloVe:** both are largely superseded by BERT/GPT for production NLP. Use them when you need lightweight embeddings for similarity search or when compute is very limited.

---

## Transformer Architecture

Both GPT and BERT are built on the Transformer. Understanding the core components is essential.

### Self-Attention

For each token, compute a weighted sum of all other tokens' values, where weights are determined by query-key dot products:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- Q (Query), K (Key), V (Value) are linear projections of the input
- `√d_k` scaling prevents dot products from growing too large in high dimensions
- Multi-head attention runs H attention heads in parallel, each learning different relationships

### Positional Encoding

Transformers have no inherent notion of order. Positional encodings (sinusoidal or learned) are added to token embeddings to inject position information.

### RoPE (Rotary Positional Embedding)

Modern LLMs (LLaMA, Mistral, GPT-NeoX) replace absolute positional encodings with RoPE. Instead of adding a positional vector to the token embedding, RoPE rotates the query and key vectors by an angle proportional to their position before computing attention.

**Key advantage:** relative positions are encoded implicitly in the dot product — the attention score between positions `m` and `n` depends only on `m - n`, not on absolute positions. This makes RoPE naturally extensible to longer sequences than seen during training.

---

## BERT (Bidirectional Encoder Representations from Transformers)

**Core idea:** encoder-only Transformer pretrained with two objectives: Masked Language Modelling (MLM — predict masked tokens using both left and right context) and Next Sentence Prediction (NSP). Bidirectional context makes BERT strong for understanding tasks.

**Architecture:**
```
[CLS] token1 token2 ... [SEP] → Transformer Encoder × L → per-token representations
```

| Task | How to fine-tune |
|------|------------------|
| Classification | Linear head on [CLS] token |
| NER / token labelling | Linear head on each token |
| Question answering | Predict start/end span positions |

**Variants:** RoBERTa (removes NSP, larger batches), DeBERTa (disentangled attention, stronger on benchmarks), ALBERT (parameter sharing for smaller size).

**When to use:** text classification, NER, extractive QA, sentence embeddings. BERT-family models are the default encoder for understanding tasks.

---

## GPT (Generative Pretrained Transformer)

**Core idea:** decoder-only Transformer trained with causal (left-to-right) language modelling — predict the next token given all previous tokens. Fine-tuned for downstream tasks.

**Architecture:**
```
Token embeddings + Positional embeddings
  → [Masked Multi-Head Self-Attention → LayerNorm → FFN → LayerNorm] × L layers
  → Linear → Softmax → Next token probability
```

Masked self-attention ensures each token only attends to previous tokens — no future leakage during training or generation.

### GPT Training Pipeline (InstructGPT / ChatGPT)

#### 1. Pretraining
Train on a large text corpus with next-token prediction. Learns general language understanding, world knowledge, and generation capability.

#### 2. Supervised Fine-Tuning (SFT)
Fine-tune on curated (prompt, response) pairs. Teaches the model to follow instructions and produce the desired response format.

#### 3. Reward Modelling
Train a reward model on human preference pairs — see [rlhf-dpo.md](rlhf-dpo.md) for details.

#### 4. Reinforcement Learning (PPO)
Fine-tune the SFT model with PPO using the reward model signal, with a KL penalty to prevent drift from the SFT model.

---

## LLaMA / Mistral / Qwen — Open-Weight LLMs

Open-weight models have made frontier-quality LLMs accessible for fine-tuning and self-hosting.

| Model | Organisation | Key characteristics |
|-------|-------------|--------------------|
| LLaMA 3 | Meta | Strong open-weight base model, 8B–405B, RoPE, GQA |
| Mistral 7B | Mistral AI | Sliding window attention, grouped query attention — fast and efficient |
| Mixtral 8×7B | Mistral AI | MoE variant of Mistral — 8 experts, top-2 routing, matches LLaMA 2 70B at lower cost |
| Qwen 2.5 | Alibaba | Strong multilingual and coding performance, 0.5B–72B |
| Phi-3 / Phi-4 | Microsoft | Small but capable — 3.8B–8B, trained on high-quality synthetic data |
| Gemma 2 | Google | Efficient architecture, strong at small sizes (2B, 9B, 27B) |

### Key architectural innovations in modern LLMs

| Innovation | What it does |
|-----------|-------------|
| Grouped Query Attention (GQA) | Shares key/value heads across multiple query heads — reduces KV cache memory without accuracy loss |
| Sliding Window Attention (SWA) | Each token attends only to a local window — O(n) memory instead of O(n²) for long sequences |
| RoPE | Rotary positional embeddings — better length generalisation than absolute positional encodings |
| Flash Attention | IO-aware exact attention — same result as standard attention but 2–4× faster, much lower memory |
| KV Cache | Cache key/value tensors across generation steps — avoids recomputing attention for previous tokens |

---

## Fine-Tuning Techniques

### LoRA (Low-Rank Adaptation)

**Core idea:** instead of updating all model weights during fine-tuning, freeze the original weights and inject small trainable low-rank matrices into each attention layer. The number of trainable parameters drops by 10–100×.

```
Original weight: W ∈ R^(d×k)  — frozen
LoRA update:     ΔW = A · B   where A ∈ R^(d×r), B ∈ R^(r×k), r << d
Forward pass:    h = Wx + ΔWx = Wx + ABx
```

Only A and B are trained. With rank r=8 on a 7B model, trainable parameters drop from 7B to ~20M.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                          # rank
    lora_alpha=32,                # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

**QLoRA:** quantise the base model to 4-bit (NF4) and apply LoRA on top. Enables fine-tuning a 70B model on a single 48GB GPU.

| Method | VRAM for 7B fine-tune | Accuracy vs full fine-tune |
|--------|----------------------|---------------------------|
| Full fine-tune | ~80GB | Baseline |
| LoRA (r=8) | ~24GB | ~95–98% |
| QLoRA (4-bit + LoRA) | ~10GB | ~92–96% |

---

## DeepSeek — Mixture of Experts (MoE)

**Core idea:** instead of activating all model parameters for every token, a Mixture of Experts model routes each token to a small subset of specialised sub-networks (experts). Only the selected experts are computed — the rest are skipped.

**Architecture:**
```
Input token
  → Router (learned gating network) → selects top-K experts
  → Selected experts process the token in parallel
  → Weighted sum of expert outputs → next layer
```

**Sparse activation:** a model with 100 experts and top-2 routing activates only 2% of parameters per token. This allows a very large total parameter count (capacity) with the compute cost of a much smaller dense model.

**DeepSeek-specific innovations:**
- Multi-head Latent Attention (MLA) — compresses KV cache for efficient long-context inference
- Fine-grained expert segmentation — more, smaller experts rather than fewer large ones
- Auxiliary load balancing loss — prevents all tokens routing to the same few experts

| Pros | Cons |
|------|------|
| Large capacity at low compute cost per token | Complex training — load balancing is non-trivial |
| Scales to very large parameter counts | Expert routing adds latency overhead |
| Strong performance on reasoning and coding tasks | Communication overhead in distributed training |
| Open weights available | Less predictable than dense models |

**When to use:** when you need a large-capacity model but cannot afford the compute of a dense model of equivalent size. MoE is increasingly the architecture of choice for frontier models.

---

## Reasoning Models

A new class of LLMs trained to produce explicit chain-of-thought reasoning before answering. They spend more compute at inference time ("thinking") to improve accuracy on complex tasks.

| Model | Organisation | Approach |
|-------|-------------|----------|
| o1 / o3 | OpenAI | Reinforcement learning on reasoning traces — model learns when to think longer |
| DeepSeek-R1 | DeepSeek | RL-based reasoning with GRPO — open weights, matches o1 on benchmarks |
| Gemini 2.0 Flash Thinking | Google | Integrated thinking mode in Gemini |
| QwQ | Alibaba | Open-weight reasoning model |

**When to use:** mathematics, coding, multi-step logical reasoning, scientific problem solving. Not needed for simple classification or extraction tasks — the extra compute is wasted.

---

## NLP Tasks

### Text Classification

Map a text input to a discrete label. Fine-tune a pretrained encoder (BERT, RoBERTa) by adding a classification head on the [CLS] token:

```
Input → BERT encoder → [CLS] representation → Linear → Softmax → Label
```

### Named Entity Recognition (NER)

Label each token with an entity type (person, organisation, location, etc.). Fine-tune with a token-level classification head:

```
Input → BERT encoder → per-token representations → Linear → BIO tags per token
```

### Summarisation

Encoder-decoder models (T5, BART) encode the source document and decode a shorter summary autoregressively.

### Translation

Same encoder-decoder architecture as summarisation. The encoder processes the source language; the decoder generates the target language with cross-attention to the encoder output.

### Question Answering

Extractive QA: given a context passage and a question, predict the start and end token positions of the answer span within the passage.

```
[CLS] question [SEP] context [SEP] → BERT → start logits, end logits per token
```

---

## Data Augmentation for NLP

| Technique | What it does | When to use |
|-----------|-------------|------------|
| Synonym replacement | Replace words with synonyms from WordNet | Low-resource classification |
| Random insertion | Insert a random synonym of a non-stopword | Low-resource tasks |
| Random swap | Swap two random words in the sentence | Low-resource tasks |
| Random deletion | Delete each word with probability p | Low-resource tasks |
| Back-translation | Translate to another language and back | When translation models are available |
| Paraphrase generation | Generate paraphrases using a seq2seq model | High-quality augmentation |

```python
import random

def synonym_replacement(
    tokens: list[str],
    synonyms: dict[str, list[str]],
    n: int = 1,
) -> list[str]:
    """Replace n random non-stopword tokens with a synonym."""
    augmented = tokens.copy()
    candidates = [i for i, t in enumerate(tokens) if t in synonyms]
    for idx in random.sample(candidates, min(n, len(candidates))):
        augmented[idx] = random.choice(synonyms[tokens[idx]])
    return augmented
```
