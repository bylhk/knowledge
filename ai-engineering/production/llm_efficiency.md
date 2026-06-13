# LLM Efficiency

## Overview

LLM efficiency techniques reduce latency, memory usage, compute cost, and token spend — making models faster, cheaper, and deployable on constrained hardware. This covers inference optimisation, quantisation, caching, batching, and architecture tricks.

---

## Quantisation

Reduce model weight precision from 16-bit to 8/4-bit, shrinking memory and speeding up inference.

### GPTQ (Post-Training Quantisation)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# Load a GPTQ-quantised model (pre-quantised from HuggingFace)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")

# Or quantise yourself
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map="auto",
)
```

### AWQ (Activation-Aware Weight Quantisation)

```python
from transformers import AutoModelForCausalLM, AwqConfig

# AWQ preserves salient weights (those that matter most for activations)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    device_map="auto",
)
```

### BitsAndBytes (Dynamic Quantisation)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantisation (used with QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)

# 8-bit (less aggressive, higher quality)
model_8bit = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    load_in_8bit=True,
    device_map="auto",
)
```

### GGUF (llama.cpp)

```python
# For CPU/hybrid inference via llama.cpp
# Download quantised model
# Q4_K_M = good balance of quality and speed
# Q5_K_M = higher quality
# Q8_0 = near-lossless

from llama_cpp import Llama

model = Llama(
    model_path="./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,  # Offload layers to GPU
    n_threads=8,
)

output = model("What is machine learning?", max_tokens=200)
```

### Quantisation Comparison

| Method | Bits | Quality Loss | Speed Gain | Memory Reduction | Best For |
|--------|------|-------------|------------|-----------------|----------|
| FP16 (baseline) | 16 | None | 1x | 1x | Reference |
| GPTQ | 4 | Minimal | ~3x | ~4x | GPU inference |
| AWQ | 4 | Minimal | ~3x | ~4x | GPU inference (slightly better quality) |
| BitsAndBytes | 4/8 | Minimal | ~2x | ~4x/2x | Fine-tuning (QLoRA) |
| GGUF Q4_K_M | 4 | Low | ~2-3x | ~4x | CPU/hybrid inference |
| GGUF Q8_0 | 8 | Negligible | ~1.5x | ~2x | Quality-sensitive CPU |

---

## KV Cache Optimisation

The key-value cache stores attention states for previously generated tokens. It grows linearly with sequence length and is the main memory bottleneck for long contexts.

### PagedAttention (vLLM)

```python
from vllm import LLM, SamplingParams

# vLLM uses paged attention — manages KV cache like virtual memory
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

params = SamplingParams(temperature=0.7, max_tokens=500)
outputs = llm.generate(["What is query optimisation?"], params)
print(outputs[0].outputs[0].text)
```

### Sliding Window Attention

Models like Mistral use sliding window attention (4096 tokens) instead of full attention, reducing KV cache from O(n²) to O(n × w).

### Grouped Query Attention (GQA)

Reduces KV cache size by sharing key-value heads across multiple query heads. Used in Llama 2 70B, Mistral, Gemma.

```
Standard MHA:  32 query heads, 32 KV heads → full KV cache
GQA:           32 query heads, 8 KV heads  → 4x smaller KV cache
MQA:           32 query heads, 1 KV head   → 32x smaller KV cache
```

---

## Inference Servers

### vLLM (High Throughput)

```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --port 8000 \
    --tensor-parallel-size 2  # Multi-GPU
```

```python
from openai import OpenAI

# vLLM exposes OpenAI-compatible API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### TGI (Text Generation Inference — HuggingFace)

```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id mistralai/Mistral-7B-Instruct-v0.2 \
    --quantize gptq \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

### Ollama (Local, Easy)

```bash
ollama pull mistral
ollama run mistral "What is machine learning?"
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", temperature=0.7)
response = llm.invoke("What is query optimisation?")
```

---

## Speculative Decoding

Use a small fast model to draft tokens, verify with the large model in parallel. Same output quality, ~2-3x faster.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Large target model
target_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map="auto")

# Small draft model (same tokenizer)
draft_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# With assisted generation (HuggingFace)
inputs = tokenizer("The transformer architecture uses", return_tensors="pt").to("cuda")
outputs = target_model.generate(
    **inputs,
    assistant_model=draft_model,
    max_new_tokens=200,
)
```

---

## Prompt Caching & Token Reduction

### Prompt Caching (API-Level)

```python
# OpenAI and Anthropic cache common prompt prefixes automatically
# Structure prompts with static system/context first, dynamic query last

# Good: static prefix is cached across requests
messages = [
    {"role": "system", "content": LONG_SYSTEM_PROMPT},      # Cached after first call
    {"role": "user", "content": FEW_SHOT_EXAMPLES},         # Cached
    {"role": "user", "content": dynamic_user_query},        # Only this varies
]
```

### Prompt Compression (LLMLingua)

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device_map="cpu",
)

compressed = compressor.compress_prompt(
    context=[long_context_1, long_context_2, long_context_3],
    instruction="Answer the question based on the context.",
    question="What techniques improve model performance?",
    rate=0.5,  # Compress to 50% of original length
)

# Use compressed prompt — fewer tokens, similar quality
print(f"Original: {compressed['origin_tokens']} tokens")
print(f"Compressed: {compressed['compressed_tokens']} tokens")
print(f"Ratio: {compressed['ratio']:.1f}x reduction")
```

### Semantic Caching

Cache LLM responses for semantically similar queries.

```python
from langchain_community.cache import RedisSemanticCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import langchain

# Queries with similar meaning hit the cache
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    score_threshold=0.95,  # How similar queries must be to hit cache
)

# First call: hits the LLM
response1 = llm.invoke("What is machine learning?")

# Second call: hits cache (semantically similar)
response2 = llm.invoke("Can you explain what ML is?")  # Cache hit!
```

---

## Batching & Continuous Batching

### Static Batching

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

# Batch multiple prompts together
prompts = [
    "Summarise: ...",
    "Translate: ...",
    "Answer: ...",
]

results = pipe(prompts, max_new_tokens=200, batch_size=len(prompts))
```

### Continuous Batching (vLLM)

vLLM automatically batches incoming requests and processes them together, maximising GPU utilisation even with varying sequence lengths.

---

## Model Distillation

Train a smaller model to mimic a larger one.

```python
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 1. Generate training data from teacher (large model)
teacher_responses = []
for prompt in training_prompts:
    response = teacher_model.generate(prompt)
    teacher_responses.append({"prompt": prompt, "response": response})

# 2. Train student (small model) on teacher outputs
student_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

trainer = SFTTrainer(
    model=student_model,
    train_dataset=teacher_dataset,
    args=TrainingArguments(output_dir="./distilled", num_train_epochs=3, bf16=True),
)
trainer.train()
```

---

## Efficient Architectures

| Technique | What It Does | Memory Saving | Speed Gain |
|-----------|-------------|---------------|------------|
| Flash Attention 2 | Fused attention kernel, tiled computation | ~2x | ~2x |
| Grouped Query Attention | Share KV heads | ~4-8x KV cache | Marginal |
| Sliding Window | Limited attention span | O(n×w) vs O(n²) | Faster for long context |
| Mixture of Experts (MoE) | Activate subset of parameters | Same | ~2-4x (fewer active params) |
| Multi-Query Attention | Single KV head | ~32x KV cache | Faster decoding |

### Flash Attention

```python
from transformers import AutoModelForCausalLM

# Enable flash attention (automatic for supported models)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map="auto",
)
```

---

## Cost Optimisation Strategies

| Strategy | Token Savings | Quality Impact | Complexity |
|----------|--------------|----------------|------------|
| Prompt caching | 50-90% on repeated prefixes | None | Low |
| Prompt compression | 30-60% | Minimal | Low |
| Smaller model for easy tasks | 5-20x cheaper | Varies | Moderate |
| Semantic caching | 100% (cache hit) | None | Moderate |
| Batching | Throughput gain | None | Low |
| Quantisation (self-hosted) | N/A (cheaper infra) | Minimal | Moderate |
| Routing (easy→small, hard→large) | 3-5x overall | Minimal | High |

### Model Routing

```python
from langchain_google_genai import ChatGoogleGenerativeAI

small_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")  # Cheap
large_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")       # Better

def route_query(query: str) -> str:
    """Route simple queries to small model, complex to large."""
    complexity_indicators = ["compare", "analyse", "explain why", "step by step", "trade-offs"]
    is_complex = any(indicator in query.lower() for indicator in complexity_indicators)

    if is_complex or len(query) > 500:
        return large_llm.invoke(query).content
    else:
        return small_llm.invoke(query).content
```

---

## Benchmarking Inference

```python
import time
import numpy as np

def benchmark_inference(model, tokenizer, prompts: list[str], max_tokens: int = 100) -> dict:
    """Measure inference speed and throughput."""
    latencies = []
    tokens_generated = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        end = time.perf_counter()

        output_len = outputs.shape[1] - input_len
        latencies.append(end - start)
        tokens_generated.append(output_len)

    total_tokens = sum(tokens_generated)
    total_time = sum(latencies)

    return {
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "tokens_per_second": total_tokens / total_time,
        "time_to_first_token_ms": latencies[0] * 1000 / tokens_generated[0] if tokens_generated[0] else 0,
    }
```

---

## Summary: When to Use What

| Goal | Technique |
|------|-----------|
| Run 7B model on laptop | GGUF Q4_K_M + llama.cpp |
| Reduce GPU memory for fine-tuning | QLoRA (4-bit + LoRA) |
| High-throughput API serving | vLLM + continuous batching |
| Faster generation | Speculative decoding, Flash Attention |
| Reduce API costs | Prompt caching, compression, routing |
| Avoid redundant LLM calls | Semantic caching |
| Deploy on edge/mobile | Distillation + quantisation |
| Long context without OOM | Sliding window, GQA, PagedAttention |

## Best Practices

- Profile before optimising — measure latency, memory, and cost first
- Quantisation is the easiest win: 4-bit GPTQ/AWQ with minimal quality loss
- Use vLLM or TGI for production serving — don't call `.generate()` directly
- Enable Flash Attention 2 always (free speed)
- Cache aggressively — most RAG queries have repeated context prefixes
- Route simple tasks to smaller/cheaper models
- Batch requests when possible — GPU utilisation often below 50% without batching
- Monitor tokens per second and time-to-first-token as key latency metrics
