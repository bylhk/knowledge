# Fine-Tuning LLMs with LoRA & GLoRA

## What Is Fine-Tuning?

Fine-tuning adapts a pre-trained LLM to a specific task or domain by continuing training on a smaller, task-specific dataset. Full fine-tuning updates all model parameters, which is expensive. Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA dramatically reduce this cost.

## LoRA (Low-Rank Adaptation)

### Core Idea

Instead of updating all weights in a layer, LoRA freezes the original weights and injects small trainable low-rank matrices (A and B) alongside them.

For a weight matrix W (d × k):
```
W' = W + ΔW = W + B × A
```
- A: (d × r) — down-projection
- B: (r × k) — up-projection
- r (rank): typically 4–64, much smaller than d or k

### Why It Works

- Original model weights stay frozen (no catastrophic forgetting)
- Only trains r × (d + k) parameters instead of d × k
- At inference, ΔW merges into W — zero additional latency
- Multiple LoRA adapters can be swapped in/out for different tasks

### Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `r` (rank) | Rank of decomposition matrices | 8–64 |
| `lora_alpha` | Scaling factor (effective scale = alpha/r) | 16–32 |
| `lora_dropout` | Dropout on LoRA layers | 0.05–0.1 |
| `target_modules` | Which layers to apply LoRA to | q_proj, v_proj, k_proj, o_proj |

## GLoRA (Generalised LoRA)

### How It Extends LoRA

GLoRA generalises LoRA by learning an additional scaling and shifting transformation per layer, providing more expressiveness with minimal extra parameters.

```
W' = W + s * (B × A) + b
```
- `s` — learnable per-layer scale vector
- `b` — learnable per-layer bias/shift

### When to Use GLoRA Over LoRA

- Tasks requiring more nuanced per-layer adaptation
- When standard LoRA plateaus on your eval metric
- Multi-task scenarios where layers need different adaptation strengths

## Example: Fine-Tuning with LoRA (Hugging Face + PEFT)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load base model
model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# 3. Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 6.5M || all params: 7.2B || trainable%: 0.09%

# 4. Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# 6. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer.train()

# 7. Save adapter (small — typically 10-50MB)
model.save_pretrained("./lora-adapter")
```

## Example: Loading & Merging a LoRA Adapter

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype="auto",
    device_map="auto",
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Option A: Use with adapter (swappable)
output = model.generate(...)

# Option B: Merge into base weights (zero overhead at inference)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

## LoRA vs Full Fine-Tuning vs QLoRA

| Aspect | Full Fine-Tune | LoRA | QLoRA |
|--------|---------------|------|-------|
| Trainable params | 100% | ~0.1% | ~0.1% |
| GPU memory (7B) | ~60GB | ~16GB | ~6GB |
| Training speed | Slow | Fast | Fast |
| Quality | Best | Near-full | Slightly lower |
| Adapter size | Full model | 10-50MB | 10-50MB |
| Inference overhead | None | None (merged) | Quantisation cost |

## QLoRA (Quantised LoRA)

Combines 4-bit quantisation of the base model with LoRA adapters, enabling fine-tuning of 7B+ models on a single consumer GPU.

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
# Then apply LoRA as normal
```

## Example: Unsloth (Fast LoRA on Consumer GPUs)

Unsloth patches the model's attention kernels with custom Triton/CUDA code, achieving 2-5x faster training and 60% less memory than standard PEFT — no code changes needed beyond the import.

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load model with Unsloth (auto-applies optimised kernels)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-7B-v0.3",
    max_seq_length=2048,
    dtype=None,        # auto-detect
    load_in_4bit=True, # QLoRA mode
)

# 2. Add LoRA adapters (Unsloth's patched version)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
)

# 3. Prepare dataset
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# 4. Train (standard SFTTrainer — Unsloth is transparent)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./unsloth-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
    ),
)
trainer.train()

# 5. Save — supports multiple export formats
model.save_pretrained_merged("./merged-model", tokenizer)  # Full merged model
model.save_pretrained_gguf("./gguf-model", tokenizer, quantization_method="q4_k_m")  # llama.cpp
model.save_pretrained("./lora-only")  # Adapter only
```

---

## Example: LLaMA-Factory (No-Code / Low-Code Fine-Tuning)

LLaMA-Factory provides a YAML-driven interface for fine-tuning 100+ models with LoRA, QLoRA, full fine-tune, DPO, RLHF — no custom training scripts needed.

### Install

```bash
pip install llmtuner
# or clone for latest:
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .
```

### YAML Config (`examples/lora_sft.yaml`)

```yaml
### Model
model_name_or_path: mistralai/Mistral-7B-Instruct-v0.2
quantization_bit: 4  # QLoRA

### LoRA
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: q_proj,v_proj,k_proj,o_proj

### Dataset
dataset: my_custom_data          # registered in dataset_info.json
template: mistral
cutoff_len: 2048

### Training
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
output_dir: ./output/mistral-lora

### Eval
val_size: 0.05
evaluation_strategy: steps
eval_steps: 100
```

### Run Training

```bash
# CLI
llamafactory-cli train examples/lora_sft.yaml

# Or launch the Web UI
llamafactory-cli webui
```

### Register Custom Dataset (`dataset_info.json`)

```json
{
  "my_custom_data": {
    "file_name": "data/my_training_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```

### Data Format (ShareGPT style)

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What drives customer churn in subscription services?"},
      {"from": "gpt", "value": "Key factors include service quality, competitive offers, and contract length..."}
    ]
  }
]
```

### Key Features

- Supports 100+ models (LLaMA, Mistral, Qwen, Gemma, Phi, etc.)
- Training methods: SFT, DPO, RLHF, PPO, KTO, ORPO
- Web UI for no-code training and evaluation
- Built-in dataset management and formatting
- Multi-GPU with DeepSpeed integration
- Export to GGUF, vLLM, and merged formats

---

## Example: DeepSpeed (Multi-GPU / Multi-Node Scaling)

DeepSpeed is a training optimisation library from Microsoft that enables training large models across multiple GPUs/nodes with ZeRO (Zero Redundancy Optimizer) memory partitioning.

### DeepSpeed ZeRO Stages

| Stage | What's Partitioned | Memory Saving | Use Case |
|-------|-------------------|---------------|----------|
| ZeRO-1 | Optimizer states | ~4x | Multi-GPU, model fits in one GPU |
| ZeRO-2 | + Gradients | ~8x | Larger models, multi-GPU |
| ZeRO-3 | + Parameters | ~Nx (N GPUs) | Models too large for one GPU |

### DeepSpeed Config (`ds_config_zero2.json`)

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### Using DeepSpeed with Hugging Face Trainer

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    # DeepSpeed integration — just point to config
    deepspeed="ds_config_zero2.json",
)

# Then use SFTTrainer or Trainer as normal
# Launch with: deepspeed --num_gpus=4 train.py
```

### Launch Commands

```bash
# Single node, 4 GPUs
deepspeed --num_gpus=4 train.py

# Multi-node (2 nodes × 4 GPUs)
deepspeed --num_nodes=2 --num_gpus=4 \
    --hostfile=hostfile.txt \
    train.py

# With Accelerate (alternative launcher)
accelerate launch --config_file accelerate_config.yaml train.py
```

### ZeRO-3 Config (for 70B+ models)

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

### DeepSpeed + LLaMA-Factory

```yaml
### Add to LLaMA-Factory YAML config
deepspeed: examples/deepspeed/ds_z2_config.json
```

```bash
# Launch multi-GPU with LLaMA-Factory
deepspeed --num_gpus=4 src/train.py examples/lora_sft.yaml
```

---

## Tools Comparison

| Tool | Best For | GPU Requirement | Ease of Use |
|------|----------|-----------------|-------------|
| HF PEFT + TRL | Custom training loops | 1× 24GB+ | Moderate |
| Unsloth | Fast single-GPU LoRA | 1× 16GB+ | Easy |
| LLaMA-Factory | No-code / rapid experiments | 1× 16GB+ | Very easy |
| DeepSpeed | Multi-GPU / large models | Multi-GPU | Config-heavy |
| Unsloth + DeepSpeed | Fast multi-GPU | Multi-GPU | Moderate |

---

## Data Format for Fine-Tuning

```jsonl
{"text": "<s>[INST] Summarise this technical report... [/INST] The report shows...</s>"}
{"text": "<s>[INST] What features improve model accuracy? [/INST] Key features include...</s>"}
```

## Best Practices

- Start with r=16, increase only if eval metrics plateau
- Target attention layers first (q_proj, v_proj) — add more if needed
- Use QLoRA for experimentation, full LoRA for production quality
- Always evaluate on a held-out set, not just training loss
- Keep training data high-quality — garbage in, garbage out
- Monitor for overfitting (small datasets overfit fast with LoRA)
- Version your adapters — they're small enough to store many variants
