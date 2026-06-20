# Training Strategy — Agentic Sales AI

## Overview

This document covers the training strategy for all ML models in the pipeline. Each component has its own training needs:

| Component | What it learns | Training approach |
|-----------|---------------|-------------------|
| Product Ranking Model (6b) | Which products to show, optimised for revenue | Supervised learning on conversion data |
| Cross-sell Reranker (7) | Which complementary items to suggest | Collaborative filtering / learning-to-rank |
| Salesperson LLM (9) | How to present products persuasively | SFT → DPO → optional PPO |

For model-specific detail, see:
- [PRODUCT_RANKING.md](PRODUCT_RANKING.md) — ranking architecture, features, score fusion
- [SALESPERSON_LLM.md](SALESPERSON_LLM.md) — LLM training pipeline, sales framework, DPO vs PPO

---

## Training Architecture (Full System)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Data Sources                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ Impression   │  │ Conversation │  │ Expert       │             │
│  │ Logs         │  │ Logs         │  │ Annotations  │             │
│  │ (shown,      │  │ (full chat,  │  │ (ideal vs    │             │
│  │  clicked,    │  │  outcome)    │  │  poor)       │             │
│  │  purchased)  │  │              │  │              │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                  │                      │
│         ▼                 ▼                  ▼                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              Feature Store / Data Lake                │          │
│  └───┬────────────────────┬─────────────────────┬───────┘          │
│      │                    │                     │                   │
│      ▼                    ▼                     ▼                   │
│ ┌──────────┐       ┌───────────┐        ┌────────────┐            │
│ │ Ranking  │       │ Cross-sell│        │ Salesperson│            │
│ │ Model    │       │ Reranker  │        │ LLM        │            │
│ │ Training │       │ Training  │        │ Training   │            │
│ └──────────┘       └───────────┘        └────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Product Ranking Model Training

### Objective
Learn to rank products by **expected business value** = relevance × P(conversion) × revenue.

### Training Data

| Label | Source | Volume needed |
|-------|--------|--------------|
| Positive (purchased) | Impression logs: product shown → customer bought | 50k+ |
| Negative (not purchased) | Impression logs: product shown → customer didn't buy | 50k+ |
| Revenue label | Order value associated with each conversion | Same as positives |

### Feature Engineering

```
For each (customer, context, product) triple, compute:

Product features:
  - Historical CVR, price, category, review score, return rate, stock

Customer features:
  - Purchase frequency, avg spend, brand preferences, loyalty tier, size

Context features:
  - Season, destination weather, stated budget, urgency signals

Interaction features:
  - price / customer_avg_spend
  - brand == customer_past_brand?
  - price / stated_budget
  - product_temp_rating vs destination_temp
```

### Model Training Pipeline

```
Phase 1 (V0): Rule-based
  Score = relevance × margin
  No ML, fast to deploy, baseline

Phase 2 (V1): XGBoost / LightGBM
  Features: product + customer + context
  Label: purchased (1) / not purchased (0)
  Loss: LambdaRank (pairwise) or binary cross-entropy
  Retrain: weekly

Phase 3 (V2): DeepFM / DCN v2
  Same features, but model learns crosses automatically
  Requires: >100k impressions
  Retrain: weekly, warm-start from previous checkpoint

Phase 4 (V3): Multi-objective (MMoE)
  Jointly predicts: P(click), P(add-to-cart), P(purchase)
  Enables full-funnel optimisation
  Requires: event-level logging at each funnel stage
```

### Score Fusion Training

If using learned fusion (Option C from PRODUCT_RANKING.md):

```python
import lightgbm as lgb

# Features = [relevance_score, p_conversion, expected_revenue, 
#             price_budget_ratio, brand_match, stock_level, ...]
# Label = purchased (1/0), weighted by revenue

train_data = lgb.Dataset(
    features, 
    label=purchased,
    weight=revenue_if_purchased  # Revenue-weighted learning
)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [3, 5],
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_data_in_leaf': 50,
}

model = lgb.train(params, train_data, num_boost_round=500)
```

See [PRODUCT_RANKING.md](PRODUCT_RANKING.md) for full architecture, cold-start handling, and exploration strategies.

---

## (Optional) Part 2: Cross-sell Reranker Training

### Objective
Learn which complementary products to suggest alongside the primary purchase.

### Training Data

| Label | Source |
|-------|--------|
| Positive | Items bought in same order / same session as primary product |
| Negative | Items shown as cross-sell but not purchased |
| Hard negative | Items similar to primary (another jacket) — not complementary |

### Model Options (progression)

```
V0: Association Rules (Apriori / FP-Growth)
    - Mine: "jacket → base layer" with support/confidence
    - No ML, fast to deploy
    - Cold-start: rule-based bundles

V1: LambdaMART reranker
    - Features: co-purchase rate, complementarity tag, mission relevance, margin
    - Label: co-purchased (1) / not (0)
    - Fast inference (~5ms)

V2: Two-Tower model
    - Product embedding + context embedding → dot product
    - Handles large catalog, real-time scoring
    - Requires: >50k co-purchase events
```

### Key Training Signal

```python
# Positive: items bought together
# Negative: items shown but not added
# Hard negative: items from same category as primary (similar, not complementary)

training_pairs = [
    {"primary": "waterproof_jacket_123", "candidate": "base_layer_456", "label": 1},  # co-purchased
    {"primary": "waterproof_jacket_123", "candidate": "dry_bag_789", "label": 0},     # shown, not bought
    {"primary": "waterproof_jacket_123", "candidate": "rain_jacket_321", "label": 0}, # hard negative (same category)
]
```

---

## Part 3: Salesperson LLM Training

### Objective
Generate responses that maximise conversion rate and revenue while maintaining customer satisfaction.

### Three-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Stage 1: SFT          Stage 2: DPO          Stage 3: PPO (opt)  │
│  ───────────────       ──────────────         ────────────────    │
│  Learn sales style     Learn what converts    Optimise revenue    │
│                                                                    │
│  Data: 10-20k          Data: 5-50k pairs      Data: live reward   │
│  high-converting       (chosen/rejected)      signal              │
│  transcripts                                                       │
│                                                                    │
│  Duration: 2-3 epochs  Duration: 1-2 epochs   Duration: ongoing   │
│  LR: 2e-5              LR: 5e-7              LR: 1e-6            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Stage 1: SFT Data Collection

| Source | How to collect | Volume |
|--------|---------------|--------|
| Expert-written | Sales team writes ideal responses given context | 2,000-5,000 |
| Top human agents | Filter chat logs by conversion rate > 20% | 5,000-10,000 |
| Synthetic (GPT-4) | Generate with sales prompt + context, filter by quality | 10,000-20,000 |

### Stage 2: DPO Preference Pairs

#### Labelling Strategy

```
Priority 1: Outcome-labelled (strongest signal)
  chosen  = response that led to actual purchase
  rejected = response that didn't lead to purchase
  Source: production conversation logs

Priority 2: Revenue-weighted
  When both responses converted:
    chosen  = higher revenue outcome
    rejected = lower revenue outcome
  Weight in loss = revenue_difference / max_revenue

Priority 3: Expert-labelled (bootstrap)
  Sales team rates which response is better
  Use when no production data yet

Priority 4: Synthetic (weakest but scalable)
  GPT-4 generates good + bad variants
  Filter with LLM-as-judge + sales rubric
```

#### Fine-Tuning Method: LoRA / QLoRA

All LLM training stages use **LoRA** (Low-Rank Adaptation) — only a small adapter is trained while base weights stay frozen. This enables fast iteration and cheap A/B testing.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- Option A: LoRA (1× A100 40GB) ---
lora_config = LoraConfig(
    r=32,                    # rank — 32 sufficient for sales style
    lora_alpha=64,           # scaling (2× rank)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype=torch.bfloat16)
model = get_peft_model(model, lora_config)
# Trainable params: ~40M / 8B total (0.5%)


# --- Option B: QLoRA (1× 24GB GPU — RTX 4090 / A5000) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
# Same adapter, but base model in 4-bit — fits in 24GB VRAM
```

| Method | GPU | Training time (DPO, 10k pairs) | Quality |
|--------|-----|-------------------------------|---------|
| LoRA (bf16) | 1× A100 40GB | ~2h | Best |
| QLoRA (4-bit) | 1× 24GB | ~3h | ~98% of LoRA |
| Full fine-tune | 4× A100 80GB | ~6h | ~102% of LoRA (marginal gain) |

**Why LoRA for sales LLM:**
- Task is stylistic (tone, structure, persuasion) — doesn't need full weight updates
- Monthly retraining on new DPO pairs: 2h with LoRA vs 6h full fine-tune
- A/B testing: swap adapters at serving time without reloading base model
- Multiple adapters on same base: sales agent + outbound campaign + analyst

#### DPO Training Code (with LoRA)

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load SFT LoRA model (becomes reference policy)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype=torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
], task_type="CAUSAL_LM"))
model.load_adapter("./sales-agent-sft-lora", adapter_name="sft")  # load SFT adapter

# Reference model = same base + SFT adapter (frozen)
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Dataset columns: prompt, chosen, rejected
dataset = load_dataset("json", data_files="sales_preferences.jsonl")

training_args = DPOConfig(
    output_dir="./sales-agent-dpo",
    beta=0.1,                          # Start conservative
    learning_rate=5e-7,                 # Low to preserve SFT style
    num_train_epochs=2,                 # DPO overfits fast
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    max_prompt_length=1024,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./sales-agent-dpo-lora")  # saves only the adapter (~100MB)
```

#### Revenue-Weighted DPO (Phase 2)

```python
# Extend standard DPO with revenue weighting
# Higher revenue difference = stronger learning signal

def compute_pair_weights(dataset):
    """Weight each preference pair by revenue impact."""
    weights = []
    for example in dataset:
        chosen_rev = example.get("chosen_revenue", 0)
        rejected_rev = example.get("rejected_revenue", 0)
        # Normalise to [0.5, 1.0] — even low-revenue pairs contribute
        weight = 0.5 + 0.5 * (chosen_rev - rejected_rev) / max_revenue
        weights.append(weight)
    return weights

# Apply weights to DPO loss
# In practice: modify DPOTrainer or use sample_weight in dataset
```

### Stage 3: PPO with Business Reward (Optional)

```python
# Reward model trained on business outcomes
# Input: (context, response) → predicted business value

reward_signal = (
    alpha * converted          # Did they buy? (0 or 1)
    + beta * revenue           # How much revenue?
    + gamma * csat_score       # Customer satisfaction
    - delta * complaint_flag   # Penalise complaints
)

# PPO training loop (simplified)
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-6,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
)

# reward_model predicts business value of a response
# PPO optimises the policy to maximise this reward
```

See [SALESPERSON_LLM.md](SALESPERSON_LLM.md) for full progression path (Phase 1-4) and when to graduate from DPO to PPO.

---

## Data Collection Strategy (All Models)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection Timeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Month 1-2: Bootstrap                                           │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ • Synthetic DPO pairs from GPT-4 (5,000 pairs)      │       │
│  │ • Expert-written SFT data (2,000 responses)         │       │
│  │ • Rule-based ranking (no ML needed yet)             │       │
│  │ • Association rules for cross-sell                  │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  Month 3-4: Production data starts flowing                      │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ • Log all impressions + outcomes (ranking model)    │       │
│  │ • Log all conversations + purchase outcomes (DPO)   │       │
│  │ • Log cross-sell impressions + co-purchases         │       │
│  │ • Human review to correct auto-labels               │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  Month 5+: Scale + iterate                                      │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ • 10k+ outcome-labelled DPO pairs per month         │       │
│  │ • 100k+ impressions for ranking model               │       │
│  │ • Hard-case mining: where model failed to convert   │       │
│  │ • Revenue-weighted DPO pairs                        │       │
│  │ • A/B test → measure → retrain monthly              │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Framework

### Offline Metrics (all models)

| Model | Metric | Target |
|-------|--------|--------|
| Ranking | NDCG@5 | > 0.7 |
| Ranking | Revenue per impression | > £2.50 |
| Cross-sell | Co-purchase precision@2 | > 0.4 |
| Salesperson LLM | Win rate vs SFT baseline | > 65% |
| Salesperson LLM | Sales framework adherence | > 90% |
| Salesperson LLM | Factual accuracy | > 98% |

### Online Metrics (A/B test)

| Metric | Target vs baseline | Measured by |
|--------|-------------------|-------------|
| Conversion rate | +15-20% | Purchase / conversation |
| Revenue per conversation | +20-30% | Total order value / conversation |
| Upsell acceptance | +10% | Higher-priced option chosen |
| Cross-sell attachment | +25% | Add-on item purchased |
| CSAT | ≥ baseline | Post-chat survey |
| Escalation rate | ≤ baseline | Handoff to human |

### Evaluation Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Offline eval │────▶│ Shadow mode  │────▶│ A/B test     │
│ (holdout set)│     │ (log only,   │     │ (50/50 live  │
│              │     │  no serving) │     │  traffic)    │
└──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
  Gate: metrics        Gate: no             Gate: +15%
  above threshold      regressions          conversion
                       in shadow            for 2 weeks
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │ Full rollout │
                                         └──────────────┘
```

---

## Continuous Learning Loop

```
┌──────────────────────────────────────────────────────────────────┐
│                    Monthly Retrain Cycle                           │
│                                                                  │
│  Week 1: Data collection + labelling                             │
│  ├── Auto-label conversations by outcome (converted / didn't)    │
│  ├── Human review + correction (sample 10%)                      │
│  ├── Revenue-weight DPO pairs                                    │
│  └── Mine hard cases (model failed to convert)                   │
│                                                                  │
│  Week 2: Training                                                │
│  ├── Retrain ranking model (full retrain, weekly data)           │
│  ├── Update cross-sell similarity matrix                         │
│  └── DPO retrain on new pairs (incremental, from latest SFT)    │
│                                                                  │
│  Week 3: Evaluation                                              │
│  ├── Offline eval on holdout set                                 │
│  ├── Shadow mode deployment (log predictions, don't serve)       │
│  └── Compare metrics vs current production model                 │
│                                                                  │
│  Week 4: Deployment                                              │
│  ├── A/B test new model vs current (50/50 split)                 │
│  ├── Monitor conversion, revenue, CSAT for 1 week               │
│  └── If positive → promote to 100% traffic                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Requirements

| Component | Tool / Service | Purpose |
|-----------|---------------|---------|
| Feature store | Feast / Tecton / custom | Serve features at training + inference time |
| Experiment tracking | MLflow / W&B | Track hyperparameters, metrics, model versions |
| Model registry | MLflow / SageMaker | Version, stage (dev/staging/prod), rollback |
| Training compute | 1× A100 40GB (LoRA) or 1× 24GB (QLoRA) / CPU (ranking) | SFT: ~3h, DPO: ~2h, Ranking: ~30min |
| Serving | vLLM + LoRA adapter swap (LLM) / Triton (ranking) | Low-latency inference, A/B via adapters |
| A/B testing | LaunchDarkly / custom | Traffic splitting + metric collection |
| Logging | Kafka → data lake | Impression, click, purchase events |

---

## Key Principles

1. **Outcome > opinion** — label by business results (purchased, revenue), not human judgment of "which sounds better"
2. **Revenue-weight everything** — a £320 conversion should matter more than a £50 one in training signal
3. **Quality > quantity** — 5,000 well-labelled DPO pairs beat 50,000 noisy synthetic ones
4. **Test before deploy** — offline eval → shadow mode → A/B test → full rollout
5. **Retrain regularly** — models decay as inventory, trends, and customer behaviour shift
6. **Separate concerns** — ranking model optimises *what to show*, LLM optimises *how to say it*
