# Salesperson LLM — Detail

## Purpose

The Salesperson LLM is the final generation step — it takes structured context (ranked products, customer profile, weather, cross-sell items) and produces a natural, persuasive, converting response. It's not a general-purpose LLM; it's fine-tuned specifically to sell like an expert retail assistant.

---

## How It Differs from a General LLM

| Aspect | General LLM | Salesperson LLM |
|--------|-------------|-----------------|
| Objective | Be helpful and informative | Generate revenue while being helpful |
| Tone | Neutral, balanced | Warm expert — confident but not pushy |
| Product mention | May or may not name specifics | Always names products, prices, key specs |
| Personalisation | Generic | References customer history, size, preferences |
| Structure | Free-form | Follows a sales framework (recommend → justify → upsell → CTA) |
| Upsell behaviour | Won't suggest spending more | Strategically positions upgrades with justification |
| Call to action | Rarely includes one | Always ends with a clear next step |

---

## Training Pipeline

### Fine-Tuning Approach: LoRA / QLoRA

All stages (SFT, DPO, PPO) use **LoRA** (Low-Rank Adaptation) rather than full fine-tuning. Only a small adapter (~0.5-1% of parameters) is trained while the base model weights stay frozen.

| Approach | Parameters updated | GPU required | Training time (8B) |
|----------|-------------------|-------------|-------------------|
| Full fine-tune | All 8B | 4× A100 80GB | 8-12h (SFT), 4-8h (DPO) |
| **LoRA** | ~40-80M (rank 32-64) | 1× A100 40GB | 2-4h (SFT), 1-3h (DPO) |
| **QLoRA** | ~40-80M (4-bit base) | 1× 24GB GPU | 3-5h (SFT), 2-4h (DPO) |

**Why LoRA for this use case:**
- The task is *stylistic* (sales framework, tone, persuasion) — LoRA captures this well
- Lower risk of catastrophic forgetting (base knowledge preserved)
- Cheap A/B testing: swap adapters to compare DPO variants without duplicating the full model
- Fast iteration: retrain monthly on new preference pairs in hours, not days

**LoRA Configuration:**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                    # rank — 32 is sufficient for style tasks
    lora_alpha=64,           # scaling factor (typically 2× rank)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**QLoRA Configuration (single 24GB GPU):**

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base model in 4-bit, train LoRA adapter on top
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
)
model = get_peft_model(model, lora_config)
```

**Serving with Adapters:**

```
Base model (shared, loaded once)
    │
    ├── Sales SFT adapter (v1)
    ├── Sales DPO adapter (v2) ← production
    ├── Sales DPO adapter (v3) ← A/B test variant
    └── Outbound campaign adapter (v1) ← reuses same base
```

Multiple adapter versions can be hot-swapped without reloading the base model — enables fast A/B testing of different DPO training runs.

### Pipeline Stages

```
┌──────────────────┐
│ Base Model       │  (Llama 3 8B / Mistral 7B / GPT-4 distilled)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 1: SFT     │  Learn sales conversation style
│ (LoRA r=32)      │  Data: curated high-converting transcripts
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 2: DPO     │  Learn what converts vs what doesn't
│ (LoRA r=32)      │  Data: (prompt, chosen, rejected) pairs
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 3 (opt):   │  Fine-tune on business outcome signal
│ RLHF / PPO       │  Reward = actual conversion + CSAT
│ (LoRA r=16)      │  Lower rank — smaller refinement on top of DPO
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sales Agent LLM  │  Production model
└──────────────────┘
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

### Goal
Teach the model the *style* of a great salesperson — how to structure responses, when to upsell, how to reference customer context.

### Training Data

| Source | Volume | Quality |
|--------|--------|---------|
| Expert-written sales responses | 2,000-5,000 | High — gold standard |
| Top-performing human agent transcripts | 5,000-10,000 | High — real conversions |
| GPT-4/Claude generated (with sales prompt) | 10,000-20,000 | Medium — needs filtering |

### Data Format

```json
{
  "system": "You are an expert outdoor retail assistant. Use customer context and product data to make personalised recommendations that convert.",
  "context": {
    "customer": { "size": "L", "gender": "M", "brand_pref": ["Arc'teryx"], "past_purchases": ["hiking boots", "fleece"] },
    "weather": { "destination": "Iceland", "temp": -2, "conditions": "heavy rain, strong wind" },
    "ranked_products": [ ... ],
    "cross_sell": [ ... ]
  },
  "user": "I'm going to Iceland next month, I need a waterproof jacket, budget £300",
  "assistant": "Iceland next month will be around -2°C with heavy rain and strong wind — you'll want serious 3-layer waterproof protection.\n\nMy top pick for you: Arc'teryx Beta LT (£289) — Gore-Tex 3L, 360g, packs small..."
}
```

### What the Model Learns in SFT
- Response structure (recommend → explain why → upsell → cross-sell → CTA)
- How to weave in weather context naturally
- How to reference customer history without being creepy
- Product presentation format (name, price, key spec, benefit)
- Appropriate upsell framing

---

## Stage 2: DPO (Direct Preference Optimization)

### Goal
Sharpen the model's judgment on *which* response style converts better.

### Preference Pair Design

Each pair has the same input context but two different responses:

```
┌─────────────────────────────────────────────────┐
│ Same prompt + context                           │
├─────────────────────────────────────────────────┤
│ Chosen (converts):                              │
│ • Specific product + price                      │
│ • Weather-justified recommendation              │
│ • Natural upsell with reason                    │
│ • Cross-sell tied to customer history           │
│ • Scarcity signal + CTA                         │
├─────────────────────────────────────────────────┤
│ Rejected (doesn't convert):                     │
│ • Generic product list                          │
│ • No personalisation                            │
│ • No upsell or pushy upsell                     │
│ • No CTA                                        │
│ • No weather/context reasoning                  │
└─────────────────────────────────────────────────┘
```

### Preference Dimensions

| Dimension | Chosen demonstrates | Rejected lacks |
|-----------|-------------------|----------------|
| Personalisation | "Since you bought Arc'teryx before..." | Generic to any customer |
| Contextual reasoning | "Iceland at -2°C with rain means..." | No mention of conditions |
| Specificity | "Beta LT, £289, Gore-Tex 3L, 360g" | "Here are some jackets" |
| Justified upsell | "£30 more gets Pro membrane for multi-day hikes" | No upsell or "want something more expensive?" |
| Cross-sell | "Pair with merino base layer for -2°C" | Single product only |
| Scarcity | "Only 3 left in L" (when true) | No urgency |
| CTA | "Want me to reserve it?" | "Let me know!" |
| Objection handling | Addresses concern with data | Deflects or agrees |

### DPO Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| β (beta) | 0.1 – 0.3 | Start low; higher = more conservative |
| Learning rate | 5e-7 | Lower than SFT to preserve learned style |
| Epochs | 1–2 | DPO overfits fast |
| Dataset size | 5,000–50,000 pairs | Quality > quantity |

See [TRAINING.md](TRAINING.md) for full DPO code and pipeline.

---

## Stage 3 (Optional): RLHF / PPO with Business Reward

### Goal
Directly optimise for business outcomes — not just human preference labels, but actual conversion and revenue.

### Reward Signal

```
Reward = α × converted(0/1) + β × revenue_generated + γ × CSAT_score - δ × complaint_flag
```

| Component | Weight | Why |
|-----------|--------|-----|
| Conversion | High | Core business goal |
| Revenue | Medium | Prefer higher-margin sales |
| CSAT | Medium | Ensure customer satisfaction doesn't drop |
| Complaint penalty | High (negative) | Avoid pushy/misleading behaviour |

### When to Use PPO vs DPO
- **DPO**: Sufficient when you have good preference pairs and want offline training
- **PPO**: Better when you can build a reward model from real business outcomes (conversion, revenue, CSAT)
- **Practical**: Start with DPO, add PPO later when you have enough production data to train a reward model

---

## Input Prompt Template

The model receives a structured prompt at inference:

```
<|system|>
You are an expert outdoor retail sales assistant. Generate a personalised 
product recommendation that maximises purchase likelihood. Follow the 
sales framework: recommend → justify → upsell → cross-sell → CTA.

Rules:
- Always name specific products with prices
- Justify recommendations with weather/context data
- Reference customer history naturally
- Upsell only if genuinely better for the use case (max 10% over budget)
- Include one cross-sell suggestion
- End with a clear call to action
- Never lie about specs, stock, or prices
- Never be pushy — if context suggests customer is price-sensitive, lead with value
<|end|>

<|context|>
Customer: Men's L, prefers Arc'teryx & Rab, past purchases: hiking boots + fleece
Destination: Iceland, next month, -2°C, heavy rain, 40km/h wind
Budget: £300 (flexible to £330)

Ranked products:
1. Arc'teryx Beta LT — £289, Gore-Tex 3L, 360g, waterproof 28k mm, windproof, 4.8★ (3 left in L)
2. Montane Phase XT — £250, waterproof + windproof, 420g, 4.6★ (8 in stock)
3. Rab Downpour Eco — £180, recycled, 15k waterproof, breathable, 4.5★
4. Arc'teryx Beta AR — £320 (upsell), Gore-Tex Pro, 490g, most durable, 4.9★ (5 left)

Cross-sell:
- Icebreaker Merino 200 Base Layer — £45, 4.7★, co-purchase 65%
<|end|>

<|user|>
I'm going to Iceland next month, I need a waterproof jacket, budget £300
<|end|>

<|assistant|>
```

---

## Sales Framework Encoded in the Model

The DPO training reinforces this response structure:

```
┌─────────────────────────────────────────────────────────┐
│ 1. HOOK — Acknowledge context (shows expertise)          │
│    "Iceland next month = -2°C, heavy rain, strong wind"  │
├─────────────────────────────────────────────────────────┤
│ 2. RECOMMEND — Top pick with reasoning                   │
│    "My pick: Beta LT (£289) — Gore-Tex 3L, 360g"       │
│    "Since you've gone with Arc'teryx before..."          │
├─────────────────────────────────────────────────────────┤
│ 3. JUSTIFY — Why this product for THIS situation         │
│    "28k waterproof rating handles Iceland's sideways     │
│     rain, and at 360g it packs without bulk"             │
├─────────────────────────────────────────────────────────┤
│ 4. UPSELL — One upgrade, justified by use case           │
│    "If you're doing multi-day hikes, the Beta AR (£320) │
│     has Gore-Tex Pro — lasts 2× longer in sustained     │
│     rain. £31 over budget but built for abuse."          │
├─────────────────────────────────────────────────────────┤
│ 5. CROSS-SELL — Complementary item, tied to context      │
│    "I'd pair it with a merino base layer (£45) —        │
│     makes a real difference at -2°C under a shell."     │
├─────────────────────────────────────────────────────────┤
│ 6. CTA — Clear next action + scarcity if genuine         │
│    "Want me to reserve the Beta LT in your size?        │
│     Only 3 left in L."                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Domain-Specific Guardrails (within the LLM step)

These are enforced at generation time, separate from the general post-processing guardrail:

| Rule | Implementation |
|------|---------------|
| Price accuracy | Cross-check generated prices against product data in context |
| Spec accuracy | Model must only state specs present in the context window |
| Stock claims | "Only 3 left" must match actual stock data provided |
| Budget respect | Upsell max 10% over stated budget unless customer signals flexibility |
| No fake scarcity | Only mention low stock if stock < 5 units |
| No brand disparagement | Never trash a competitor to sell your product |
| Tone boundary | Confident expert, not aggressive salesperson |

---

## Multi-Turn Conversation Handling

The salesperson LLM also handles follow-ups:

| Customer says | Model behaviour |
|---------------|-----------------|
| "That's too expensive" | Offer the next-best option at lower price, explain trade-offs |
| "Tell me more about the Beta LT" | Deep-dive on specs, reviews, use cases — build confidence |
| "What about waterproof trousers too?" | Trigger cross-sell retrieval for trousers, add to conversation |
| "I'll take it" | Confirm size, move to checkout flow, suggest add-ons |
| "I'm not sure yet" | No pressure — offer to save selection, remind of stock level gently |
| "Can I speak to someone?" | Immediate escalation to human agent |

### Conversation State Management

```
┌────────────────────────────┐
│ Conversation Memory         │
│                            │
│ • Products discussed: [..] │
│ • Objections raised: [..]  │
│ • Budget signals: [..]     │
│ • Intent strength: 0-1     │
│ • Turn count: N            │
│ • Escalation risk: low/med │
└────────────────────────────┘
```

Each turn re-evaluates whether to:
- Continue selling (intent strong, no frustration)
- Offer alternatives (objection raised)
- Back off (customer uncertain, don't push)
- Escalate (frustration detected, 3+ turns without progress)

---

## Evaluation

### Offline (before deployment)

| Test | Method |
|------|--------|
| Sales framework adherence | LLM-as-judge: does response follow hook → recommend → justify → upsell → CTA? |
| Factual accuracy | Automated: compare generated specs/prices against product catalog |
| Tone appropriateness | LLM-as-judge + human review: expert but not pushy |
| Personalisation depth | Automated: does response reference ≥2 customer signals? |
| Win rate vs baseline | Human blind eval: DPO model vs SFT-only model |

### Online (A/B test in production)

| Metric | Target vs SFT baseline |
|--------|----------------------|
| Conversion rate | +15-20% |
| Upsell acceptance | +10% |
| Cross-sell attachment | +25% |
| CSAT (post-chat survey) | ≥ baseline (no regression) |
| Avg response length | -10% (more concise = better) |
| Escalation rate | ≤ baseline |

---

## Model Selection

| Option | Size | Fine-tune method | Pros | Cons |
|--------|------|-----------------|------|------|
| Llama 3 8B + LoRA | 8B | LoRA r=32 | Fast inference, cheap training (1× A100), adapter swapping for A/B | Limited reasoning on edge cases |
| Llama 3 8B + QLoRA | 8B | QLoRA (4-bit + LoRA) | Train on single 24GB GPU, cheapest option | Slightly lower quality than full LoRA |
| Mistral 7B + LoRA | 7B | LoRA r=32 | Strong instruction following, concise | Similar to Llama 3 |
| Llama 3 70B + LoRA | 70B | LoRA r=16 | Better multi-turn, nuanced objection handling | Needs 2× A100 for LoRA training |
| GPT-4 (prompted, no fine-tune) | — | None | Best quality, no training needed | Expensive, no DPO control, vendor lock-in |
| Distilled GPT-4 → 8B + LoRA | 8B | LoRA r=64 | GPT-4 quality at small model cost | Requires distillation pipeline |

**Recommendation:** Start with Llama 3 8B + QLoRA for SFT + DPO (trains on a single 24GB GPU). Graduate to full LoRA on A100 when traffic justifies the cost. Use GPT-4 as a teacher model for generating SFT data and DPO preference labels.

### When to Use Full Fine-Tune Instead of LoRA

| Scenario | Use full fine-tune |
|----------|-------------------|
| Distilling from GPT-4 (large knowledge transfer) | Yes — LoRA may not capture enough |
| Very large SFT dataset (>50k examples) | Consider it — more capacity needed |
| Maximum quality, budget not a constraint | Yes — ~5% better on benchmarks |
| Need domain knowledge injection (not just style) | Yes — base weights need updating |

For pure style/tone/structure tasks like the salesperson LLM, LoRA is sufficient and preferred.

---

## Can DPO Maximise Conversion Rate and Sales?

### Short Answer

DPO **can** learn to maximise conversion — but only if your preference pairs are labelled by **actual business outcomes** (purchased vs didn't purchase), not by human opinion of "which sounds better."

DPO alone is **limited** for maximising revenue because it treats all conversions equally (a £50 sale = a £320 sale).

---

### How DPO Learns to Convert

DPO doesn't directly optimise a conversion metric. It learns from preference pairs:

```
If your labels are:
  chosen  = response that led to purchase ✓
  rejected = response that didn't lead to purchase ✗

Then DPO implicitly learns to maximise conversion.
```

| Strength | Why |
|----------|-----|
| Learns *style* that converts | "Specific product + CTA" beats "generic list" — DPO captures this |
| No reward model needed | Simpler than RLHF/PPO, fewer moving parts |
| Offline training | Train on historical data without online exploration |
| Multi-dimensional preference | Tone + specificity + personalisation all in one signal |

---

### Where DPO Falls Short for Revenue Maximisation

| Limitation | Impact |
|-----------|--------|
| Binary signal only | Can't distinguish "£289 sale" from "£320 sale" — both are just "chosen" |
| No revenue weighting | A £50 cross-sell and a £320 jacket look the same |
| Static preferences | Trains offline on fixed pairs — doesn't adapt to live signals |
| Correlation ≠ causation | A response before a purchase isn't necessarily *why* they purchased |
| No exploration | Only learns from existing data — won't discover new sales strategies |

---

### Making DPO Actually Maximise Sales

#### 1. Label by outcome, not opinion

```
✗ Human preference:  "This response sounds more professional"
✓ Business outcome:  "This response led to a £289 purchase"
```

#### 2. Revenue-weighted pair selection

When multiple "chosen" candidates exist, prioritise higher revenue:

```python
# Weight pairs by revenue difference — stronger signal for DPO loss
pair_weight = (chosen_revenue - rejected_revenue) / max_revenue

# Example:
# Pair A: chosen led to £320 sale, rejected led to nothing → weight = 1.0
# Pair B: chosen led to £180 sale, rejected led to nothing → weight = 0.56
# DPO learns more from Pair A
```

#### 3. Stratified pairs by scenario

Don't random-pair — create pairs within the same customer context:
- Same customer profile
- Same product set
- Two different response styles
- One converted, one didn't

#### 4. Hard-negative mining

Most valuable training signal comes from *close* pairs:
- Both responses are decent quality
- One converted, one didn't
- The difference teaches subtle persuasion techniques

---

### Progression Path: DPO → Revenue Optimisation

```
Phase 1: DPO with conversion-labelled pairs
         ┌──────────────────────────────────────────┐
         │ Label: converted (1) vs not converted (0) │
         │ Model learns: what style converts          │
         │ Expected lift: +15-20% conversion          │
         └──────────────────────────────────────────┘
                          │
                          ▼
Phase 2: Revenue-weighted DPO
         ┌──────────────────────────────────────────┐
         │ Pairs weighted by revenue difference       │
         │ Model learns: prefer higher-value sales    │
         │ Expected lift: +5-10% revenue (on top)     │
         └──────────────────────────────────────────┘
                          │
                          ▼
Phase 3: PPO with composite reward
         ┌──────────────────────────────────────────┐
         │ Reward = P(conv) × revenue × CSAT          │
         │ Model actively optimises business outcome  │
         │ Expected lift: +10-15% revenue (on top)    │
         │ Requires: reward model + production data   │
         └──────────────────────────────────────────┘
                          │
                          ▼
Phase 4: Online learning (continuous)
         ┌──────────────────────────────────────────┐
         │ Live A/B → new pairs → retrain monthly     │
         │ Model adapts to seasonal trends, new stock │
         │ Expected: sustained improvement over time  │
         └──────────────────────────────────────────┘
```

---

### Summary: What to Use When

| Goal | Best approach | DPO sufficient? |
|------|-------------|-----------------|
| Learn sales style (structure, tone, CTA) | DPO | ✓ Yes |
| Maximise conversion rate | DPO with outcome labels | ✓ Yes |
| Maximise revenue (not just conversion count) | Revenue-weighted DPO or PPO | ⚠️ Partial — needs weighting |
| Continuously improve from live data | Online PPO / bandit | ✗ No — DPO is offline |
| Balance revenue + customer satisfaction | Multi-objective PPO (MMoE reward) | ✗ No — DPO is single-objective |

**Bottom line:** DPO is the right starting point — it will significantly lift conversion over a generic LLM. But for true revenue maximisation, graduate to revenue-weighted DPO (Phase 2) and eventually PPO with a business reward model (Phase 3).
