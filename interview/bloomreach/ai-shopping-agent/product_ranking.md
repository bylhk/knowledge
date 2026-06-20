# Product Ranking Model — Detail

## Objective

Rank candidate products to maximise **expected business value**, not just content relevance. The ranking model must balance:
- Does this product match what the customer asked for?
- Will this customer actually buy it?
- How much revenue/margin does it generate?

---

## Core Formula

```
Final Score = P(conversion | customer, context, product) × Expected Revenue × Relevance Weight
```

Where:
- **P(conversion)** = likelihood this customer buys this product given the current context
- **Expected Revenue** = Price × Margin%
- **Relevance Weight** = query-product match score (0-1)

---

## Architecture: Two-Stage Ranking

```
Step 6a: Hybrid Search (recall-focused)
         Vector + BM25 → ~30 candidates
                    │
                    ▼
Step 6b: Multi-Objective Ranker (precision-focused)
         ┌─────────────────────────────────────────┐
         │                                         │
         │   ┌──────────┐  ┌──────────┐  ┌─────┐   │
         │   │ Relevance│  │Conversion│  │ Rev │   │
         │   │ Model    │  │ Model    │  │     │   │
         │   └────┬─────┘  └────┬─────┘  └──┬──┘   │
         │        │             │           │      │
         │        ▼             ▼           ▼      │
         │   ┌────────────────────────────────┐    │
         │   │   Score Fusion / Combination   │    │
         │   └─────────────┬──────────────────┘    │
         │                 │                       │
         └─────────────────┼───────────────────────┘
                           ▼
                  Top 3-5 ranked products
                  (optimised for expected revenue)
```

---

## Component 1: Relevance Model (Query-Product Match)

Scores how well a product matches the customer's stated requirements.

### Features

| Feature Group | Features | Example |
|---------------|----------|---------|
| Functional match | Waterproof rating vs requirement, temp rating vs destination, wind resistance | Jacket rated -10°C for -2°C trip = good |
| Size match | Exact size availability | Men's L in stock = 1.0 |
| Brand affinity | Customer's brand purchase history | Bought Arc'teryx before → 0.9 |
| Budget fit | Price / stated budget ratio | £289 / £300 = 0.96 |
| Category match | Semantic similarity of product category to query intent | "Waterproof shell jacket" vs "waterproof jacket" = 0.95 |
| Weather suitability | Product specs vs destination weather data | Gore-Tex 3L for heavy rain = high |

### Model Options

| Model | Pros | Cons | Latency |
|-------|------|------|---------|
| **LambdaMART / XGBoost Ranker** | Interpretable, fast, handles tabular features well | Needs feature engineering | ~5ms |
| **Two-Tower (customer-query + product)** | Scales to large catalog, precompute product side | Less accurate than cross-encoder | ~5ms |
| **Cross-Encoder (BERT-style)** | Most accurate, captures nuanced match | Slow for large candidate sets | ~50ms per pair |

**Recommendation:** Two-Tower for coarse scoring (all 30 candidates), Cross-Encoder for top 10 rerank. LambdaMART as a strong baseline when hand-crafted features are available.

### Training Approach: Pairwise Learning-to-Rank

The Relevance Model is best trained with **pairwise objectives** — the model learns that "product A is more relevant than product B" rather than predicting absolute relevance scores.

**Why pairwise suits relevance:**
- No need for absolute relevance labels (hard to annotate); only relative preference is needed
- Implicit feedback naturally provides pairs: purchased > clicked > ignored
- The end goal is ranking order, not a precise score

**Pair construction from implicit feedback:**

```
Good action: purchased product / added to cart / clicked with long dwell time
Bad action:  shown but not clicked / clicked then immediately bounced

Pair: (good_product, bad_product, customer_context)
Label: score(good) > score(bad)
```

For graded relevance (purchase > add-to-cart > click > ignore), LambdaMART uses ΔNDCG weighting to leverage the gap between grades.

**Pairwise methods by model type:**

| Model | Pairwise Approach |
|-------|-------------------|
| LambdaMART (XGBoost) | Native — uses lambda gradients weighted by ΔNDCG |
| Two-Tower | Triplet loss: `max(0, margin - (score_pos - score_neg))` with anchor=customer_query embedding |
| Cross-Encoder | RankNet loss: `-log(sigmoid(score_good - score_bad))` over (query, product) pairs |

**Hard negative mining:** Select products that are close in relevance but differ in outcome (e.g., clicked but not bought vs bought). These informative negatives yield stronger gradients than random negatives.

**Sampling:** Pair count is O(n²) over candidates, so use negative sampling — random negatives for efficiency, hard negatives for quality.

---

## Component 2: Conversion Model (P(purchase))

Predicts the probability that *this customer* will buy *this product* in *this context*.

### Training Data

| Label | Source |
|-------|--------|
| Positive (1) | Product was shown + customer purchased |
| Negative (0) | Product was shown + customer did not purchase |

### Input Features

| Feature Group | Features |
|---------------|----------|
| Product features | Historical CVR, price, category, review score, return rate, stock level |
| Customer features | Purchase frequency, avg spend, brand preferences, loyalty tier, size |
| Context features | Time of day, season, destination weather, stated budget, urgency signals |
| Interaction features | Price / customer avg spend, brand = past brand?, price / budget ratio |

### Model Options

| Model | Complexity | Best for |
|-------|-----------|----------|
| **Logistic Regression** | Low | Baseline, interpretable, fast to iterate |
| **XGBoost / LightGBM** | Medium | Strong with tabular features, production-proven |
| **DeepFM** | Medium-High | Captures feature interactions automatically |
| **DCN v2 (Deep & Cross Network)** | High | Rich feature crosses without manual engineering |
| **ESMM (Entire Space Multi-Task)** | High | Jointly models click → add-to-cart → purchase funnel |
| **MMoE (Multi-gate Mixture of Experts)** | High | Multi-objective: conversion + revenue + satisfaction |

**Recommendation:** Start with XGBoost for fast iteration, graduate to DeepFM/DCN when you have enough data (>100k impressions).

### Training Approach: Pointwise Binary Classification

Unlike the Relevance Model, the Conversion Model should use **pointwise training** (standard binary classification) because the Final Score formula requires a calibrated probability value that can be meaningfully multiplied by revenue.

**Why pointwise (not pairwise) for conversion:**
- The score fusion formula `P(conversion) × Revenue` requires a real probability, not just a relative ordering
- A pairwise model outputs relative scores (e.g., 0.8 vs 0.3) with no guarantee they correspond to true conversion rates
- Binary cross-entropy naturally produces calibrated sigmoid outputs

**Training setup:**

```
Label:   shown + purchased → 1,  shown + not purchased → 0
Loss:    Binary Cross-Entropy = -[y·log(p) + (1-y)·log(1-p)]
Output:  sigmoid → P(conversion) ∈ [0, 1]
```

**Key challenges and solutions:**

| Challenge | Solution |
|-----------|----------|
| Class imbalance (CVR typically <5%) | Negative downsampling + probability recalibration, or use `pos_weight = num_neg / num_pos` |
| Position bias (top-ranked items get more clicks/purchases) | Add position as training feature, fix to constant at inference; or use inverse propensity weighting (IPW) |
| Selection bias (only shown products have labels) | Use ESMM to model full impression space, or apply causal debiasing |
| Probability calibration drift | Apply isotonic regression or Platt scaling on a held-out validation set |

**Alternative — Pairwise + post-hoc calibration:**

If ranking quality is prioritised and calibrated probabilities are still needed:
1. Train with pairwise loss (better ranking performance)
2. Apply isotonic regression on held-out data to map scores → calibrated probabilities

This is viable but adds complexity and calibration instability with small data. Recommended only at V2+ maturity when data volume is sufficient.

### Personalised Conversion

The key advantage: you're not predicting *average* conversion rate — you're predicting conversion for *this specific customer*:

```
P(conversion | customer who likes Arc'teryx, budget £300, going to Iceland)
    ≠
P(conversion | average customer)
```

A £289 Arc'teryx jacket might have 8% average CVR but 18% CVR for customers who previously bought the brand.

---

## Component 3: Expected Revenue

```
Expected Revenue = Price × Margin%
```

| Product | Price | Margin% | Expected Revenue |
|---------|-------|---------|-----------------|
| Arc'teryx Beta LT | £289 | 40% | £115.60 |
| Rab Downpour Eco | £180 | 20% | £36.00 |
| Montane Phase XT | £250 | 30% | £75.00 |
| Arc'teryx Beta AR | £320 | 45% | £144.00 |

---

## Score Fusion: Putting It Together

### Option A: Multiplicative (simple, interpretable)

```
Score = Relevance × P(conversion) × Revenue
```

| Product | Relevance | P(conv) | Revenue | Final Score |
|---------|-----------|---------|---------|-------------|
| Arc'teryx Beta LT | 0.92 | 0.12 | £115 | **12.7** |
| Rab Downpour Eco | 0.85 | 0.08 | £36 | **2.4** |
| Montane Phase XT | 0.88 | 0.15 | £75 | **9.9** |
| Arc'teryx Beta AR | 0.90 | 0.06 | £144 | **7.8** |

Winner: Beta LT — best balance of relevance, conversion likelihood, and revenue.

*Note: P(conv) values here differ from the full example below because they represent different customer contexts.*

### Option B: Weighted combination (tunable)

```
Score = α × Relevance + β × P(conversion) + γ × normalised_revenue
```

Tune α, β, γ via online A/B testing to optimise for your business KPI.

### Option C: Learned fusion (end-to-end)

Train a single model that takes relevance score, conversion probability, and revenue as additional features alongside raw product/customer features. Let the model learn optimal weighting.

---

## Handling Business Constraints

| Constraint | Implementation |
|-----------|---------------|
| Minimum relevance threshold | Filter out products with relevance < 0.5 before scoring |
| Stock-aware | Penalise or remove items with < 3 units in customer's size |
| Diversity | Don't return 3 products from the same brand — apply MMR (Maximal Marginal Relevance) |
| Price spread | Ensure at least one option well under budget (anchor) + one at/above (upsell) |
| Exploration | 10% of traffic: inject a random high-quality product to discover new winners |

---

## Exploration vs Exploitation

Pure exploitation (always show highest-scoring product) leads to feedback loops — popular products get shown more, get more conversions, get shown more...

| Strategy | How it works | When to use |
|----------|-------------|-------------|
| Epsilon-greedy | 90% exploit, 10% random from top 10 | Simple, easy to implement |
| Thompson Sampling | Sample from posterior of conversion rate | Better exploration, handles uncertainty |
| Upper Confidence Bound (UCB) | Score + confidence interval width | Good for new products with little data |
| Contextual Bandits (LinUCB) | Personalised exploration based on customer features | Most sophisticated, best long-term |

---

## Cold-Start Handling

| Scenario | Solution |
|----------|----------|
| New product (no conversion data) | Use category-level CVR + product attributes as prior |
| New customer (no history) | Fall back to population-level P(conversion), use session signals |
| New category | Transfer learning from similar categories |

---

## Training Pipeline

```
┌─────────────────┐     ┌────────────────┐     ┌──────────────┐
│ Impression Logs │───▶│ Feature Store  │────▶│ Model Train  │
│ (shown, clicked,│     │ (product +     │     │ (weekly)     │
│  purchased)     │     │  customer +    │     │              │
└─────────────────┘     │  context)      │     └──────┬───────┘
                        └────────────────┘            │
                                                      ▼
┌─────────────────┐     ┌────────────────┐     ┌──────────────┐
│ Online Serving  │◀────│ Model Registry │◀───│ Offline Eval │
│ (real-time      │     │ (versioned,    │     │ (NDCG, MAP,  │
│  inference)     │     │  A/B tested)   │     │  revenue/imp)│
└─────────────────┘     └────────────────┘     └──────────────┘
```

### Offline Metrics

| Metric | What it measures |
|--------|-----------------|
| NDCG@K | Are high-converting products ranked at the top? |
| MAP | Precision of relevant products across all positions |
| Revenue per impression | Expected £ generated per recommendation shown |
| Conversion@K | Does the top-K contain at least one purchased item? |

### Online Metrics (A/B test)

| Metric | Target |
|--------|--------|
| Conversion rate | +10-15% vs relevance-only baseline |
| Revenue per conversation | +20% |
| Average order value | Stable or +5% (from better upsell positioning) |
| Customer satisfaction | ≥ baseline (ensure not too pushy) |

---

## Model Evolution Path

```
V0 (Rule-based):      Sort by: relevance × margin (no ML)
         │
V1 (Basic ML):        XGBoost ranker with product + customer features
                       P(conversion) = historical CVR per product
         │
V2 (Personalised):    DeepFM / DCN with full feature set
                       P(conversion | customer, context, product)
         │
V3 (Multi-objective): MMoE jointly optimising conversion + revenue + CSAT
                       Contextual bandits for exploration
         │
V4 (End-to-end):      Single model: retrieval → ranking → scoring
                       Trained on business outcome (revenue)
```

---

## Example: Full Scoring for Iceland Jacket Query

**Customer:** Men's L, likes Arc'teryx, budget £300, going to Iceland (-2°C, heavy rain)

| Step | Arc'teryx Beta LT (£289) | Rab Downpour (£180) | Montane Phase (£250) |
|------|--------------------------|---------------------|----------------------|
| Relevance (query match) | 0.92 (Gore-Tex, -10°C rating, perfect for rain) | 0.85 (good waterproof, less wind protection) | 0.88 (waterproof + windproof) |
| P(conversion \| this customer) | 0.18 (brand match, budget fit, high quality) | 0.06 (customer prefers premium, this is budget) | 0.11 (decent brand, good price point) |
| Expected revenue | £115 (40% margin) | £36 (20% margin) | £75 (30% margin) |
| **Final score** | 0.92 × 0.18 × 115 = **19.0** | 0.85 × 0.06 × 36 = **1.8** | 0.88 × 0.11 × 75 = **7.3** |
| **Rank** | **#1** | #3 | #2 |

The Beta LT ranks #1 not just because it's relevant — but because *this customer* (Arc'teryx fan, premium buyer) is very likely to convert on it, and the margin is strong.
