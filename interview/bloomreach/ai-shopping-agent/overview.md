# Agentic Sales AI — End-to-End Flow

## Related Documents

- [PRODUCT_RANKING.md](PRODUCT_RANKING.md) — Multi-objective ranking model (relevance × conversion × revenue)
- [SALESPERSON_LLM.md](SALESPERSON_LLM.md) — Sales LLM design, DPO training, revenue maximisation
- [TRAINING.md](TRAINING.md) — Full training strategy for all models in the pipeline

## Example Scenario

> **Customer** (previous purchases: hiking boots, fleece; size: Men's L; browses outdoor gear)
>
> "I'm going to Iceland next month, I need a waterproof jacket, budget £300"

---

## Flow

### 1. User Query
Customer asks: *"I'm going to Iceland next month, I need a waterproof jacket, budget £300"*

### 2. Pre-processing Guardrail
- Toxicity / PII / off-topic filter
- Block or sanitise before entering any downstream processing
- If blocked → generic safe response + log

### 3. Intent Classification
- Classify intent → **Product Recommendation** (vs Support / Returns / FAQ)
- Extract entities:
  - Destination: Iceland
  - Timeframe: next month
  - Category: waterproof jacket
  - Budget: £300
- Route to Sales Orchestrator

### 4. Sales Orchestrator — Knowledge Gathering (parallel)
*(Entered only if intent = Product Recommendation)*

| Tool | Query | Result (example) |
|------|-------|-------------------|
| Weather API | Iceland, next month | -2°C avg, heavy rain, 40km/h wind |
| Browsing Log | Customer session history | Viewed men's L jackets, gore-tex gear |
| CRM | Customer profile | Men's L, past purchases: hiking boots + fleece, prefers Arc'teryx & Rab |

### 5. Build Product Query
Combine knowledge into a structured retrieval query:

| Dimension | Value | Source |
|-----------|-------|--------|
| Functional | Waterproof, windproof, suitable for -2°C | Weather tool + user intent |
| Inventory | Men's size L, in stock | Browsing log + CRM |
| Preference | Arc'teryx, Rab (favoured brands) | CRM |
| Budget filter | ≤ £330 (£300 stated + £30 upsell buffer) | User intent + upsell policy |

**Fallback logic:** If zero results → relax in order: brand preference → upsell buffer (+£50) → size availability (suggest nearby sizes)

### 6. Two-Stage Retrieval

#### 6a. Hybrid Search
- **Vector search**: Semantic similarity to query (e.g. "waterproof jacket for extreme cold and rain")
- **BM25**: Keyword match on specs (gore-tex, waterproof rating, men's L)
- Deduplicate results from both indexes
- Output: ~20-50 candidate products

#### 6b. Product Ranking Model (Multi-Objective)
Details: [PRODUCT_RANKING.md](PRODUCT_RANKING.md)

Rank candidates by expected business value:

```
Final Score = Relevance × P(conversion | customer, context) × Expected Revenue
```

| Component | What it scores | Model |
|-----------|---------------|-------|
| Relevance | Query-product match (specs, size, brand, weather fit) | Two-Tower / Cross-Encoder |
| P(conversion) | Will *this customer* buy *this product* in *this context*? | XGBoost → DeepFM/DCN |
| Expected Revenue | Price × Margin% | Lookup |

Example scoring:

| Product | Relevance | P(conv) | Revenue | Final Score |
|---------|-----------|---------|---------|-------------|
| Arc'teryx Beta LT (£289) | 0.92 | 0.18 | £115 | **19.0** |
| Montane Phase XT (£250) | 0.88 | 0.11 | £75 | **7.3** |
| Rab Downpour Eco (£180) | 0.85 | 0.06 | £36 | **1.8** |

Output: Top 3-5 ranked products (optimised for expected revenue, not just relevance)

### 7. Cross-sell Retrieval (Reuses Step 6 Infrastructure)

Once top products are selected from step 6b, run a cross-sell recommender to suggest complementary items. Reuses the same hybrid search pipeline with a modified query and a cross-sell reranker:

| Dimension | Value | Source |
|-----------|-------|--------|
| Functional | Layering, thermal, accessories for -2°C + rain | Weather tool + primary product context |
| Exclude | Jackets (primary category already fulfilled) | Step 6 output |
| Exclude | Items customer already owns (hiking boots, fleece) | CRM / purchase history |
| Inventory | Men's L (where applicable), in stock | Browsing log + CRM |
| Budget | Reasonable add-on range (≤ £80) | Policy |

#### Example

| Signal | Weight | Example |
|--------|--------|---------|
| Complementarity | High | Base layer pairs with shell jacket (layering system) |
| Co-purchase rate | High | 65% of jacket buyers also buy this |
| Mission relevance | High | Useful for -2°C Iceland trip specifically |
| Customer gap | Medium | Customer doesn't own a base layer yet |
| Margin | Medium | £45 item with 50% margin |
| Review score | Low | 4.6★ |

Output: Top 1-2 cross-sell items

**Fallback:** If reranker has cold-start issues (new product, no co-purchase data), fall back to rule-based bundles:
- Shell jacket → base layer
- Boots → merino socks
- Waterproof jacket → dry bag

#### Example:
- Primary product: Arc'teryx Beta LT (waterproof jacket)
- Cross-sell query: "thermal layering for -2°C Iceland hiking" (exclude jackets, exclude boots)
- Hybrid search returns: base layers, gloves, neck gaiters, thermal leggings, dry bags
- Cross-sell reranker scores:
  - Merino base layer — complementarity: high, co-purchase: 65%, mission: high → **selected**
  - Thermal leggings — complementarity: med, co-purchase: 28%, mission: high
  - Dry bag — complementarity: low, co-purchase: 35%, mission: med
- Final cross-sell: **Icebreaker Merino 200 Base Layer (£45)**

### 8. Context Assembly
Merge into a single LLM prompt:

```
Customer context:
- Size: Men's L
- Favours: Arc'teryx, Rab
- Past purchases: hiking boots, fleece
- Budget: £300 (flexible to ~£330)

Destination context:
- Iceland, next month
- Temperature: -2°C, heavy rain, strong wind

Ranked products:
1. Arc'teryx Beta LT — £289, Gore-Tex 3L, 360g, 4.8★
2. Rab Downpour Eco — £180, recycled, breathable, 4.5★
3. Montane Phase XT — £250, windproof + waterproof, 4.6★
4. Arc'teryx Beta AR — £320 (upsell), Gore-Tex Pro, more durable, 4.9★

Cross-sell:
- Icebreaker Merino 200 Base Layer — £45, complementarity: high, co-purchase: 65%

Instructions:
- Recommend top pick with reasoning tied to weather + customer history
- Offer one upsell only if genuinely better for the use case
- Suggest one cross-sell from browsing/purchase history
- Include a clear call to action
- Do not exceed stated budget by more than 10% without justification
```

### 9. Salesperson LLM Generates Response
Details: [SALESPERSON_LLM.md](SALESPERSON_LLM.md) | Training: [TRAINING.md](TRAINING.md)

The DPO-trained sales model generates a personalised, converting response following the sales framework (hook → recommend → justify → upsell → cross-sell → CTA):

> Iceland next month will be around -2°C with heavy rain and strong wind — you'll want serious 3-layer waterproof protection.
>
> My top pick for you: **Arc'teryx Beta LT (£289)** — Gore-Tex 3L, 360g, packs small, and handles exactly these conditions. Since you've gone with Arc'teryx before, you'll know the fit.
>
> Worth knowing: the **Beta AR (£320)** uses Gore-Tex Pro — more durable membrane if you're planning multi-day hikes in sustained rain. £31 over budget but built to last twice as long.
>
> I'd also pair it with a **merino base layer (£45)** — makes a big difference at -2°C under a shell.
>
> Want me to reserve the Beta LT in your size? Only 3 left in L.

**Domain-specific guardrails at this stage:**
- Don't contradict product specs (check against catalog)
- Don't recommend out-of-stock items
- Don't exceed upsell threshold without justification
- Tone: expert + warm, never pushy

### 10. Post-processing Guardrail
- Factual accuracy check: product names, prices, specs match catalog
- Compliance: no misleading claims, no fake scarcity
- Tone check: not aggressive or manipulative
- If fails → regenerate or fall back to safe template

### 11. Response Delivered to Customer

### 12. Monitoring & Logging

| Metric | Purpose |
|--------|---------|
| Conversion rate | Did the customer buy? |
| Upsell acceptance | Did they take the higher-priced option? |
| Cross-sell rate | Did they add the suggested extra item? |
| Avg order value | Revenue per conversation |
| Escalation rate | How often does the AI hand off to human? |
| Latency (p95) | End-to-end response time |
| Guardrail block rate | False positive detection |
| Customer satisfaction | Post-interaction rating |

---

## Escalation Path

```
If at any point:
  - Customer expresses frustration or dissatisfaction → hand off to human agent
  - Confidence < threshold on intent or product match → hand off
  - Customer explicitly requests human → immediate transfer
  - 3+ turns without progress → offer human option
```

