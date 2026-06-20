# Auto Outbound Campaign — End-to-End Flow

## Overview

An automated outbound system that proactively reaches potential customers with personalised product recommendations via email/messages. Reuses core modules from the Agentic Sales AI (inbound) pipeline.

---

## Shared Modules with Agentic Sales AI

| Module | Inbound (Shopping Agent) | Outbound (This) |
|--------|--------------------------|-----------------|
| Knowledge Gathering (Step 4) | Triggered by customer question | Triggered by targeting model |
| Product Query Builder (Step 5) | Based on stated need | Based on inferred intent |
| Hybrid Search (Step 6a) | Same | Same |
| Product Ranking (Step 6b) | Same multi-objective ranker | Same — relevance × P(conv) × revenue |
| Cross-sell (Step 7) | Same | Same |
| Salesperson LLM (Step 9) | Generates chat response | Generates email / message |
| Guardrails (Step 10) | Same | Same + unsubscribe compliance |

**New modules for outbound:**
- Targeting model (who to contact)
- Trigger engine (when to send)
- Campaign LLM (email/message format instead of chat)
- Channel delivery (email API, SMS, push notification)

---

## Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. Targeting │────▶│ 2. Customer  │────▶│ 3. Product   │
│ Model        │     │ Context      │     │ Retrieval &  │
│ (who?)       │     │ (what do we  │     │ Ranking      │
│              │     │  know?)      │     │ (reuse 6a+6b)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 6. Deliver   │◀────│ 5. Guardrail │◀────│ 4. Campaign  │
│ & Monitor    │     │ & Compliance │     │ LLM Generate │
│              │     │              │     │ (email/msg)  │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

### 1. Targeting Model (Who to Contact)

Identify customers with high purchase propensity who haven't converted yet.

#### Target Segments

| Segment | Signal | Priority |
|---------|--------|----------|
| Asked but didn't buy | Had conversation with agent, no purchase within 48h | High |
| Browsed but didn't buy | Viewed product pages 3+ times, no purchase | High |
| Abandoned cart | Added to cart, didn't complete checkout | Very High |
| Lapsed buyers | Previously active, no purchase in 60+ days | Medium |
| Lookalike | Similar profile to recent converters | Medium |
| Model-scored | Propensity model predicts P(purchase) > threshold | Variable |

#### Propensity Model

```
Input features:
- Days since last visit / purchase
- Browse depth (pages viewed, categories explored)
- Previous conversion rate
- Engagement score (opened emails, clicked links)
- Cart abandonment history
- Seasonality signals (approaching trip date?)

Output:
- P(purchase in next 7 days)
- Optimal channel (email / SMS / push)
- Optimal send time
```

| Model | Use case |
|-------|----------|
| XGBoost / LightGBM | Propensity scoring (tabular features) |
| Logistic Regression | Baseline, interpretable for business |
| Uplift model (causal) | Predict *incremental* conversion from outreach (avoid annoying people who'd buy anyway) |

#### Targeting Rules (Business Logic)

- Don't contact if purchased in last 7 days
- Don't contact if unsubscribed or opted out
- Max 2 outbound messages per customer per week
- Respect quiet hours (no messages 21:00–08:00 local time)
- Cool-down: 48h after last agent conversation

---

### 2. Customer Context Assembly

For each targeted customer, gather context (reuses Step 4 from inbound):

| Source | Data | Example |
|--------|------|---------|
| CRM | Profile, size, brand preferences, loyalty tier | Men's L, prefers Arc'teryx, Gold member |
| Browsing Log | Recent pages viewed, time spent, search queries | Viewed waterproof jackets ×5, searched "Iceland gear" |
| Conversation History | Past agent chats, questions asked, objections | Asked about Beta LT, said "too expensive" |
| Purchase History | Past orders, frequency, avg spend, categories | Bought hiking boots (Mar), fleece (Jan) |
| Cart | Abandoned items | Arc'teryx Beta LT in cart (abandoned 2 days ago) |
| Weather API (if destination known) | Destination forecast | Iceland next month: -2°C, heavy rain |

#### Inferred Intent

Unlike inbound (where the customer states their need), outbound must *infer* intent:

```
Signals → Inferred intent:
- Browsed waterproof jackets + searched "Iceland" → Needs jacket for Iceland trip
- Abandoned Beta LT in cart → Still interested, price was the objection
- Bought boots + fleece but no outer layer → Building outdoor kit, missing shell
- Last purchased 90 days ago, summer approaching → Seasonal refresh
```

---

### 3. Product Retrieval & Ranking (Reuse Steps 5, 6a, 6b, 7)

Same pipeline as inbound, but the query is built from *inferred* intent rather than *stated* need:

#### Query Builder (adapted from Step 5)

| Dimension | Value | Source |
|-----------|-------|--------|
| Functional | Waterproof, windproof, cold-weather | Browsing patterns + inferred destination |
| Inventory | Men's L, in stock | CRM |
| Preference | Arc'teryx (browsed + past purchases) | Browsing + CRM |
| Budget | ~£250-330 (inferred from browsing range + past spend) | Browsing log + purchase history |
| Exclude | Items already purchased or returned | Order history |

#### Retrieval & Ranking (same as inbound)

- 6a: Hybrid search (vector + BM25) → 20-50 candidates
- 6b: Multi-objective ranking: `Relevance × P(conversion) × Revenue`
- 7: Cross-sell reranker → 1-2 complementary items

**One difference:** P(conversion) model should have an additional feature:
- `outbound_context = True` — conversion rates differ for outbound vs inbound (typically lower for outbound)

---

### 4. Campaign LLM Generates Message

Reuses the Salesperson LLM but with a different prompt template optimised for email/message format.

#### Prompt Template

```
<|system|>
You are an expert outdoor retail assistant writing a personalised outbound 
message. Generate a concise, compelling email/message that re-engages the 
customer based on their browsing and purchase history.

Rules:
- Keep subject line under 50 characters, personalised
- Email body: 3-5 sentences max
- Reference their specific browsing or past interaction naturally
- Recommend 1 primary product with key spec + price
- Include one upsell or cross-sell if relevant
- Clear CTA (link to product / reply to chat)
- Tone: helpful expert, not salesy spam
- Never lie about specs, stock, or prices
- Include unsubscribe link
<|end|>

<|context|>
Customer: Men's L, prefers Arc'teryx, past purchases: hiking boots + fleece
Trigger: Browsed waterproof jackets 5 times this week, abandoned Beta LT in cart
Inferred intent: Needs waterproof jacket, likely for upcoming trip
Last interaction: Asked agent about Beta LT 3 days ago, said "£289 is a lot"

Top product:
- Arc'teryx Beta LT — £289, Gore-Tex 3L, 360g, 4.8★ (3 left in L)

Alternative (addresses price objection):
- Rab Downpour Eco — £180, recycled, solid for 1-2 week trips

Cross-sell:
- Icebreaker Merino 200 Base Layer — £45

Campaign type: cart_abandonment
Channel: email
<|end|>
```

#### Output Formats by Channel

| Channel | Format | Length |
|---------|--------|--------|
| Email | Subject + body + CTA button | 3-5 sentences |
| SMS | Single message + link | 160 chars |
| Push notification | Title + body | 50 + 100 chars |
| WhatsApp / chat | Conversational message | 2-3 sentences |

#### Example Output (Email)

**Subject:** Still thinking about the Beta LT?

**Body:**

> Hi [Name],
>
> I noticed you were looking at the Arc'teryx Beta LT — great choice for serious rain protection. At 360g with Gore-Tex 3L, it's hard to beat for packable waterproof performance.
>
> If budget's the concern, the **Rab Downpour Eco (£180)** is a solid alternative that handles 1-2 week trips well.
>
> Either way, pairing with a **merino base layer (£45)** makes a real difference in cold rain.
>
> Only 3 left in your size (L) — [View the Beta LT →]
>
> Cheers,
> [Shop name] Gear Team

---

### 5. Guardrails & Compliance

All inbound guardrails apply, plus outbound-specific requirements:

| Rule | Implementation |
|------|---------------|
| CAN-SPAM / GDPR compliance | Unsubscribe link, sender identification, no deception |
| Frequency cap | Max 2 messages per customer per week |
| Opt-out respect | Immediately honour unsubscribe, propagate across channels |
| No fake urgency | "Only 3 left" must be real stock data |
| Price accuracy | Cross-check against catalog at send time (not build time) |
| Tone check | Not aggressive, not guilt-tripping, not manipulative |
| A/B compliance | Don't send conflicting messages to same customer |
| Quiet hours | Respect local time zones |

---

### 6. Delivery & Monitoring

#### Delivery Pipeline

```
Campaign LLM output
        │
        ▼
┌──────────────────┐
│ Guardrail check  │──── fails → discard + log
└────────┬─────────┘
         │ passes
         ▼
┌──────────────────┐
│ Schedule         │  (optimal send time per customer)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Channel router   │──── email / SMS / push / WhatsApp
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Send + log       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Track events     │  (opened, clicked, purchased, unsubscribed)
└──────────────────┘
```

#### Monitoring Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| Open rate | Subject line quality | > 25% |
| Click-through rate | Message relevance + CTA | > 5% |
| Conversion rate | Did they purchase within 7 days? | > 3% |
| Revenue per message | Business value of outbound | > £2 |
| Unsubscribe rate | Are we annoying people? | < 0.5% |
| Complaint rate | Spam reports | < 0.1% |
| Incremental lift | Uplift vs control (no message) | > 1.5× |

---

## Trigger Engine — When to Send

| Trigger | Timing | Priority |
|---------|--------|----------|
| Cart abandonment | 2h after abandonment | Very High |
| Post-conversation no-purchase | 24-48h after agent chat | High |
| Repeated browsing (no purchase) | After 3rd session viewing same category | High |
| Price drop on browsed item | Immediate (within 1h) | High |
| Back in stock (previously OOS in their size) | Immediate | High |
| Seasonal / trip approaching | 2-4 weeks before known trip date | Medium |
| Lapsed re-engagement | 60+ days since last purchase | Low |
| New arrival in preferred brand | Within 24h of product launch | Medium |

---

## Uplift Modelling (Do Not Annoy)

Not every customer benefits from outreach. Some would buy anyway; others will never buy regardless.

```
┌────────────────────────────────────────────────────┐
│           Customer Segments by Treatability          │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌────────────┐  Contact them — high uplift       │
│  │ Persuadable│  Would buy IF contacted            │
│  └────────────┘  (target these)                    │
│                                                    │
│  ┌────────────┐  Don't waste a message             │
│  │ Sure Things│  Will buy anyway without contact   │
│  └────────────┘  (save budget)                     │
│                                                    │
│  ┌────────────┐  Don't bother                      │
│  │ Lost Causes│  Won't buy even if contacted       │
│  └────────────┘  (save budget + reputation)        │
│                                                    │
│  ┌────────────┐  DO NOT contact                    │
│  │ Sleeping   │  Would buy, but contact annoys     │
│  │ Dogs       │  them into NOT buying              │
│  └────────────┘  (negative uplift — avoid!)        │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Uplift model** predicts: P(purchase | contacted) - P(purchase | not contacted)

Only send to customers where this uplift > threshold.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Auto Outbound System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────┐                                             │
│  │ Trigger Engine │  (events: cart abandon, browse, lapse...)   │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Targeting Model│  (propensity + uplift → who to contact)     │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Customer       │  ← CRM, browsing log, conversation history │
│  │ Context        │                                             │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────────────────────────────────┐                 │
│  │         REUSED FROM INBOUND AGENT          │                 │
│  │                                            │                 │
│  │  Query Builder → Hybrid Search → Ranker    │                 │
│  │  → Cross-sell → Context Assembly           │                 │
│  │                                            │                 │
│  └───────┬────────────────────────────────────┘                 │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Campaign LLM   │  (email/SMS/push — adapted Salesperson LLM) │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Guardrails +   │  (compliance, frequency cap, tone)          │
│  │ Compliance     │                                             │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Channel Router │  → Email API / SMS / Push / WhatsApp        │
│  │ + Scheduler    │                                             │
│  └───────┬────────┘                                             │
│          │                                                      │
│          ▼                                                      │
│  ┌────────────────┐                                             │
│  │ Monitor +      │  (open, click, convert, unsubscribe)        │
│  │ Feedback Loop  │  → retrain targeting + LLM                  │
│  └────────────────┘                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training the Campaign LLM

Same approach as the Salesperson LLM (see [SALESPERSON_LLM.md](SALESPERSON_LLM.md)):

| Stage | Adaptation for outbound |
|-------|------------------------|
| SFT | Train on high-performing marketing emails (opened + clicked + converted) |
| DPO | Chosen = email that led to click/purchase; Rejected = email that was ignored/unsubscribed |
| Revenue-weighted DPO | Higher weight for emails that drove larger orders |
| PPO (optional) | Reward = opened × clicked × converted × revenue - unsubscribe_penalty |

**Key DPO preference dimensions for outbound:**

| Chosen (opened + converted) | Rejected (ignored / unsubscribed) |
|----------------------------|-----------------------------------|
| Personalised subject line | Generic "Sale!" subject |
| References specific browsing | Mass blast, no personalisation |
| Addresses known objection (price) | Ignores context |
| 1 clear product + why | 10 product dump |
| Concise (3-5 sentences) | Long newsletter format |
| Helpful tone | Pushy / guilt-tripping |
| Genuine scarcity if real | Fake countdown timers |

---

## Key Differences: Inbound vs Outbound

| Aspect | Inbound (Agent) | Outbound (Campaign) |
|--------|-----------------|---------------------|
| Trigger | Customer asks a question | System identifies opportunity |
| Intent | Stated explicitly | Inferred from behaviour |
| Urgency | Customer wants answer now | Customer not expecting contact |
| Format | Chat response | Email / SMS / push |
| Length | 3-6 sentences (conversational) | Subject + 3-5 sentences (scannable) |
| CTA | "Want me to reserve it?" | "View product" link / button |
| Tone risk | Too pushy in chat | Spammy / intrusive |
| Compliance | Chat ToS | CAN-SPAM / GDPR / PECR |
| Conversion rate | ~10-15% (high intent) | ~2-5% (lower intent) |
| Volume | 1:1 (reactive) | 1:many (proactive, batch) |
