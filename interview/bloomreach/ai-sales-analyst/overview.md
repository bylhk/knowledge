# Sales Analyst AI

## Overview

An AI analyst that provides sales insights via two interfaces — an interactive chatbot and scheduled HTML reports. Both share the same data pipeline, analytics engine, and LLM generation layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    INGESTION PIPELINE                          │  │
│  │                                                               │  │
│  │  Data Sources ──▶ ETL ──▶ Aggregated Tables ──▶ Analytics     │  │
│  │                                                  Engine       │  │
│  └───────────────────────────────────────┬───────────────────────┘  │
│                                          │                          │
│                              ┌───────────┴───────────┐              │
│                              │                       │              │
│                              ▼                       ▼              │
│  ┌─────────────────────────────────┐  ┌──────────────────────────┐  │
│  │          CHATBOT                │  │    SCHEDULED REPORT      │  │
│  │                                 │  │                          │  │
│  │  1. User asks question          │  │  1. Cron triggers        │  │
│  │  2. Collect data from tables    │  │     report question      │  │
│  │  3. Collect from analytics      │  │  2. Collect data         │  │
│  │     engine                      │  │  3. Collect from         │  │
│  │  4. LLM generates:             │  │     analytics engine     │  │
│  │     • text response             │  │  4. LLM generates HTML:  │  │
│  │     • chart JSON                │  │     • text + charts      │  │
│  │  5. User asks follow-up → (1)  │  │  5. Deliver (email/      │  │
│  │                                 │  │     Slack/HTML)          │  │
│  └─────────────────────────────────┘  │  6. User follow-up →     │  │
│                                       │     load checkpoint →    │  │
│                                       │     chatbot (1)          │  │
│                                       └──────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Ingestion Pipeline

### Step 1 — Extract Data Sources & Build Aggregated Tables

```
┌──────────────────────────────────────────────────────────────────┐
│                        Data Sources                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Sales DB │ │ Web      │ │ Product  │ │ Agent    │           │
│  │ / POS    │ │ Analytics│ │ Catalog  │ │ Convos   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │             │            │             │                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Weather  │ │ Calendar │ │ Events / │ │Competitor│           │
│  │ API      │ │ (school, │ │ Festivals│ │ Monitor  │           │
│  │          │ │  holidays)│ │          │ │          │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │             │            │             │                 │
│       └─────────────┴────────────┴─────────────┘                 │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                            │
│                    │      ETL        │  (daily + real-time)       │
│                    └────────┬────────┘                            │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                            │
│                    │  Aggregated     │                            │
│                    │  Tables         │                            │
│                    └─────────────────┘                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Aggregated Tables

| Table | Grain | Contents | Refresh |
|-------|-------|----------|---------|
| `daily_revenue` | Day × category × brand | Revenue, units, orders, returns, margin | Daily |
| `daily_traffic` | Day × category × source | Sessions, page views, bounce rate, searches | Daily |
| `daily_conversion` | Day × category | CVR, AOV, cart abandonment rate | Daily |
| `product_performance` | Day × SKU | Revenue, units, stock level, return rate, review score | Daily |
| `customer_metrics` | Day | New vs returning, avg spend, loyalty breakdown | Daily |
| `weather_data` | Day × region | Temperature, conditions, forecast (7-14 days ahead) | Daily |
| `calendar_events` | Day | School terms, bank holidays, festivals, local events | Weekly |
| `competitor_prices` | Day × SKU (matched) | Competitor price, promotions, availability | Daily |
| `agent_performance` | Day | Conversations, agent CVR, top queries, escalation rate | Daily |

### Step 2 — Analytics Engine

Runs on top of aggregated tables, computes derived signals:

```
┌──────────────────────────────────────────────────────────────────┐
│                       Analytics Engine                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ METRICS                                                      ││
│  │ • KPIs: revenue, CVR, AOV, margin (with WoW, MoM, YoY)     ││
│  │ • Category breakdown + ranking                               ││
│  │ • Product-level performance (top sellers, worst performers)  ││
│  │ • Stock: days-until-OOS per SKU                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ TREND DETECTION                                              ││
│  │ • Moving average crossover (3d vs 7d vs 28d)                ││
│  │ • STL decomposition (trend + seasonality + residual)        ││
│  │ • Change point detection (PELT)                              ││
│  │ • Category velocity (acceleration / deceleration)           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ ANOMALY DETECTION                                            ││
│  │ • Z-score vs rolling baseline (flag > 2σ deviation)         ││
│  │ • Control charts (UCL / LCL breach)                         ││
│  │ • Revenue drop alerts, CVR collapse, traffic anomalies      ││
│  │ • Root cause correlation (weather? price? stock? competitor?)││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ FORECASTING                                                  ││
│  │ • 7-day / 30-day revenue forecast (Prophet / LightGBM)      ││
│  │ • Category-level demand prediction                           ││
│  │ • External signal integration (weather, calendar, events)   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Output: pre-computed analytics results stored and available for query by both chatbot and report.

---

## Part 2: Chatbot

Interactive Q&A interface for the sales team.

### Flow

```
┌─────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1.  │────▶│ 2. Collect  │────▶│ 3. Collect   │────▶│ 4. LLM      │
│User │     │ data from   │     │ from         │     │ generate    │
│asks │     │ aggregated  │     │ analytics    │     │ response    │
│     │     │ tables      │     │ engine       │     │             │
└─────┘     └─────────────┘     └──────────────┘     └──────┬───────┘
                                                             │
   ┌─────────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────┐
│ Output:                   │
│ a. Text message (insight) │
│ b. Chart JSON (rendered)  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 5. User asks follow-up   │──── loops back to Step 1
│    (multi-turn)          │
└──────────────────────────┘
```

### Step 1 — User Asks a Question

> "Why did shorts sales spike this week?"
> "What should we promote next week?"
> "Compare June vs last June"

### Step 2 — Collect Data from Aggregated Tables

LLM determines what data is needed and queries the relevant tables:

```python
# LLM generates tool calls:
get_metrics(category="shorts", period="this_week", compare="last_week")
get_metrics(category="shorts", period="last_14d", granularity="daily")
```

Returns structured data for the LLM to reason over.

### Step 3 — Collect from Analytics Engine

LLM pulls pre-computed analysis relevant to the question:

```python
trend_detect(metric="shorts_revenue", window="14d")
# → "Strong upward trend, +40% WoW, acceleration started Monday"

root_cause(observation="shorts_revenue_spike", check=["weather", "events", "competitor", "price"])
# → "Correlation: temperature r=0.92. Heatwave started Monday (28°C vs 18°C seasonal avg)"
```

### Step 4 — LLM Generates Response

The LLM produces two types of output:

**a. Text message:**
> Shorts revenue spiked +40% this week (£18,200 vs £13,000 last week).
> The primary driver is the heatwave — temperature hit 28°C and shorts sales
> correlate with temperature at r=0.92. Expect this to reverse from Friday
> when rain is forecast.

**b. Chart JSON (rendered inline):**
```json
{
  "tool": "line_chart",
  "params": {
    "title": "Shorts Revenue vs Temperature (14 days)",
    "series": [
      {"name": "Shorts (£)", "data": [800, 900, 1200, 1800, 2400, 2800, 3200, ...]},
      {"name": "Temp (°C)", "data": [16, 18, 20, 22, 24, 26, 28, ...], "axis": "right"}
    ],
    "annotations": [{"x": "Fri", "label": "🌧️ Rain forecast"}]
  }
}
```

### Step 5 — User Asks Follow-up

> "What will happen when the rain starts?"

→ Loop back to Step 1 with conversation context preserved.

---

## Part 3: Scheduled Report

Automated report generation on a schedule. Uses the **same pipeline as chatbot** but triggered by cron instead of a user.

### Flow

```
┌─────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1.Cron  │──▶│ 2. Collect  │──▶│ 3. Collect   │──▶│ 4. LLM      │
│ triggers│   │ data from   │   │ from         │   │ generate    │
│ report  │   │ aggregated  │   │ analytics    │   │ HTML report │
│ question│   │ tables      │   │ engine       │   │             │
└─────────┘   └─────────────┘   └──────────────┘   └──────┬───────┘
                                                           │
   ┌───────────────────────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────┐     ┌──────────────────────────┐
│ 5. Deliver                    │     │ 6. User wants to         │
│ • Email (HTML)               │     │    ask questions          │
│ • Slack (message + images)   │     │                          │
│ • Dashboard (interactive)    │     │ → Load checkpoint        │
│                              │     │ → Enter chatbot (Step 1) │
└──────────────────────────────┘     └──────────────────────────┘
```

### Step 1 — Scheduled Report Question

The cron job sends a pre-defined question to the same LLM pipeline:

```
Schedule: Monday 08:00
Question: "Generate the weekly sales report for the last 7 days. 
           Include: revenue summary, category performance, trends, 
           anomalies, weather impact, upcoming events, stock risks, 
           forecast, and top 3 recommended actions."
```

Different schedules, different questions:

| Schedule | Question sent to LLM |
|----------|---------------------|
| Daily 08:00 | "Daily flash: yesterday's revenue, anomalies, and today's outlook" |
| Monday 08:00 | "Full weekly report: performance, trends, root causes, forecast, actions" |
| 1st of month | "Monthly review: MoM performance, YoY comparison, trend shifts, next month forecast" |
| Real-time (alert) | "Anomaly detected: {metric} deviated by {amount}. Explain why and recommend action." |

### Step 2 & 3 — Same as Chatbot

Identical data collection and analytics engine calls. The LLM fetches everything needed to answer the report question comprehensively.

### Step 4 — LLM Generates HTML Report

Same LLM, but output format is HTML with embedded charts:

**a. Text (as HTML sections):**
```html
<section class="insight critical">
  <h3>🔴 Fleece CVR Collapsed</h3>
  <p>CVR dropped from 4.2% to 1.8% after Wednesday's price increase...</p>
</section>
```

**b. Chart JSON (rendered to images/interactive):**
```json
{
  "tool": "bar_chart",
  "params": { "title": "Category Performance WoW", ... },
  "render_as": "svg_embedded"
}
```

The chart rendering service converts JSON → SVG/PNG and embeds in the HTML.

### Step 5 — Deliver

| Channel | Format | Charts |
|---------|--------|--------|
| Email | HTML email | Inline SVG/PNG |
| Slack | Message blocks + image attachments | PNG uploaded |
| Dashboard | Interactive HTML page | Plotly/D3 (hover, zoom) |
| PDF | Rendered HTML → PDF | Static images |

### Step 6 — Checkpoint & Follow-up

When the report is delivered, a **checkpoint** is saved:

```json
{
  "checkpoint_id": "report_2024_06_18_weekly",
  "timestamp": "2024-06-18T08:00:00Z",
  "question": "Weekly sales report for 12-18 June",
  "data_snapshot": { ... },
  "analytics_results": { ... },
  "report_output": "...",
  "conversation_state": { "topics": [...], "period": "12-18 Jun" }
}
```

When a user wants to ask follow-up questions about the report:
1. Click "Ask about this report" (in email/Slack/dashboard)
2. System loads the checkpoint into chatbot context
3. User enters chatbot at Step 1 with full report context pre-loaded

> **User clicks:** "Ask about this report"
> **User:** "Why is the forecast below target?"
> **Bot:** (already has the report data loaded — no re-fetching needed)
> "The forecast is £148k vs £150k target. The gap is mainly from..."

---

## Shared Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    What's Shared (build once)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ DATA LAYER                                               │   │
│  │ • Aggregated tables (same tables for both)               │   │
│  │ • Data query tools (sql_query, get_metrics, get_stock)   │   │
│  │ • External APIs (weather, calendar, events, competitor)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ANALYTICS ENGINE                                         │   │
│  │ • Trend detection (STL, Prophet)                         │   │
│  │ • Anomaly detection (Z-score, Isolation Forest)          │   │
│  │ • Root cause analysis (correlation engine)               │   │
│  │ • Forecasting (Prophet, LightGBM)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ CHART TOOLS                                              │   │
│  │ • line_chart, bar_chart, heatmap, waterfall_chart        │   │
│  │ • correlation_plot, gauge, calendar_view, sparkline      │   │
│  │ • Rendering: JSON spec → PNG/SVG/interactive             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LLM (same model)                                         │   │
│  │ • Interprets questions                                   │   │
│  │ • Decides which tools to call                            │   │
│  │ • Generates text + chart JSON                            │   │
│  │ • Only difference: system prompt (chat vs report style)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  What's Different (per interface)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CHATBOT                        │  SCHEDULED REPORT             │
│  ───────                        │  ────────────────             │
│  Trigger: user message          │  Trigger: cron schedule       │
│  Scope: focused on question     │  Scope: comprehensive         │
│  Output: text + 1-2 charts      │  Output: HTML with 6-8 charts │
│  Format: conversational         │  Format: structured sections  │
│  Multi-turn: yes                │  Multi-turn: via checkpoint   │
│  Delivery: inline (chat UI)     │  Delivery: email/Slack/HTML   │
│                                 │                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example: End-to-End

### Scheduled Report (Monday 08:00)

```
Cron fires → "Generate weekly report for 12-18 June"
    │
    ├── get_metrics(period="last_week") → revenue, CVR, AOV
    ├── compare(this_week, last_week, by="category") → WoW changes
    ├── trend_detect(all_categories, window="28d") → trending up/down
    ├── anomaly_check(all_metrics) → fleece CVR anomaly detected
    ├── root_cause(fleece_drop) → price increase + weather + competitor
    ├── get_weather(forecast="7d") → rain from Thursday
    ├── get_calendar(next="14d") → school ends Friday, bank holiday Monday
    ├── forecast(revenue, horizon="7d") → £148k projected
    ├── get_stock(critical=True) → Beta LT 2 days, kids waterproof 7 days
    │
    ├── bar_chart(category_performance_wow)
    ├── line_chart(revenue_trend_4_weeks)
    ├── heatmap(demand_forecast_next_7d)
    ├── waterfall_chart(revenue_bridge_wow)
    ├── correlation_plot(temp_vs_shorts)
    ├── stock_risk_chart(critical_skus)
    ├── calendar_view(events_weather_14d)
    │
    └── LLM assembles HTML report → deliver via email + Slack
```

### User Follow-up (Monday 09:30)

```
User clicks "Ask about this report" in Slack
    │
    ├── Load checkpoint (report_2024_06_18_weekly)
    │   (data already in context — no re-fetching)
    │
    └── User: "Should we really restock Beta LT or push people to the Montane?"
            │
            ├── compare(product="beta_lt", vs="montane_phase", metrics=["cvr", "margin", "review"])
            ├── bar_chart(beta_lt_vs_montane_comparison)
            │
            └── Bot: "Beta LT converts at 12% vs Montane at 7% for this customer
                      segment. Margin is similar (£115 vs £75). Given the brand 
                      loyalty signal (60% of Beta LT buyers are repeat Arc'teryx 
                      customers), restocking is the better revenue play."
                      
                      [📊 BAR CHART: Beta LT vs Montane — CVR, margin, satisfaction]
```

---

## Summary

| Concept | Implementation |
|---------|---------------|
| Ingestion | Data sources → ETL → aggregated tables → analytics engine (runs continuously) |
| Chatbot | User question → query tables → query analytics → LLM response (text + charts) → follow-up loop |
| Report | Cron question → same pipeline → LLM HTML output → deliver → checkpoint saved for follow-up |
| Reuse | Same tables, same analytics, same chart tools, same LLM — two entry points |
| Checkpoint | Report saves state so users can continue the conversation in chatbot mode |
