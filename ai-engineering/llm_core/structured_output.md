# LLM Structured Output

## What Is It?

Structured output forces an LLM to return responses in a predictable, machine-parseable format (JSON, Pydantic models, enums, etc.) rather than free-form text. This is essential for building reliable pipelines where downstream code needs to consume LLM outputs programmatically.

## Why It Matters

- Eliminates brittle regex/string parsing of LLM responses
- Enables type-safe integration with application code
- Reduces hallucinated or malformed outputs
- Makes LLM calls composable in data pipelines

## Approaches

### 1. Schema-Constrained Decoding (Provider-Level)

The model provider enforces the output schema during token generation — the model literally cannot produce invalid output.

- OpenAI: `response_format={"type": "json_schema", ...}`
- Google Gemini: `generation_config={"response_mime_type": "application/json", "response_schema": ...}`
- Ollama/vLLM: JSON mode with grammar constraints

### 2. Token-Level Constraint Libraries (Guidance / Outlines)

Enforce structure at the decoding step for local models using FSMs, regex, or template DSLs.

### 3. Framework-Level Extraction (LangChain / Instructor)

Wraps the LLM call with schema validation, retries, and parsing logic.

### 4. Prompt-Based (Least Reliable)

Ask the model to output JSON in the prompt. Works but fragile — no guarantee of valid structure.

---

## Cloud API Examples

### LangChain `with_structured_output`

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# 1. Define output schema
class MovieRecommendation(BaseModel):
    """Structured movie recommendation."""
    title: str = Field(description="Recommended movie title")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    reasoning: str = Field(description="Brief explanation of the recommendation")
    genre: str = Field(description="Primary genre of the movie")

# 2. Bind schema to LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
structured_llm = llm.with_structured_output(MovieRecommendation)

# 3. Invoke — returns a MovieRecommendation instance, not raw text
result = structured_llm.invoke(
    "User enjoys sci-fi films with complex plots like Inception and Interstellar. "
    "They prefer movies under 2.5 hours. Recommend a movie."
)

print(result.title)       # "Arrival"
print(result.confidence)  # 0.82
print(result.genre)       # "sci-fi"
```

### OpenAI Structured Outputs (Native)

```python
from openai import OpenAI
from pydantic import BaseModel

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    confidence: float

class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    summary: str

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract entities from the text."},
        {"role": "user", "content": "Harrison Chase founded LangChain in October 2022."},
    ],
    response_format=ExtractionResult,
)

result = response.choices[0].message.parsed
for entity in result.entities:
    print(f"{entity.name} ({entity.entity_type}): {entity.confidence}")
```

### Instructor Library

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Review(BaseModel):
    sentiment: Sentiment
    key_topics: list[str] = Field(max_length=5)
    summary: str = Field(max_length=100)

client = instructor.from_openai(OpenAI())

review = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Review,
    messages=[
        {"role": "user", "content": "The new laptop is great but delivery took 3 weeks."}
    ],
)

print(review.sentiment)    # Sentiment.NEUTRAL
print(review.key_topics)   # ["laptop quality", "delivery delay"]
```

### Gemini Native JSON Mode

```python
import google.generativeai as genai

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "importance_scores": {
                    "type": "array",
                    "items": {"type": "number"},
                },
            },
            "required": ["features", "importance_scores"],
        },
    },
)

response = model.generate_content("List the top 3 features for churn prediction.")
# Response is guaranteed valid JSON matching the schema
```

---

## Token-Level Constraint Libraries (Local Models)

### Guidance (by Microsoft)

Interleaves generation with programmatic control flow. You write a template mixing static text, constrained generation, and logic — the model fills in the gaps while respecting your constraints.

```bash
pip install guidance
```

#### Constrained JSON Generation

```python
import guidance

model = guidance.models.Transformers("mistralai/Mistral-7B-Instruct-v0.2")

@guidance
def extract_product_info(lm):
    lm += "Extract product information from the review.\n"
    lm += "Review: Excellent build quality, but the battery only lasts 4 hours.\n\n"
    lm += "```json\n"
    lm += "{\n"
    lm += f'  "sentiment": "{guidance.select(["positive", "negative", "mixed"])}",\n'
    lm += f'  "price_mentioned": {guidance.gen(name="price", regex=r"(true|false)")},\n'
    lm += f'  "price_value": {guidance.gen(name="price_val", regex=r"[0-9]+\\.?[0-9]*")},\n'
    lm += f'  "summary": "{guidance.gen(name="summary", max_tokens=50, stop='"')}"\n'
    lm += "}\n```"
    return lm

result = model + extract_product_info()
print(result["price_val"])   # "35"
print(result["summary"])     # "Good speed but overpriced"
```

#### Guided Choice & Branching

```python
import guidance

model = guidance.models.Transformers("mistralai/Mistral-7B-Instruct-v0.2")

@guidance
def classify_ticket(lm, ticket_text):
    lm += f"Classify this support ticket: {ticket_text}\n\n"
    lm += f"Category: {guidance.select(['billing', 'technical', 'cancellation', 'upgrade'], name='category')}\n"
    lm += f"Priority: {guidance.select(['low', 'medium', 'high', 'critical'], name='priority')}\n"
    lm += f"Needs escalation: {guidance.select(['yes', 'no'], name='escalate')}\n"
    return lm

result = model + classify_ticket("My internet has been down for 3 days and nobody is helping!")
print(result["category"])   # "technical"
print(result["priority"])   # "critical"
print(result["escalate"])   # "yes"
```

#### Regex-Constrained Generation

```python
import guidance

model = guidance.models.Transformers("mistralai/Mistral-7B-Instruct-v0.2")

@guidance
def generate_email(lm):
    lm += "Generate a valid email address for this customer:\n"
    lm += f"Email: {guidance.gen(name='email', regex=r'[a-z]+\.[a-z]+@[a-z]+\.(com|co\.uk)')}\n"
    return lm

result = model + generate_email()
print(result["email"])  # "john.smith@company.co.uk"
```

#### Guidance Key Features

- Token-level constraint enforcement (not post-hoc validation)
- Supports `select` (enum), `gen` with regex, and control flow
- Works with local models (Transformers, llama.cpp, exllamav2)
- Stateful — can branch logic based on generated values
- Much faster than retry-based approaches

---

### Outlines (by .txt / dottxt)

Uses finite-state machines (FSMs) and regular expressions compiled into token masks to guarantee structured output. Integrates directly with the model's sampling step.

```bash
pip install outlines
```

#### JSON with Pydantic Schema

```python
import outlines
from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    document_id: str = Field(pattern=r"DOC-[0-9]{6}")
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str]
    summary: str

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")
generator = outlines.generate.json(model, SentimentAnalysis)

result = generator(
    "Analyse sentiment for document DOC-004521. "
    "The product review mentions poor battery life and overheating issues repeatedly."
)

print(type(result))        # <class 'SentimentAnalysis'>
print(result.sentiment)    # "negative"
print(result.confidence)   # 0.87
print(result.summary)      # "Negative review citing hardware reliability concerns"
```

#### Regex-Constrained Generation

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

# Generate a valid UK phone number
phone_generator = outlines.generate.regex(model, r"(\+44|0)7[0-9]{9}")
phone = phone_generator("Generate a UK mobile number:")
print(phone)  # "07912345678"

# Generate a date in ISO format
date_generator = outlines.generate.regex(
    model,
    r"20[2-3][0-9]-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
)
date = date_generator("What is today's date?")
print(date)  # "2026-06-13"
```

#### Choice / Enum

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

sentiment_generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral", "mixed"]
)

sentiment = sentiment_generator(
    "Review: 'Speed is fantastic but the router keeps dropping connection.'\n"
    "Sentiment:"
)
print(sentiment)  # "mixed"
```

#### Grammar-Based (CFG)

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2")

arithmetic_grammar = r"""
    start: expr
    expr: term (("+"|"-") term)*
    term: factor (("*"|"/") factor)*
    factor: NUMBER | "(" expr ")"
    NUMBER: /[0-9]+(\.[0-9]+)?/
"""

math_generator = outlines.generate.cfg(model, arithmetic_grammar)
expression = math_generator("Generate a mathematical expression for the area of a rectangle with sides 5 and 3:")
print(expression)  # "5*3"
```

#### Outlines Key Features

- FSM/regex compiled to token masks — zero invalid tokens possible
- First-class Pydantic support (JSON schema → FSM)
- Context-free grammar support for complex structures
- Works with Hugging Face Transformers, vLLM, llama.cpp
- Batch generation with constraints
- Significantly faster than rejection sampling

---

## Comparison of All Approaches

| Approach | Reliability | Retry Needed | Latency | Model Support |
|----------|-------------|--------------|---------|---------------|
| Outlines (FSM) | 100% valid | No | Fast | Local only |
| Guidance (template) | 100% valid | No | Fast | Local only |
| Provider constrained decoding | 100% valid | No | Same | Cloud APIs |
| Instructor / LangChain | ~99% (retries) | Auto-retry | Slightly higher | Any |
| Prompt-only JSON | ~80-90% | Manual | Same | Any |

## Guidance vs Outlines

| Aspect | Guidance | Outlines |
|--------|----------|----------|
| Approach | Template + interleaved control | FSM/regex token masking |
| Schema support | Manual (select/gen/regex) | Pydantic, JSON Schema, regex, CFG |
| Control flow | Yes (if/else, loops) | No (pure generation) |
| Speed | Fast | Very fast (pre-compiled FSMs) |
| Best for | Complex multi-step generation | Strict schema enforcement |
| Learning curve | Moderate (template DSL) | Low (just pass a Pydantic model) |

## When to Use What

| Scenario | Recommended |
|----------|-------------|
| Pydantic model output from local LLM | **Outlines** |
| Complex conditional generation logic | **Guidance** |
| Simple JSON extraction with cloud APIs | **LangChain `with_structured_output`** |
| Maximum reliability + local model | **Outlines** |
| Multi-step reasoning with constrained outputs | **Guidance** |
| Production API with OpenAI/Gemini | **Native provider JSON mode** |
| Quick prototyping with retries | **Instructor** |

---

## Pydantic Patterns for LLM Schemas

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum

# Use Literal for constrained string fields
class Classification(BaseModel):
    category: Literal["bug", "feature", "question"]
    priority: Literal["low", "medium", "high", "critical"]

# Use Enum for reusable option sets
class Status(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

# Use Field for descriptions (helps the LLM understand intent)
class FeatureImportance(BaseModel):
    feature_name: str = Field(description="Name of the ML feature")
    importance: float = Field(ge=0, le=1, description="Normalised importance score")
    direction: Literal["positive", "negative"] = Field(
        description="Whether higher values increase or decrease the target"
    )

# Use validators for complex constraints
class ScoreResult(BaseModel):
    score: float = Field(gt=0, lt=10)
    category: str = Field(min_length=1)

    @field_validator("score")
    @classmethod
    def score_increment(cls, v):
        """Scores must be in 0.5 increments."""
        if round(v * 2) != v * 2:
            raise ValueError("Score must be in 0.5 increments")
        return v
```

## Best Practices

- Prefer constrained decoding when your provider supports it — zero parsing failures
- Use Outlines when you have a clear Pydantic schema and a local model
- Use Guidance when you need interleaved reasoning and generation
- Use Pydantic `Field(description=...)` liberally — descriptions guide the LLM
- Keep schemas flat where possible — deeply nested structures increase error rates
- Use `Literal` and `Enum` over free-form strings for categorical fields
- Add `field_validator` for business rules the LLM can't enforce alone
- For LangChain, `with_structured_output` handles retries and schema binding cleanly
- For cloud-hosted models (GPT-4, Gemini), provider-native structured output is simpler
- Combine Outlines/Guidance with quantisation (GPTQ/AWQ) for consumer GPUs
- Test edge cases: adversarial inputs, empty fields, long text
