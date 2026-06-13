# Prompt Optimisation

## What Is It?

Prompt optimisation is the systematic process of refining prompts to maximise LLM output quality for a specific task. Rather than manually tweaking prompts through trial and error, modern approaches use automated techniques — often LLM-driven — to iteratively improve prompts based on evaluation metrics.

## Why It Matters

- Manual prompt engineering is time-consuming and doesn't scale
- Small wording changes can dramatically shift model behaviour
- Optimised prompts reduce token usage, latency, and cost
- Enables non-experts to get expert-level prompt performance

## Key Techniques

### 1. Few-Shot Example Selection
Dynamically selecting the most relevant examples to include in the prompt based on the input query (e.g. using semantic similarity).

### 2. Instruction Tuning via DSPy
DSPy compiles declarative task descriptions into optimised prompts by treating prompt engineering as a programming problem rather than a writing problem.

### 3. Automatic Prompt Engineering (APE)
Uses an LLM to generate candidate prompts, evaluates them against a scoring function, and iterates.

### 4. Prompt Compression
Reduces prompt length while preserving semantic content (e.g. LLMLingua), cutting costs and fitting more context.

### 5. Chain-of-Thought Optimisation
Automatically discovering the best reasoning structure (e.g. step-by-step, tree-of-thought) for a given task.

## Example: DSPy Prompt Optimisation

```python
import dspy

# 1. Define the task as a signature
class QA(dspy.Signature):
    """Answer the question based on the given context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. Build a simple module
class SimpleQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(QA)

    def forward(self, context, question):
        return self.generate(context=context, question=question)

# 3. Define training examples
trainset = [
    dspy.Example(
        context="Python was created by Guido van Rossum and first released in 1991.",
        question="Who created Python?",
        answer="Guido van Rossum",
    ).with_inputs("context", "question"),
    dspy.Example(
        context="The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need'.",
        question="When was the transformer introduced?",
        answer="2017",
    ).with_inputs("context", "question"),
]

# 4. Define a metric
def exact_match(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()

# 5. Optimise the prompt
optimiser = dspy.MIPROv2(metric=exact_match, auto="medium")
optimised_qa = optimiser.compile(SimpleQA(), trainset=trainset)

# 6. Use the optimised module
result = optimised_qa(
    context="LangChain was created by Harrison Chase in October 2022.",
    question="Who created LangChain?",
)
print(result.answer)  # "Harrison Chase"
```

## Example: Manual Prompt Iteration Pattern

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Version 1: Basic prompt
v1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

# Version 2: Optimised with role, constraints, and format
v2 = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior data scientist. "
        "Answer concisely in 1-2 sentences. "
        "If uncertain, say 'I'm not sure' rather than guessing."
    )),
    ("human", "{question}"),
])

# Compare outputs
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
chain_v1 = v1 | llm
chain_v2 = v2 | llm

# Evaluate both on a test set and pick the winner
```

## Tools & Frameworks

| Tool | Approach | Best For |
|------|----------|----------|
| DSPy | Compiler-based optimisation | Complex pipelines, reproducibility |
| PromptFoo | Eval-driven iteration | Testing prompt variants at scale |
| LangSmith | Tracing + evaluation | Debugging and A/B testing prompts |
| TextGrad | Gradient-based text optimisation | Research, fine-grained control |
| OPRO (Google) | LLM-as-optimiser | Meta-prompt search |

## Best Practices

- Always define an evaluation metric before optimising
- Start simple — add complexity only when metrics justify it
- Version control your prompts alongside code
- Test across diverse inputs, not just happy-path examples
- Monitor prompt performance in production (drift detection)
- Consider cost/latency tradeoffs, not just accuracy
