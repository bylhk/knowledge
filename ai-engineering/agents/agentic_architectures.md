# Agentic Architectures

## Overview

Agentic architectures enable LLMs to go beyond single-turn Q&A — they plan, use tools, maintain state, and iterate toward goals autonomously. An agent is an LLM in a loop: observe → think → act → observe the result → repeat until the task is done.

## Core Concepts

| Concept | Description |
|---------|-------------|
| Agent | LLM + tools + loop — decides what action to take next |
| Tool | Function the agent can call (search, calculate, API call) |
| State | Information carried between steps (memory, intermediate results) |
| Planning | Breaking a complex goal into sub-tasks |
| Reflection | Agent evaluating its own output and correcting mistakes |
| Human-in-the-loop | Pausing for human approval before critical actions |

---

## Pattern 1: ReAct (Reasoning + Acting)

The most common pattern — interleave thinking with tool calls.

```
Thought: I need to find the relevant documentation
Action: search_docs("system architecture components")
Observation: "Key components: API gateway (handles routing), cache layer (Redis)..."
Thought: I have the answer now
Action: final_answer("The key components are the API gateway and cache layer")
```

### LangGraph Implementation

```python
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@tool
def search_knowledge(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = vectorstore.similarity_search(query, k=5)
    return "\n".join([doc.page_content for doc in results])

@tool
def run_sql(query: str) -> str:
    """Execute a BigQuery SQL query and return results."""
    return bigquery_client.query(query).to_dataframe().to_string()

# Create ReAct agent
agent = create_react_agent(
    model=llm,
    tools=[search_knowledge, run_sql],
    state_modifier="You are a helpful data science assistant.",
)

# Run
result = agent.invoke({"messages": [{"role": "user", "content": "What are the system requirements?"}]})
```

---

## Pattern 2: Plan-and-Execute

Separate planning from execution — a planner creates a step-by-step plan, an executor handles each step.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered list of steps to complete the task")
    reasoning: str = Field(description="Why this plan was chosen")

class AgentState(TypedDict):
    messages: list
    plan: Plan | None
    current_step: int
    results: list[str]

def planner(state: AgentState) -> AgentState:
    """Create a plan for the task."""
    planner_llm = llm.with_structured_output(Plan)
    user_query = state["messages"][-1].content
    plan = planner_llm.invoke(
        f"Create a step-by-step plan to answer: {user_query}\n"
        f"Available tools: search_knowledge, run_sql, calculate"
    )
    return {"plan": plan, "current_step": 0, "results": []}

def executor(state: AgentState) -> AgentState:
    """Execute the current step of the plan."""
    step = state["plan"].steps[state["current_step"]]
    result = react_agent.invoke({"messages": [{"role": "user", "content": step}]})
    return {
        "current_step": state["current_step"] + 1,
        "results": state["results"] + [result["messages"][-1].content],
    }

def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"].steps):
        return "synthesise"
    return "execute"

def synthesise(state: AgentState) -> AgentState:
    """Combine all step results into a final answer."""
    combined = "\n".join(f"Step {i+1}: {r}" for i, r in enumerate(state["results"]))
    response = llm.invoke(
        f"Synthesise these results into a final answer:\n{combined}"
    )
    return {"messages": state["messages"] + [response]}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("plan", planner)
graph.add_node("execute", executor)
graph.add_node("synthesise", synthesise)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", should_continue, {"execute": "execute", "synthesise": "synthesise"})
graph.add_edge("synthesise", END)

app = graph.compile()
```

---

## Pattern 3: Tool Calling (Function Calling)

The LLM decides which tool to call and with what arguments. The framework executes the tool and returns results.

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

class DataQuery(BaseModel):
    """Query metrics data from BigQuery."""
    category: Literal["users", "orders", "inventory"]
    metric: Literal["conversion_rate", "revenue", "churn"]
    time_period: str = Field(description="e.g. 'last_30_days', '2024-Q4'")

@tool(args_schema=DataQuery)
def query_metrics(category: str, metric: str, time_period: str) -> str:
    """Query business metrics from the data warehouse."""
    sql = f"""
        SELECT {metric}, COUNT(*) as n
        FROM analytics.events
        WHERE category = '{category}' AND period = '{time_period}'
        GROUP BY 1
    """
    return execute_sql(sql)

@tool
def create_chart(data: str, chart_type: str, title: str) -> str:
    """Create a visualisation from data."""
    # Generate chart and return path
    return f"Chart saved: /tmp/{title}.png"

# Bind tools to LLM
llm_with_tools = llm.bind_tools([query_metrics, create_chart])

# The LLM returns tool call messages, framework executes them
response = llm_with_tools.invoke("Show me user conversion rates for Q4 2024")
# AIMessage with tool_calls: [{"name": "query_metrics", "args": {...}}]
```

---

## Pattern 4: Multi-Agent Systems

Multiple specialised agents collaborate on complex tasks.

### Supervisor Pattern

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal

class RouterDecision(BaseModel):
    next_agent: Literal["researcher", "analyst", "writer", "done"]
    reasoning: str

def supervisor(state: AgentState) -> AgentState:
    """Route to the appropriate specialist agent."""
    router = llm.with_structured_output(RouterDecision)
    decision = router.invoke(
        f"Given the task and progress so far, which agent should act next?\n"
        f"Task: {state['task']}\n"
        f"Progress: {state['results']}\n\n"
        f"Agents:\n"
        f"- researcher: Searches knowledge base and documents\n"
        f"- analyst: Runs SQL queries and calculations\n"
        f"- writer: Synthesises findings into a report\n"
        f"- done: Task is complete"
    )
    return {"next_agent": decision.next_agent}

def researcher(state: AgentState) -> AgentState:
    """Research agent with access to RAG tools."""
    result = research_agent.invoke(state)
    return {"results": state["results"] + [f"[Research] {result}"]}

def analyst(state: AgentState) -> AgentState:
    """Analysis agent with access to SQL and Python."""
    result = analysis_agent.invoke(state)
    return {"results": state["results"] + [f"[Analysis] {result}"]}

def writer(state: AgentState) -> AgentState:
    """Writing agent that synthesises findings."""
    result = writing_agent.invoke(state)
    return {"results": state["results"] + [f"[Report] {result}"]}

# Build supervisor graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next_agent"], {
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    "done": END,
})
graph.add_edge("researcher", "supervisor")
graph.add_edge("analyst", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()
```

### Handoff Pattern

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Each agent can "hand off" to another
@tool
def transfer_to_analyst():
    """Transfer to the data analyst agent for SQL queries and calculations."""
    pass

@tool
def transfer_to_researcher():
    """Transfer to the researcher agent for knowledge base search."""
    pass

researcher_agent = create_react_agent(
    model=llm,
    tools=[search_knowledge, transfer_to_analyst],
    state_modifier="You are a researcher. Search for information. Hand off to analyst for data queries.",
)

analyst_agent = create_react_agent(
    model=llm,
    tools=[run_sql, calculate, transfer_to_researcher],
    state_modifier="You are a data analyst. Run queries and calculations. Hand off to researcher for context.",
)
```

---

## Pattern 5: Reflection & Self-Correction

Agent evaluates its own output and iterates.

```python
from langgraph.graph import StateGraph, START, END

class ReflectionState(TypedDict):
    messages: list
    draft: str
    critique: str
    iteration: int

def generate(state: ReflectionState) -> ReflectionState:
    """Generate or revise a response."""
    if state.get("critique"):
        prompt = f"Revise this based on the feedback:\nDraft: {state['draft']}\nFeedback: {state['critique']}"
    else:
        prompt = state["messages"][-1].content

    response = llm.invoke(prompt)
    return {"draft": response.content, "iteration": state.get("iteration", 0) + 1}

def reflect(state: ReflectionState) -> ReflectionState:
    """Critique the current draft."""
    critique = llm.invoke(
        f"Critique this response. Is it accurate, complete, and well-structured?\n"
        f"If it's good enough, say 'APPROVED'.\n\nDraft:\n{state['draft']}"
    )
    return {"critique": critique.content}

def should_continue(state: ReflectionState) -> str:
    if "APPROVED" in state.get("critique", "") or state.get("iteration", 0) >= 3:
        return "end"
    return "revise"

graph = StateGraph(ReflectionState)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)

graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {"revise": "generate", "end": END})

app = graph.compile()
```

---

## Pattern 6: Human-in-the-Loop

Pause execution for human approval before critical actions.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

def sensitive_action(state: AgentState) -> AgentState:
    """Action that requires human approval."""
    action = state["pending_action"]

    # Pause and ask human
    human_response = interrupt(
        f"The agent wants to: {action['description']}\n"
        f"Tool: {action['tool']}\n"
        f"Args: {action['args']}\n\n"
        f"Approve? (yes/no)"
    )

    if human_response.lower() == "yes":
        result = execute_tool(action["tool"], action["args"])
        return {"results": state["results"] + [result]}
    else:
        return {"results": state["results"] + ["Action rejected by human"]}

# Compile with checkpointing (enables interrupt/resume)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Run until interrupt
config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke({"messages": [...]}, config)

# Resume after human approval
result = app.invoke(Command(resume="yes"), config)
```

---

## Pattern 7: Stateful Workflows (LangGraph)

Full state machine with persistence, branching, and recovery.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add

class WorkflowState(TypedDict):
    messages: Annotated[list, add]  # Append-only message list
    context: list[str]
    tool_calls: list[dict]
    error_count: int
    status: str

def retrieve(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant context."""
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs], "status": "retrieved"}

def generate(state: WorkflowState) -> WorkflowState:
    """Generate response using context."""
    context = "\n".join(state["context"])
    response = llm.invoke(
        f"Context:\n{context}\n\nQuestion: {state['messages'][-1].content}"
    )
    return {"messages": [response], "status": "generated"}

def handle_error(state: WorkflowState) -> WorkflowState:
    """Error recovery node."""
    return {
        "messages": [AIMessage(content="I encountered an error. Let me try differently.")],
        "error_count": state.get("error_count", 0) + 1,
        "status": "error_handled",
    }

def route(state: WorkflowState) -> str:
    if state.get("error_count", 0) >= 3:
        return "escalate"
    if not state.get("context"):
        return "retrieve"
    return "generate"

# Build with error handling
graph = StateGraph(WorkflowState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("error_handler", handle_error)

graph.add_conditional_edges(START, route, {
    "retrieve": "retrieve",
    "generate": "generate",
    "escalate": END,
})
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
graph.add_edge("error_handler", "retrieve")

# Compile with persistence
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# State persists across invocations
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [HumanMessage("What is caching?")]}, config)
```

---

## LangGraph Key Concepts

| Concept | Description |
|---------|-------------|
| Node | A function that transforms state |
| Edge | Connection between nodes (unconditional or conditional) |
| State | TypedDict shared across all nodes |
| Checkpointer | Persistence layer (memory, SQLite, Postgres) |
| `interrupt()` | Pause execution for human input |
| `Command(resume=...)` | Resume from interrupt with human response |
| Thread | A conversation/workflow session (identified by thread_id) |
| `Annotated[list, add]` | Reducer — how state updates are merged (append vs replace) |

---

## Architecture Comparison

| Pattern | Complexity | Best For |
|---------|-----------|----------|
| ReAct | Low | Simple tool-use agents |
| Plan-and-Execute | Medium | Multi-step tasks with clear goals |
| Tool Calling | Low | Structured API interactions |
| Multi-Agent (Supervisor) | High | Complex tasks needing specialisation |
| Multi-Agent (Handoff) | Medium | Conversational routing |
| Reflection | Medium | Quality-sensitive outputs |
| Human-in-the-Loop | Medium | High-stakes actions |
| Stateful Workflow | High | Production systems with recovery |

## Error Handling Strategies

| Strategy | Implementation |
|----------|---------------|
| Retry with backoff | Catch tool errors, retry N times |
| Fallback tools | If primary tool fails, try alternative |
| Self-correction | Agent detects errors in output, retries |
| Escalation | After N failures, hand off to human |
| State rollback | Revert to last known good checkpoint |

## Best Practices

- Start with ReAct — only add complexity when you need it
- Use structured output for tool arguments (Pydantic schemas prevent malformed calls)
- Always set a max iteration limit — agents can loop forever without one
- Add observability (LangSmith/Langfuse) to trace agent reasoning steps
- Human-in-the-loop for any action with side effects (writes, sends, deletes)
- Test agents with diverse scenarios including adversarial inputs
- Keep tool descriptions clear and unambiguous — the LLM relies on them to choose
- Use checkpointing for any agent that runs longer than a few seconds
- Separate planning from execution for complex multi-step tasks
- Monitor token usage per agent run — complex agents can burn through context quickly
