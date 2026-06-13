# Memory & Conversation Management

## Overview

LLMs are stateless — each call starts fresh with no memory of past interactions. Memory systems give agents the ability to maintain context across turns, sessions, and long-term relationships. The right memory strategy depends on conversation length, cost tolerance, and what information matters.

## Types of Memory

| Type | Scope | Stores | Use Case |
|------|-------|--------|----------|
| Buffer (short-term) | Current session | Full message history | Simple chatbots |
| Window | Current session | Last N messages | Long conversations |
| Summary | Current session | Compressed summary | Very long sessions |
| Entity | Cross-session | Facts about entities | Personalisation |
| Vector (RAG-backed) | Cross-session | Embedded past interactions | Knowledge recall |
| Long-term | Persistent | User preferences, facts | Ongoing relationships |

---

## Method 1: Buffer Memory (Full History)

Pass the entire conversation history to the LLM every turn. Simple but expensive for long conversations.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Simple in-memory history
conversation_history = []

def chat(user_message: str) -> str:
    conversation_history.append(HumanMessage(content=user_message))

    response = llm.invoke(conversation_history)

    conversation_history.append(AIMessage(content=response.content))
    return response.content

# Every call sends full history — grows linearly with turns
chat("What frameworks are best for web APIs?")
chat("How does FastAPI compare specifically?")  # Has context from turn 1
```

### LangGraph with Message State

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add]

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Checkpointer persists state between invocations
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Same thread_id = same conversation
config = {"configurable": {"thread_id": "user-123"}}
app.invoke({"messages": [HumanMessage("Hello")]}, config)
app.invoke({"messages": [HumanMessage("What did I just say?")]}, config)  # Remembers
```

---

## Method 2: Window Memory (Last N Messages)

Keep only the most recent N messages. Simple, predictable token usage.

```python
from langchain_core.messages import trim_messages

def windowed_chat(user_message: str, history: list, window_size: int = 20) -> str:
    history.append(HumanMessage(content=user_message))

    # Trim to last N messages (keep system message if present)
    trimmed = trim_messages(
        history,
        max_messages=window_size,
        strategy="last",
        start_on="human",  # Ensure we start on a human message
        include_system=True,
    )

    response = llm.invoke(trimmed)
    history.append(AIMessage(content=response.content))
    return response.content
```

### Token-Based Trimming

```python
from langchain_core.messages import trim_messages

# Trim by token count instead of message count
trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    token_counter=llm,  # Uses the LLM's tokenizer
    include_system=True,
    allow_partial=False,
)
```

---

## Method 3: Summary Memory

Periodically summarise older messages to compress history.

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class SummaryMemory:
    def __init__(self, llm, max_messages: int = 10):
        self.llm = llm
        self.max_messages = max_messages
        self.summary = ""
        self.recent_messages = []

    def add_message(self, message):
        self.recent_messages.append(message)

        # Compress when history gets too long
        if len(self.recent_messages) > self.max_messages:
            self._compress()

    def _compress(self):
        """Summarise older messages and keep only recent ones."""
        old_messages = self.recent_messages[: -5]  # Summarise all but last 5
        self.recent_messages = self.recent_messages[-5:]

        messages_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in old_messages
        )

        new_summary = self.llm.invoke(
            f"Summarise this conversation, preserving key facts and context:\n\n"
            f"Previous summary: {self.summary}\n\n"
            f"New messages:\n{messages_text}"
        ).content

        self.summary = new_summary

    def get_messages(self) -> list:
        """Get current context for the LLM."""
        messages = []
        if self.summary:
            messages.append(SystemMessage(
                content=f"Conversation summary so far: {self.summary}"
            ))
        messages.extend(self.recent_messages)
        return messages
```

---

## Method 4: Entity Memory

Extract and store facts about entities (people, projects, concepts) mentioned in conversation.

```python
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

class ExtractedEntities(BaseModel):
    entities: list[dict] = Field(
        description="List of entities with name, type, and facts"
    )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
extractor = llm.with_structured_output(ExtractedEntities)

class EntityMemory:
    def __init__(self):
        self.entities: dict[str, dict] = {}  # name -> {type, facts: []}

    def update_from_conversation(self, messages: list):
        """Extract entities and facts from recent messages."""
        text = "\n".join(m.content for m in messages[-4:])  # Last 4 messages

        result = extractor.invoke(
            f"Extract entities and facts from this conversation:\n{text}\n\n"
            f"For each entity, provide: name, type (person/project/concept), "
            f"and key facts learned."
        )

        for entity in result.entities:
            name = entity["name"]
            if name not in self.entities:
                self.entities[name] = {"type": entity.get("type"), "facts": []}
            self.entities[name]["facts"].extend(entity.get("facts", []))

    def get_context(self, query: str) -> str:
        """Get relevant entity context for a query."""
        if not self.entities:
            return ""

        relevant = []
        query_lower = query.lower()
        for name, info in self.entities.items():
            if name.lower() in query_lower:
                facts = "; ".join(info["facts"][-5:])  # Last 5 facts
                relevant.append(f"{name} ({info['type']}): {facts}")

        return "\n".join(relevant) if relevant else ""
```

---

## Method 5: Vector-Backed Memory (Long-Term RAG)

Store past interactions as embeddings — retrieve relevant ones when needed.

```python
import chromadb
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.collection = self.client.get_or_create_collection(
            name=f"memory_{user_id}",
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-base-en-v1.5"
            ),
        )

    def store_interaction(self, user_msg: str, ai_msg: str, metadata: dict = None):
        """Store a conversation turn as a memory."""
        memory_text = f"User asked: {user_msg}\nAssistant answered: {ai_msg}"
        self.collection.add(
            documents=[memory_text],
            ids=[f"mem_{datetime.now().isoformat()}"],
            metadatas=[{
                "timestamp": datetime.now().isoformat(),
                "user_message": user_msg[:200],
                **(metadata or {}),
            }],
        )

    def recall(self, query: str, n_results: int = 5) -> list[str]:
        """Retrieve relevant past interactions."""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []

    def get_memory_context(self, current_query: str) -> str:
        """Get relevant memories formatted for the LLM."""
        memories = self.recall(current_query, n_results=3)
        if not memories:
            return ""
        return "Relevant past interactions:\n" + "\n---\n".join(memories)

# Usage in a chat
memory = VectorMemory(user_id="user-123")

def chat_with_memory(user_message: str, history: list) -> str:
    # Recall relevant past conversations
    memory_context = memory.get_memory_context(user_message)

    messages = []
    if memory_context:
        messages.append(SystemMessage(content=memory_context))
    messages.extend(history[-10:])  # Recent window
    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)

    # Store this interaction
    memory.store_interaction(user_message, response.content)

    return response.content
```

---

## Method 6: LangGraph Store (Built-in Long-Term Memory)

LangGraph's native memory store for cross-session persistence.

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Store for long-term memories (survives across threads)
store = InMemoryStore()

def chat_with_store(state, config, *, store):
    """Node that reads/writes long-term memory."""
    user_id = config["configurable"]["user_id"]

    # Read user memories
    memories = store.search(("memories", user_id), query=state["messages"][-1].content, limit=5)
    memory_text = "\n".join(m.value["content"] for m in memories)

    # Generate response with memory context
    messages = state["messages"]
    if memory_text:
        messages = [SystemMessage(content=f"User memories:\n{memory_text}")] + messages

    response = llm.invoke(messages)

    # Store important facts from this interaction
    store.put(
        ("memories", user_id),
        key=f"mem_{len(memories)}",
        value={"content": f"Q: {state['messages'][-1].content} A: {response.content}"},
    )

    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat", chat_with_store)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile(checkpointer=MemorySaver(), store=store)
```

---

## Method 7: Conversation Persistence (Session Management)

Store and restore full conversation sessions.

```python
import json
from pathlib import Path
from langchain_core.messages import messages_from_dict, messages_to_dict

class ConversationStore:
    def __init__(self, storage_dir: str = "./conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_session(self, session_id: str, messages: list):
        """Persist a conversation session."""
        path = self.storage_dir / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "messages": messages_to_dict(messages),
            "updated_at": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    def load_session(self, session_id: str) -> list | None:
        """Restore a conversation session."""
        path = self.storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return messages_from_dict(data["messages"])

    def list_sessions(self, user_id: str = None) -> list[str]:
        """List available sessions."""
        return [p.stem for p in self.storage_dir.glob("*.json")]
```

---

## Choosing a Memory Strategy

| Scenario | Recommended |
|----------|-------------|
| Simple chatbot (<20 turns) | Buffer (full history) |
| Long conversations | Window (last N) + summary |
| Multi-session assistant | Vector memory + window |
| Personalised agent | Entity memory + vector memory |
| Production with persistence | LangGraph checkpointer + store |
| Cost-sensitive | Window (strict token limit) |

## Memory Architecture for Production

```
┌─────────────────────────────────────────────────┐
│                  User Message                    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           Memory Assembly Layer                  │
│                                                 │
│  1. System prompt (static)                      │
│  2. Long-term memory (vector recall)            │
│  3. Entity facts (relevant entities)            │
│  4. Conversation summary (compressed history)   │
│  5. Recent messages (last N turns)              │
│  6. Current user message                        │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│                   LLM Call                       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│           Post-Response Updates                  │
│                                                 │
│  • Store interaction in vector memory           │
│  • Update entity facts                          │
│  • Append to conversation history               │
│  • Trigger summary if history too long          │
└─────────────────────────────────────────────────┘
```

## Best Practices

- Start with window memory — simplest and sufficient for most use cases
- Add summary memory when conversations regularly exceed 20+ turns
- Vector memory is essential for cross-session personalisation
- Always keep the system prompt outside of memory management (never trim it)
- Token budget: allocate fixed percentages (e.g. 20% memory, 30% context, 50% generation)
- Store metadata (timestamps, topics) with memories for better recall filtering
- Periodically clean stale memories — old facts may be outdated
- Test memory recall quality with representative queries
- Use LangGraph checkpointer for conversation state, store for long-term facts
- Never store sensitive PII in memory without encryption/access controls
