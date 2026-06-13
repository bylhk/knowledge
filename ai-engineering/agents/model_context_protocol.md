# Model Context Protocol (MCP)

## Overview

MCP is an open standard (by Anthropic) that provides a universal interface for connecting LLMs to external tools and data sources. Think of it like USB-C for AI — write a tool server once, and any MCP-compatible host (Claude, Kiro, custom agents) can use it without custom integration code.

## Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────────┐
│  MCP Host   │◄───────►│  MCP Client │◄───────►│   MCP Server    │
│ (Claude,    │  JSON   │ (built into │  stdio/  │ (your tools &   │
│  Kiro, etc) │  RPC    │  the host)  │  SSE     │  data sources)  │
└─────────────┘         └─────────────┘         └─────────────────┘
```

| Component | Role | Example |
|-----------|------|---------|
| Host | The AI application that needs tools | Claude Desktop, Kiro, custom agent |
| Client | Protocol handler within the host | Manages connection lifecycle |
| Server | Exposes tools and resources | Your Python/Node server |

## Key Concepts

| Concept | Description |
|---------|-------------|
| Tools | Functions the LLM can call (with input schemas) |
| Resources | Data the server exposes for context (files, DB schemas) |
| Prompts | Reusable prompt templates the server provides |
| Transports | Communication layer (stdio for local, SSE for remote) |

---

## Building an MCP Server (Python)

### Install

```bash
pip install "mcp[cli]"
```

### Basic Server

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-tools")

@mcp.tool()
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        city: City name (e.g. 'London', 'New York')
        units: Temperature units ('celsius' or 'fahrenheit')
    """
    result = weather_api.get_current(city=city, units=units)
    return f"{city}: {result['temp']}°{'C' if units == 'celsius' else 'F'}, {result['condition']}"

@mcp.tool()
def search_knowledge(query: str, n_results: int = 5) -> str:
    """Search the knowledge base for relevant information.

    Args:
        query: Natural language search query
        n_results: Number of results to return
    """
    results = vectorstore.query(query_texts=[query], n_results=n_results)
    return "\n---\n".join(results["documents"][0])

@mcp.resource("config://app-settings")
def get_app_config() -> str:
    """Current application configuration."""
    return yaml.dump(load_config())

@mcp.prompt()
def analysis_prompt(topic: str) -> str:
    """Generate a structured analysis prompt."""
    return f"Analyse the following topic with data-backed insights:\n\nTopic: {topic}\n\nProvide: 1) Key findings 2) Supporting data 3) Recommendations"
```

### Running the Server

```bash
# Development (with inspector)
mcp dev server.py

# Install in Claude Desktop / Kiro
mcp install server.py

# Direct run (stdio transport)
python -m mcp run server.py
```

### With Dependencies and Configuration

```python
from mcp.server.fastmcp import FastMCP
import chromadb

mcp = FastMCP(
    "knowledge-base",
    dependencies=["chromadb", "sentence-transformers", "pyyaml"],
)

# Lifespan management for expensive resources
@mcp.lifespan
async def app_lifespan(server):
    """Initialize resources on startup, clean up on shutdown."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("knowledge")
    yield {"collection": collection}
    # Cleanup happens after yield

@mcp.tool()
def search(query: str, ctx=None) -> str:
    """Search the vector knowledge base."""
    collection = ctx.state["collection"]
    results = collection.query(query_texts=[query], n_results=5)
    return format_results(results)
```

---

## Server with ChromaDB (Knowledge Base)

```python
from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import yaml

mcp = FastMCP("docs-server")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=config["db_path"])

def get_collection(name: str):
    collection_config = config["collections"][name]
    ef = SentenceTransformerEmbeddingFunction(
        model_name=collection_config["embedding_model"]
    )
    return client.get_or_create_collection(name=name, embedding_function=ef)

@mcp.tool()
def search_docs(
    query: str,
    n_results: int = 5,
    category: str | None = None,
    source: str | None = None,
) -> str:
    """Search documentation knowledge base.

    Args:
        query: Natural language search query
        n_results: Number of results to return
        category: Filter by document category (e.g. 'api', 'guide', 'reference')
        source: Filter by source document name
    """
    collection = get_collection("docs")

    # Build metadata filter
    where_filter = {}
    if category:
        where_filter["category"] = category
    if source:
        where_filter["source"] = {"$contains": source}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter if where_filter else None,
    )

    return format_results(results)
```

---

## Transport Options

### stdio (Local — Default)

Server runs as a subprocess, communicates via stdin/stdout. Used for local tools.

```json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["-m", "mcp", "run", "server.py"],
      "env": {"PYTHONPATH": "/path/to/project"}
    }
  }
}
```

### SSE (Remote — HTTP)

Server runs as a web service, communicates via Server-Sent Events. Used for shared/remote tools.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("remote-tools")

# Run with SSE transport
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

```json
{
  "mcpServers": {
    "remote-tools": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

---

## Configuration (mcp.json)

### Workspace-level (.kiro/settings/mcp.json)

```json
{
  "mcpServers": {
    "docs-server": {
      "command": "uv",
      "args": ["--directory", "/path/to/docs-server", "run", "docs-server"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      },
      "disabled": false,
      "autoApprove": ["search_docs"]
    },
    "aws-docs": {
      "command": "uvx",
      "args": ["awslabs.aws-documentation-mcp-server@latest"],
      "env": {"FASTMCP_LOG_LEVEL": "ERROR"},
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### User-level (~/.kiro/settings/mcp.json)

Same format — applies globally across all workspaces. Workspace configs override user-level.

---

## Tool Design Best Practices

### Good Tool Definition

```python
@mcp.tool()
def search_products(
    query: str,
    category: str | None = None,
    max_results: int = 10,
) -> str:
    """Search the product catalogue by name or description.

    Returns matching products with name, price, and availability.
    Results are sorted by relevance.

    Args:
        query: Search query (product name, keyword, or description)
        category: Optional category filter (e.g. 'electronics', 'clothing')
        max_results: Maximum number of results to return (default: 10)
    """
    # Implementation
    ...
```

### Tool Design Principles

| Principle | Why |
|-----------|-----|
| Clear docstring | LLM uses it to decide when to call the tool |
| Typed parameters | Generates accurate JSON schema for the LLM |
| Descriptive arg names | Helps LLM provide correct values |
| Sensible defaults | Reduces required decisions |
| Return strings | LLM consumes text — format results as readable text |
| Error messages | Return helpful errors, not stack traces |

---

## Testing MCP Servers

### With MCP Inspector

```bash
# Interactive testing UI
mcp dev server.py

# Opens browser with tool list, input forms, and response viewer
```

### Programmatic Testing

```python
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.fixture
async def mcp_session():
    params = StdioServerParameters(command="python", args=["-m", "mcp", "run", "server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def test_search_docs(mcp_session):
    result = await mcp_session.call_tool(
        "search_docs",
        arguments={"query": "getting started guide", "n_results": 3},
    )
    assert result.content
    assert len(result.content) > 0
```

---

## MCP vs Direct Tool Integration

| Aspect | MCP Server | Direct LangChain Tool |
|--------|-----------|----------------------|
| Reusability | Any MCP host can use it | Only LangChain apps |
| Protocol | Standardised JSON-RPC | Framework-specific |
| Discovery | Tools self-describe schemas | Must be registered in code |
| Deployment | Separate process | In-process |
| State management | Server manages its own state | Shared with app |
| Best for | Shared organisational tools | App-specific logic |

## Common MCP Server Patterns

| Pattern | Description |
|---------|-------------|
| Knowledge base | Vector DB search over domain docs |
| API wrapper | Expose internal APIs as LLM tools |
| Database query | Natural language → SQL → results |
| File system | Read/write project files |
| External service | Jira, Confluence, Slack integration |

## Best Practices

- One server per domain (don't mix unrelated tools together)
- Keep tools focused — one clear action per tool
- Return formatted text, not raw JSON (LLMs read text better)
- Add metadata filters to search tools for precision
- Use lifespan management for expensive resources (DB connections, models)
- Test tools with the MCP inspector before integrating with hosts
- Version your servers — breaking changes affect all connected hosts
- Log tool calls for observability (who called what, when)
- Use `autoApprove` for read-only tools, require approval for write operations
- Handle errors gracefully — return error messages the LLM can understand
