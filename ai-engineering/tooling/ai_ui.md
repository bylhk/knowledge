# AI UI Frameworks

## Overview

Building chat and agent interfaces for LLM applications. These frameworks handle streaming, message history, file uploads, tool call rendering, and authentication — letting you focus on the AI logic.

---

## Chainlit (Python-Native Chat UI)

Production-ready chat interface with deep LangChain/LangGraph integration. Used in our agentic-ai-poc.

### Install

```bash
pip install chainlit
```

### Basic Chat App

```python
# app.py
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! Ask me anything.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Stream response
    msg = cl.Message(content="")
    await msg.send()

    response = llm.invoke(history)
    msg.content = response.content
    await msg.update()

    history.append({"role": "assistant", "content": response.content})
```

```bash
chainlit run app.py -w  # -w for auto-reload
```

### With LangChain Agent

```python
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@cl.on_chat_start
async def start():
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    cl.user_session.set("executor", executor)

@cl.on_message
async def main(message: cl.Message):
    executor = cl.user_session.get("executor")

    # Streaming with intermediate steps
    msg = cl.Message(content="")
    await msg.send()

    result = await executor.ainvoke(
        {"input": message.content, "chat_history": []},
        callbacks=[cl.AsyncLangchainCallbackHandler()],
    )

    msg.content = result["output"]
    await msg.update()
```

### File Upload Handling

```python
@cl.on_message
async def main(message: cl.Message):
    # Handle uploaded files
    if message.elements:
        for element in message.elements:
            if element.type == "file":
                content = element.content  # bytes
                filename = element.name
                await cl.Message(content=f"Received: {filename} ({len(content)} bytes)").send()
```

### Key Features

- Streaming responses out of the box
- File upload/download support
- Authentication (header-based, OAuth)
- Chat history persistence
- Tool call visualisation
- Multi-modal (images, PDFs)
- Custom UI elements (buttons, selects)
- LangChain/LangGraph callback integration

---

## Gradio (Quick Demo UIs)

Fast prototyping — build a UI in minutes. Great for demos and internal tools.

### Install

```bash
pip install gradio
```

### Chat Interface

```python
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def respond(message, history):
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                for pair in history for i, m in enumerate(pair) if m]
    messages.append({"role": "user", "content": message})

    response = llm.invoke(messages)
    return response.content

demo = gr.ChatInterface(
    respond,
    title="AI Assistant",
    description="Ask about product features and capabilities",
    examples=["What features are available?", "How does gradient boosting work?"],
    theme="soft",
)

demo.launch(share=False, server_port=7860)
```

### With Streaming

```python
import gradio as gr

def respond_stream(message, history):
    messages = []
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial

demo = gr.ChatInterface(respond_stream, title="AI Assistant")
demo.launch()
```

### Multi-Tab App (Chat + RAG + Upload)

```python
import gradio as gr

def chat_fn(message, history):
    return llm.invoke(message).content

def rag_query(question, file):
    # Process uploaded file and query
    return f"Answer based on {file.name}: ..."

with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant")

    with gr.Tab("Chat"):
        gr.ChatInterface(chat_fn)

    with gr.Tab("Document Q&A"):
        file_input = gr.File(label="Upload PDF")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        btn = gr.Button("Ask")
        btn.click(rag_query, [question, file_input], answer)

demo.launch()
```

---

## Streamlit (Data App + Chat)

Best when you need data visualisation alongside chat.

### Install

```bash
pip install streamlit
```

### Chat App

```python
# app.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("AI Assistant")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Session state for history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response
    with st.chat_message("assistant"):
        response = st.write_stream(llm.stream(st.session_state.messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
```

```bash
streamlit run app.py
```

### With Sidebar Controls

```python
import streamlit as st

st.sidebar.title("Settings")
model = st.sidebar.selectbox("Model", ["gemini-2.0-flash", "gemini-2.0-flash-lite"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
k_results = st.sidebar.number_input("Retrieval k", 1, 20, 5)

# Use in your chain
llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
```

---

## Mesop (Google — Python UI Framework)

Google's Python-native UI framework for AI apps. No JavaScript needed.

```bash
pip install mesop
```

```python
import mesop as me
import mesop.labs as mel

@me.page(path="/")
def page():
    mel.chat(transform, title="AI Assistant", bot_user="AI")

def transform(prompt: str, history: list[mel.ChatMessage]) -> str:
    response = llm.invoke(prompt)
    return response.content
```

```bash
mesop app.py
```

---

## Panel (HoloViz — Data Science UIs)

Good for data scientists who want chat + plots + widgets.

```bash
pip install panel
```

```python
import panel as pn
from langchain_google_genai import ChatGoogleGenerativeAI

pn.extension()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

async def respond(contents, user, chat_interface):
    response = llm.invoke(contents)
    yield response.content

chat = pn.chat.ChatInterface(callback=respond, sizing_mode="stretch_width")
chat.send("Ask me anything!", user="Assistant", respond=False)
chat.servable()
```

```bash
panel serve app.py --autoreload
```

---

## Open WebUI (Self-Hosted ChatGPT-Like)

Full-featured chat UI that connects to any OpenAI-compatible API (Ollama, vLLM, etc.).

```bash
docker run -d -p 3000:8080 \
    -v open-webui:/app/backend/data \
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    ghcr.io/open-webui/open-webui:main
```

Features: multi-model, RAG pipeline, user management, model marketplace, API keys.

---

## Comparison

| Framework | Language | Streaming | Auth | File Upload | Best For |
|-----------|----------|-----------|------|-------------|----------|
| Chainlit | Python | Yes | Yes | Yes | Production chat agents |
| Gradio | Python | Yes | Basic | Yes | Quick demos, sharing |
| Streamlit | Python | Yes | Community | Yes | Data apps + chat |
| Mesop | Python | Yes | No | Limited | Simple Google-style UIs |
| Panel | Python | Yes | Basic | Yes | Data science dashboards |
| Open WebUI | Docker | Yes | Yes | Yes | Self-hosted ChatGPT |

## When to Use What

| Scenario | Recommended |
|----------|-------------|
| Production agent with tools | Chainlit |
| Quick demo / hackathon | Gradio |
| Data exploration + chat | Streamlit |
| Internal tool (self-hosted LLM) | Open WebUI |
| Minimal Python UI | Mesop |
| Dashboards with chat | Panel |
| Custom React frontend | FastAPI backend + any React chat lib |

## Deployment

### Chainlit

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Gradio (HuggingFace Spaces — Free)

```yaml
# README.md header for HF Spaces
---
title: AI Assistant
emoji: 💰
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
---
```

### Streamlit (Streamlit Cloud — Free)

```
# requirements.txt
streamlit
langchain-google-genai
```

Push to GitHub → connect to streamlit.io → auto-deploys.

## Best Practices

- Use streaming for any response > 1 second — perceived latency drops dramatically
- Show intermediate steps (retrieval, tool calls) to build user trust
- Add copy buttons, source citations, and feedback (thumbs up/down)
- Persist conversation history server-side for observability
- Rate limit per user to control costs
- Handle errors gracefully — show friendly messages, not stack traces
- Test with real users early — chat UX is hard to get right from intuition alone
