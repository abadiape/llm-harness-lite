# Build a purpose-specific agent

This guide grows one small support agent in steps: first a model and one Python
tool, then optional retrieval and MCP tools. You can stop at any step. The
harness does not require a configuration file, a vector database, or MCP.

## 1. Choose a provider

Install the extra for the provider you want. For example, for OpenAI:

```bash
pip install "llm-harness-lite[openai]"
export OPENAI_API_KEY="your-api-key"
```

Keep credentials in environment variables or a secret manager, not in source
control. Other provider options and local-model setup are in
[Choosing an LLM and configuring your app](configuration.md).

## 2. Give the agent one capability

An agent is an LLM, a registry of tools your application implements, and
instructions that describe the agent's job. This example uses a small
in-memory order lookup so it can run without a database or external service:

```python
import llm_harness_lite

llm = llm_harness_lite.create_llm("openai", "gpt-4o-mini")
tools = llm_harness_lite.ToolRegistry()

orders = {"A-100": "shipped"}

@llm_harness_lite.tool(tools)
def lookup_order(order_id: str) -> str:
    """Look up the status of an order by its ID."""
    return orders.get(order_id, "No matching order found.")

agent = llm_harness_lite.Agent(
    llm,
    tools,
    system="You are a support assistant. Use lookup_order for order-status questions. Do not guess.",
    max_steps=5,
    budget_usd=0.02,
)

try:
    result = agent.run("What is the status of order A-100?")
    print(result.answer)
    print(f"Stopped: {result.stop_reason}; model calls: {result.steps}")
finally:
    llm.close()
```

`@llm_harness_lite.tool(tools)` registers the function and derives a tool schema from its
name, docstring, and annotated parameters. When the model requests that tool,
the harness calls the Python function and sends its result back to the model.
Replace the example dictionary with an application function that accesses only
the data this agent is allowed to use.

The system prompt guides the model, but it is not an access-control boundary.
Enforce permissions, validate inputs, and limit what each tool can do in the
tool implementation. `max_steps` bounds the agent loop; `budget_usd` limits
model spend when provider cost data is available.

## 3. Change the model without changing the agent

The tools and agent setup stay the same when you switch providers. Change the
`create_llm` call and install the matching optional dependency:

```python
# Anthropic: pip install "llm-harness-lite[anthropic]"
llm = llm_harness_lite.create_llm("anthropic", "your-model-name")

# Ollama's OpenAI-compatible endpoint: pip install "llm-harness-lite[openai]"
llm = llm_harness_lite.create_llm(
    "openai_compat",
    "qwen3:8b",
    base_url="http://localhost:11434/v1",
)
```

Choose one `create_llm()` call before constructing the agent. The examples
above are alternatives, not calls to run one after another.

For Ollama, start the Ollama service and pull the model first. The supported
provider names are `openai`, `anthropic`, `openrouter`, `openai_compat`,
`ollama`, `vllm`, and `lmstudio`. Provider SDKs are optional; install only the
extra you use.

## 4. Use it from an async application

In an existing event loop, await `arun()` rather than calling `run()`. Built-in
backends support async cleanup:

```python
async def handle_request(agent, question: str) -> str:
    result = await agent.arun(question)
    return result.answer
```

When your application owns the backend lifecycle, close it with
`await llm.aclose()` after use. See [RAG and MCP integrations](integrations.md)
for an async example that also manages an MCP connection.

## Optional: Add a sub-agent for a distinct job

Most applications can grow one agent by adding focused tools. If a specialist
needs its own prompt and, optionally, a restricted tool set, create another
`Agent` and expose it through an async tool on the parent. Using `llm`, `tools`,
and `agent` from step 2, register this before calling `agent.run()`:

```python
specialist_tools = llm_harness_lite.ToolRegistry()
# Register tools that should be available only to the specialist.
specialist = llm_harness_lite.Agent(
    llm,
    specialist_tools,
    system="Handle account-access questions only.",
    max_steps=3,
    budget_usd=0.01,
)

@llm_harness_lite.tool(tools)
async def ask_account_specialist(question: str) -> str:
    result = await specialist.arun(question)
    return result.answer
```

This is ordinary tool composition, not built-in delegation. The child has its
own step and budget limits, and its usage is not included in the parent's
`AgentResult`; enforce or aggregate a total budget yourself. Use a child only
when the separation justifies the extra latency and cost.

## Next steps

- Add retrieval or remote MCP tools in [RAG and MCP integrations](integrations.md).
- Move model and agent settings into your application's config in
  [Choosing an LLM and configuring your app](configuration.md).
- Browse the runnable scripts in [`examples/`](../examples/).
