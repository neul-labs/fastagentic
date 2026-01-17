# Semantic Kernel Adapter

The Semantic Kernel adapter wraps [Microsoft Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/) for deployment through FastAgentic. Deploy SK functions, plugins, and agents with full checkpointing and streaming.

## TL;DR

Wrap any Semantic Kernel `Kernel` and get REST + MCP + streaming + durability + function call events.

## Why Semantic Kernel + FastAgentic?

Semantic Kernel excels at enterprise AI orchestration with Microsoft ecosystem integration. FastAgentic adds:

| Capability | Semantic Kernel | FastAgentic |
|------------|-----------------|-------------|
| Plugins | Built-in | Inherited |
| Planners | Built-in | Inherited |
| Azure OpenAI | Native | Inherited |
| .NET compatibility | Built-in | N/A |
| REST API | Manual | Automatic |
| MCP Protocol | No | Yes |
| Checkpoints | No | Built-in |
| Function events | Filters | Automatic |
| Auth & Policy | Application code | Middleware |

**Orchestrate with Semantic Kernel. Deploy with FastAgentic.**

## Before FastAgentic

```python
from fastapi import FastAPI
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

app = FastAPI()

kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(
    deployment_name="gpt-4",
    endpoint="https://my-endpoint.openai.azure.com/",
    api_key="..."
))

@app.post("/chat")
async def chat(message: str):
    result = await kernel.invoke_prompt(message)
    return {"response": str(result)}

# No streaming, no checkpoints, no function tracking, no auth...
```

## After FastAgentic

```python
from fastagentic import App, agent_endpoint
from fastagentic.adapters.semantic_kernel import SemanticKernelAdapter
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(
    deployment_name="gpt-4",
    endpoint="https://my-endpoint.openai.azure.com/",
    api_key="..."
))

app = App(
    title="SK Chat",
    oidc_issuer="https://auth.company.com",
    durable_store="redis://localhost",
)

@agent_endpoint(
    path="/chat",
    runnable=SemanticKernelAdapter(kernel),
    stream=True,
    durable=True,
)
async def chat(message: str) -> str:
    pass
```

## What You Get

### Automatic Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /chat` | Run kernel synchronously |
| `POST /chat/stream` | Run with token streaming |
| `GET /chat/{run_id}` | Get run status and result |
| `POST /chat/{run_id}/resume` | Resume from checkpoint |

### Semantic Kernel-Specific Features

**Prompt Invocation**

```python
adapter = SemanticKernelAdapter(kernel)

# Invokes kernel.invoke_prompt() with the input message
```

**Function Invocation**

```python
adapter = SemanticKernelAdapter(
    kernel,
    function_name="summarize",
    plugin_name="TextPlugin",
)

# Invokes specific function from plugin
```

**Agent Invocation**

```python
from semantic_kernel.agents import ChatCompletionAgent

agent = ChatCompletionAgent(kernel=kernel, name="Assistant")
adapter = SemanticKernelAdapter(kernel, agent=agent)

# Invokes agent with chat history tracking
```

## Configuration Options

### SemanticKernelAdapter Constructor

```python
SemanticKernelAdapter(
    kernel: Kernel,
    function_name: str | None = None,
    plugin_name: str | None = None,
    agent: Agent | None = None,
    settings: PromptExecutionSettings | None = None,
    checkpoint_functions: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel` | `Kernel` | required | Semantic Kernel instance |
| `function_name` | `str` | `None` | Function to invoke (if using functions) |
| `plugin_name` | `str` | `None` | Plugin containing the function |
| `agent` | `Agent` | `None` | SK Agent instance |
| `settings` | `PromptExecutionSettings` | `None` | Prompt execution settings |
| `checkpoint_functions` | `bool` | `True` | Checkpoint after function calls |

### Builder Methods

```python
# Target specific function
adapter = SemanticKernelAdapter(kernel).with_function("chat", "ChatPlugin")

# Configure settings
adapter = SemanticKernelAdapter(kernel).with_settings(my_settings)

# Configure checkpointing
adapter = SemanticKernelAdapter(kernel).with_checkpoints(enabled=True)

# Use agent
adapter = SemanticKernelAdapter(kernel).with_agent(agent)
```

## Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `node_start` | Resume from checkpoint | `{name: "__resume__", checkpoint}` |
| `token` | Output token | `{content}` |
| `tool_call` | Function invoked | `{name, input}` |
| `tool_result` | Function returns | `{name, output}` |
| `checkpoint` | State saved | `{step}` or `{step, function}` |
| `done` | Run complete | `{result}` |

## Checkpoint State

The adapter automatically creates checkpoints containing:

```python
{
    "state": {
        "completed": bool,
        "function": str,     # Function name (for function invocations)
    },
    "messages": [            # Chat history (for agents)
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ],
    "function_calls": [      # Function invocation history
        {"name": "TextPlugin.summarize", "input": {...}, "output": "..."},
    ],
    "context": {"result": "..."},
}
```

**Checkpoint Triggers:**
- After function completion (if `checkpoint_functions=True`)
- On run completion

**Resume Behavior:**
When resuming, the adapter:
1. Restores `chat_history` from checkpoint
2. Sets `_is_resumed = True` on the run context
3. Emits a `NODE_START` event with `name: "__resume__"`

## Common Patterns

### Prompt-Based Chat

```python
kernel = sk.Kernel()
kernel.add_service(AzureChatCompletion(...))

@agent_endpoint(
    path="/chat",
    runnable=SemanticKernelAdapter(kernel),
    stream=True,
)
async def chat(message: str) -> str:
    pass
```

### Plugin Function

```python
from semantic_kernel.functions import kernel_function

class TextPlugin:
    @kernel_function(name="summarize", description="Summarize text")
    async def summarize(self, text: str) -> str:
        # Summarization logic
        return summary

kernel.add_plugin(TextPlugin(), "TextPlugin")

@agent_endpoint(
    path="/summarize",
    runnable=SemanticKernelAdapter(
        kernel,
        function_name="summarize",
        plugin_name="TextPlugin",
    ),
    stream=True,
)
async def summarize(text: str) -> str:
    pass
```

### Agent with Tools

```python
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

kernel.add_plugin(SearchPlugin(), "Search")

agent = ChatCompletionAgent(
    kernel=kernel,
    name="ResearchAssistant",
    instructions="You are a helpful research assistant.",
    execution_settings=AzureChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    ),
)

@agent_endpoint(
    path="/research",
    runnable=SemanticKernelAdapter(kernel, agent=agent),
    stream=True,
    durable=True,
)
async def research(question: str) -> str:
    pass
```

### With Custom Settings

```python
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

settings = AzureChatPromptExecutionSettings(
    temperature=0.7,
    max_tokens=1000,
)

@agent_endpoint(
    path="/creative",
    runnable=SemanticKernelAdapter(kernel, settings=settings),
    stream=True,
)
async def creative_write(prompt: str) -> str:
    pass
```

## Azure Integration

Semantic Kernel has native Azure OpenAI support:

```python
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel.add_service(AzureChatCompletion(
    deployment_name="gpt-4",
    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
))

# Works seamlessly with FastAgentic
adapter = SemanticKernelAdapter(kernel)
```

## Next Steps

- [Adapters Overview](index.md) - Compare adapters
- [AutoGen Adapter](autogen.md) - For multi-agent conversations
- [LangChain Adapter](langchain.md) - For chain deployment
