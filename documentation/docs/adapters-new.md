# New Adapters

FastAgentic v1.1 adds support for additional agent frameworks.

## Semantic Kernel

Adapter for Microsoft Semantic Kernel.

```python
import semantic_kernel as sk
from fastagentic.adapters import SemanticKernelAdapter

# Create kernel
kernel = sk.Kernel()
kernel.add_service(...)

# Function-based adapter
adapter = SemanticKernelAdapter(
    kernel,
    function_name="chat",
    plugin_name="ChatPlugin",
)

# Agent-based adapter
agent = create_agent(kernel)
adapter = SemanticKernelAdapter(kernel, agent=agent)

@agent_endpoint(path="/chat", runnable=adapter, stream=True)
async def chat(input: ChatInput) -> ChatOutput:
    ...
```

### Features
- Function and agent support
- Streaming with `invoke_stream`
- Plugin integration
- Prompt execution settings

## AutoGen

Adapter for Microsoft AutoGen multi-agent conversations.

```python
from autogen import AssistantAgent, UserProxyAgent
from fastagentic.adapters import AutoGenAdapter

# Create agents
assistant = AssistantAgent("assistant", llm_config=config)
user_proxy = UserProxyAgent("user", code_execution_config=False)

# Two-agent adapter
adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=assistant,
    max_turns=10,
)

# Group chat adapter
from autogen import GroupChat
group_chat = GroupChat(agents=[agent1, agent2, agent3])
adapter = AutoGenAdapter(
    initiator=user_proxy,
    recipient=assistant,
    group_chat=group_chat,
)

@agent_endpoint(path="/collaborate", runnable=adapter, stream=True)
async def collaborate(input: Input) -> Output:
    ...
```

### Features
- Two-agent conversations
- Group chat support
- Message streaming
- Tool call events

## LlamaIndex

Adapter for LlamaIndex agents, query engines, and chat engines.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from fastagentic.adapters import LlamaIndexAdapter

# Query engine adapter
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
adapter = LlamaIndexAdapter(query_engine=query_engine)

# Agent adapter
agent = ReActAgent.from_tools(tools, llm=llm)
adapter = LlamaIndexAdapter(agent=agent)

# Chat engine adapter
chat_engine = index.as_chat_engine()
adapter = LlamaIndexAdapter(chat_engine=chat_engine)

@agent_endpoint(path="/query", runnable=adapter, stream=True)
async def query(input: QueryInput) -> QueryOutput:
    ...
```

### Features
- Query engines with RAG
- ReAct agents
- Chat engines with memory
- Source node streaming

## DSPy

Adapter for DSPy modules and compiled programs.

```python
import dspy
from fastagentic.adapters import DSPyAdapter, DSPyProgramAdapter

# Signature
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Module adapter
qa_module = dspy.ChainOfThought(QA)
adapter = DSPyAdapter(qa_module)

# With trace information
adapter = DSPyAdapter(qa_module, trace=True)

# Compiled program adapter
from dspy.teleprompt import BootstrapFewShot
teleprompter = BootstrapFewShot(metric=my_metric)
compiled = teleprompter.compile(RAG(), trainset=trainset)
adapter = DSPyProgramAdapter(compiled)

@agent_endpoint(path="/qa", runnable=adapter)
async def qa(input: QAInput) -> QAOutput:
    ...
```

### Features
- Module and signature support
- Compiled program support
- Trace information (rationale, reasoning)
- Retrieval result integration

## Community Adapter SDK

Build custom adapters with the SDK.

```python
from fastagentic.adapters.sdk import (
    CommunityAdapter,
    AdapterMetadata,
    SimpleAdapter,
    register_adapter,
)
from fastagentic.types import StreamEvent, StreamEventType

# Define adapter metadata
metadata = AdapterMetadata(
    name="my-adapter",
    version="1.0.0",
    description="My custom adapter",
    author="Your Name",
    framework="myframework",
    capabilities=["streaming", "tools"],
)

# Simple function adapter
def my_agent(input: str) -> str:
    return f"Response: {input}"

adapter = SimpleAdapter(invoke_fn=my_agent, metadata=metadata)

# Full custom adapter
@register_adapter
class MyAdapter(CommunityAdapter):
    metadata = metadata

    def __init__(self, agent):
        self.agent = agent

    async def invoke(self, input, ctx):
        return await self.agent.run(input)

    async def stream(self, input, ctx):
        async for chunk in self.agent.stream(input):
            yield StreamEvent(
                type=StreamEventType.TOKEN,
                data={"content": chunk},
            )
        yield StreamEvent(type=StreamEventType.DONE, data={})
```

### Adapter Registry

```python
from fastagentic.adapters.sdk import AdapterRegistry, get_registry

# Global registry
registry = get_registry()
adapters = registry.list_adapters()

# Find by framework
sk_adapters = registry.find_by_framework("semantic-kernel")

# Find by capability
streaming_adapters = registry.find_by_capability("streaming")
```
