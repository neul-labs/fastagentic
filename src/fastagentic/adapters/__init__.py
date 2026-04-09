"""Adapters for various agent frameworks.

Each adapter wraps a specific framework (PydanticAI, LangGraph, CrewAI, LangChain,
Semantic Kernel, AutoGen, LlamaIndex, DSPy) and provides a unified interface
for FastAgentic's deployment features.

Optional imports are lazy-loaded to avoid requiring all dependencies.
"""

from fastagentic.adapters.base import AdapterContext, BaseAdapter
from fastagentic.types import StreamEvent

__all__ = [
    "BaseAdapter",
    "AdapterContext",
    "StreamEvent",
    # Original adapters (lazy-loaded)
    "PydanticAIAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "LangChainAdapter",
    # New adapters (lazy-loaded)
    "SemanticKernelAdapter",
    "AutoGenAdapter",
    "LlamaIndexAdapter",
    "DSPyAdapter",
    "DSPyProgramAdapter",
]


def __getattr__(name: str) -> type:
    """Lazy-load optional adapters."""
    if name == "PydanticAIAdapter":
        from fastagentic.adapters.pydanticai import PydanticAIAdapter

        return PydanticAIAdapter
    elif name == "LangGraphAdapter":
        from fastagentic.adapters.langgraph import LangGraphAdapter

        return LangGraphAdapter
    elif name == "CrewAIAdapter":
        from fastagentic.adapters.crewai import CrewAIAdapter

        return CrewAIAdapter
    elif name == "LangChainAdapter":
        from fastagentic.adapters.langchain import LangChainAdapter

        return LangChainAdapter
    elif name == "SemanticKernelAdapter":
        from fastagentic.adapters.semantic_kernel import SemanticKernelAdapter

        return SemanticKernelAdapter
    elif name == "AutoGenAdapter":
        from fastagentic.adapters.autogen import AutoGenAdapter

        return AutoGenAdapter
    elif name == "LlamaIndexAdapter":
        from fastagentic.adapters.llamaindex import LlamaIndexAdapter

        return LlamaIndexAdapter
    elif name == "DSPyAdapter":
        from fastagentic.adapters.dspy import DSPyAdapter

        return DSPyAdapter
    elif name == "DSPyProgramAdapter":
        from fastagentic.adapters.dspy import DSPyProgramAdapter

        return DSPyProgramAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
