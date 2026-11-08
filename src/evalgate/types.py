"""The contract between the harness and any agent it evaluates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: Any = None


@dataclass
class Response:
    answer: str
    citations: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # what the answer may rely on: retrieved text, tool results
    tool_calls: list[ToolCall] = field(default_factory=list)
    abstained: bool = False
    refused: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Agent(Protocol):
    name: str

    def respond(self, question: str) -> Response: ...
