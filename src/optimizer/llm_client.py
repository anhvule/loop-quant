"""Claude API client with forced tool use.

`tool_choice={"type": "tool", "name": ...}` means the model MUST return a
structured tool call; it cannot return prose that some regex then has to parse.
Combined with the schemas in prompt_builder, the output arrives as validated JSON
or not at all.

`MockLLMClient` exists so the whole Module 4 cycle -- packaging, validation,
sandboxing, deploy, shadow, rollback -- is testable end-to-end with zero network
and zero cost. The optimizer drill uses it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.2


class LLMResponse:
    __slots__ = ("tool_input", "raw_text", "stop_reason", "usage", "model")

    def __init__(self, tool_input: dict[str, Any], raw_text: str = "",
                 stop_reason: str = "", usage: dict[str, Any] | None = None,
                 model: str = "") -> None:
        self.tool_input = tool_input
        self.raw_text = raw_text
        self.stop_reason = stop_reason
        self.usage = usage or {}
        self.model = model


class LLMClient(Protocol):
    async def call_tool(self, system: str, messages: list[dict[str, Any]],
                        tool: dict[str, Any], temperature: float = DEFAULT_TEMPERATURE
                        ) -> LLMResponse: ...


class AnthropicLLMClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run the optimizer with "
                "MockLLMClient (see scripts/optimizer_drill.py)."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any = None

    def _ensure(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def call_tool(self, system: str, messages: list[dict[str, Any]],
                        tool: dict[str, Any],
                        temperature: float = DEFAULT_TEMPERATURE) -> LLMResponse:
        client = self._ensure()
        resp = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
            tools=[tool],
            # Force the structured path: no free-text answer is accepted.
            tool_choice={"type": "tool", "name": tool["name"]},
        )

        tool_input: dict[str, Any] | None = None
        text_parts: list[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                tool_input = dict(block.input)
            elif getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        if tool_input is None:
            raise RuntimeError(
                f"model did not call {tool['name']} despite forced tool_choice "
                f"(stop_reason={resp.stop_reason})"
            )

        usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
                 "output_tokens": getattr(resp.usage, "output_tokens", None)}
        log.info("LLM %s -> %s (in=%s out=%s)", self.model, tool["name"],
                 usage["input_tokens"], usage["output_tokens"])
        return LLMResponse(tool_input, "\n".join(text_parts), str(resp.stop_reason),
                           usage, self.model)


class MockLLMClient:
    """Replays canned tool inputs, keyed by tool name.

    Each queue entry may be a dict (returned as tool_input) or an Exception
    (raised) -- which is how the drill exercises the malformed-output path.
    """

    def __init__(self, responses: dict[str, list[Any]] | None = None) -> None:
        self.responses: dict[str, list[Any]] = responses or {}
        self.calls: list[dict[str, Any]] = []

    def queue(self, tool_name: str, payload: Any) -> None:
        self.responses.setdefault(tool_name, []).append(payload)

    async def call_tool(self, system: str, messages: list[dict[str, Any]],
                        tool: dict[str, Any],
                        temperature: float = DEFAULT_TEMPERATURE) -> LLMResponse:
        name = tool["name"]
        self.calls.append({"tool": name, "system": system, "messages": messages,
                           "temperature": temperature})
        q = self.responses.get(name) or []
        if not q:
            raise RuntimeError(f"MockLLMClient has no queued response for {name!r}")
        item = q.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(dict(item), "", "tool_use", {"input_tokens": 0, "output_tokens": 0},
                           "mock")
