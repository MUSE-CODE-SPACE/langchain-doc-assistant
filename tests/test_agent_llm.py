"""Tests for the LLM-backed agent path using a scripted, offline chat model.

No API keys or network access are needed: the fake model emits one tool call
followed by a final answer, which exercises the LangChain 1.x ``create_agent``
loop (tool execution + final-text extraction) end to end.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

os.environ.setdefault("LLM_PROVIDER", "none")

from app.agents.document_agent import DocumentAssistantAgent, extract_final_text

TOOL_NAME = "list_documents"
TOOL_ARGS: dict[str, Any] = {}
FINAL_ANSWER = "No documents are loaded yet."


class FakeToolModel(GenericFakeChatModel):
    """Scripted chat model that accepts ``bind_tools`` (returns itself)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolModel:  # type: ignore[override]
        return self


def _scripted_model() -> FakeToolModel:
    return FakeToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": TOOL_NAME, "args": TOOL_ARGS, "id": "call_1", "type": "tool_call"}
                    ],
                ),
                AIMessage(content=[{"type": "text", "text": FINAL_ANSWER}]),
            ]
        )
    )


def test_agent_executes_tool_and_returns_final_text() -> None:
    agent = DocumentAssistantAgent(llm=_scripted_model())
    assert agent._agent_executor is not None

    result = agent._agent_executor.invoke({"messages": [HumanMessage(content="Which documents do you have?")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].name == TOOL_NAME
    assert str(tool_messages[0].content).strip()
    assert extract_final_text(result) == FINAL_ANSWER


def test_chat_uses_llm_reply() -> None:
    agent = DocumentAssistantAgent(llm=_scripted_model())
    reply = agent.chat("Which documents do you have?")
    assert reply.startswith(FINAL_ANSWER)


def test_chat_falls_back_to_keyword_router_when_model_fails() -> None:
    # An exhausted script makes the fake model raise on first use.
    agent = DocumentAssistantAgent(llm=FakeToolModel(messages=iter([])))
    reply = agent.chat("Which documents do you have?")
    assert isinstance(reply, str)
    assert reply.strip()


def test_extract_final_text_edge_cases() -> None:
    assert extract_final_text({}) == ""
    assert extract_final_text({"messages": []}) == ""
    assert extract_final_text({"messages": [HumanMessage(content="hi")]}) == ""
    assert extract_final_text({"messages": [AIMessage(content="  plain  ")]}) == "plain"
    blocks = AIMessage(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert extract_final_text({"messages": [blocks]}) == "ab"
