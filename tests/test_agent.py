import pytest

from grok_bot.agent import Agent, TagHoldback, parse_text_tool_calls
from grok_bot.tools import Tool, ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn


def echo_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                "echo",
                "Echo text",
                {"type": "object", "properties": {"text": {"type": "string"}}},
                lambda text: f"echo: {text}",
            ),
        ]
    )


async def collect(agent, messages):
    return [e async for e in agent.run(messages)]


@pytest.mark.anyio
async def test_plain_answer_streams_tokens():
    llm = FakeLLM([text_turn("Hello", " world")])
    agent = Agent(llm, echo_registry(), "m")
    msgs = [{"role": "user", "content": "hi"}]
    events = await collect(agent, msgs)
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Hello world"
    assert msgs[-1] == {"role": "assistant", "content": "Hello world"}
    assert llm.calls[0]["tools"][0]["function"]["name"] == "echo"


@pytest.mark.anyio
async def test_structured_tool_call_roundtrip():
    llm = FakeLLM([tool_turn("echo", '{"text": "ping"}'), text_turn("got it")])
    agent = Agent(llm, echo_registry(), "m")
    msgs = [{"role": "user", "content": "x"}]
    events = await collect(agent, msgs)
    types = [e["type"] for e in events]
    assert types == ["tool_call", "tool_result", "token"]
    assert events[0]["arguments"] == {"text": "ping"}
    assert events[1]["output"] == "echo: ping" and not events[1]["is_error"]
    assert msgs[1]["tool_calls"][0]["id"] == "call_1"
    assert msgs[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "echo",
        "content": "echo: ping",
    }
    second_call_msgs = llm.calls[1]["messages"]
    assert second_call_msgs[-1]["role"] == "tool"
    assert agent.new_messages == msgs[1:]


@pytest.mark.anyio
async def test_text_tool_call_fallback_hidden_from_stream():
    llm = FakeLLM(
        [
            text_turn(
                "Let me check. <tool",
                '_call>{"name": "echo", "arguments": {"text": "a"}}',
                "</tool_call>",
            ),
            text_turn("done"),
        ]
    )
    agent = Agent(llm, echo_registry(), "m")
    msgs = [{"role": "user", "content": "x"}]
    events = await collect(agent, msgs)
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "<tool" not in tokens
    assert tokens == "Let me check. done"
    assert [e["type"] for e in events if e["type"] != "token"] == ["tool_call", "tool_result"]
    assert msgs[1]["content"] == "Let me check."


@pytest.mark.anyio
async def test_unknown_tool_reports_error_and_continues():
    llm = FakeLLM([tool_turn("nope", "{}"), text_turn("ok")])
    events = await collect(Agent(llm, echo_registry(), "m"), [{"role": "user", "content": "x"}])
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["is_error"] and "Unknown tool" in result["output"]


@pytest.mark.anyio
async def test_max_steps_forces_final_answer_without_tools():
    llm = FakeLLM([tool_turn("echo", '{"text": "1"}'), text_turn("final")])
    agent = Agent(llm, echo_registry(), "m", max_steps=2)
    await collect(agent, [{"role": "user", "content": "x"}])
    assert llm.calls[0]["tools"] and llm.calls[1]["tools"] == []


def test_tag_holdback_passes_lookalikes():
    h = TagHoldback()
    out = h.feed("a <") + h.feed("b> c <tool_ca") + h.feed("ll>{}</tool_call> d")
    assert out + h.flush() == "a <b> c  d"


def test_parse_text_tool_calls_ignores_bad_json():
    text, calls = parse_text_tool_calls(
        'hi <tool_call>{bad}</tool_call> <tool_call>{"name":"x","parameters":{"k":1}}</tool_call>'
    )
    assert [c.name for c in calls] == ["x"] and calls[0].arguments == {"k": 1}
    assert "<tool_call>" not in text
