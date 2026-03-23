import pytest

from app.services.duplug_client import DuplugClient


@pytest.mark.asyncio
async def test_duplug_nonidle_and_speak_mapping():
    events = []

    async def on_start():
        events.append(("start", None))

    async def on_interim(text: str):
        events.append(("interim", text))

    async def on_final(text: str):
        events.append(("final", text))

    async def on_idle():
        events.append(("idle", None))

    client = DuplugClient(
        on_user_speech_start=on_start,
        on_user_interim=on_interim,
        on_user_turn_final=on_final,
        on_turn_idle=on_idle,
        on_debug=None,
        url="ws://unused",
    )

    await client._process_turn_state({"state": "nonidle", "asr_buffer": "你好"})
    await client._process_turn_state({"state": "nonidle", "asr_buffer": "你好"})
    await client._process_turn_state({"state": "speak", "text": "你好啊"})

    assert events == [
        ("start", None),
        ("interim", "你好"),
        ("final", "你好啊"),
    ]


@pytest.mark.asyncio
async def test_duplug_idle_is_forwarded():
    events = []

    async def on_start():
        return None

    async def on_interim(_text: str):
        return None

    async def on_final(_text: str):
        return None

    async def on_idle():
        events.append("idle")

    client = DuplugClient(
        on_user_speech_start=on_start,
        on_user_interim=on_interim,
        on_user_turn_final=on_final,
        on_turn_idle=on_idle,
        on_debug=None,
        url="ws://unused",
    )

    await client._process_turn_state({"state": "idle"})
    assert events == ["idle"]
