import asyncio
import json

import pytest

from app.audio import unpack_binary_audio_message
from app.transport import BrowserTransport


class FakeWebSocket:
    def __init__(self):
        self.sent_text = []
        self.sent_bytes = []
        self.block = None

    async def send_text(self, text: str):
        if self.block is not None:
            await self.block.wait()
        await asyncio.sleep(0)
        self.sent_text.append(json.loads(text))

    async def send_bytes(self, payload: bytes):
        if self.block is not None:
            await self.block.wait()
        await asyncio.sleep(0)
        self.sent_bytes.append(payload)


@pytest.mark.asyncio
async def test_clear_audio_preempts_media_queue():
    control = FakeWebSocket()
    audio = FakeWebSocket()
    transport = BrowserTransport(control)
    await transport.start()
    await transport.bind_audio_socket(audio)

    await transport.send_audio_chunk(b"\x00\x00" * 1600, 16000, "resp-1", chunk_seq=1, generated_at_ms=1.0)
    await transport.clear_audio("resp-1")
    await asyncio.sleep(0)
    await transport.stop()

    assert control.sent_text[0]["type"] == "interrupt"
    assert control.sent_text[0]["kind"] == "clear_audio"
    assert audio.sent_bytes == []


@pytest.mark.asyncio
async def test_invalidate_response_drops_stale_media():
    control = FakeWebSocket()
    audio = FakeWebSocket()
    control.block = asyncio.Event()
    audio.block = asyncio.Event()
    transport = BrowserTransport(control)
    await transport.start()
    await transport.bind_audio_socket(audio)

    audio_task = asyncio.create_task(
        transport.send_audio_chunk(b"\x00\x00" * 1600, 16000, "resp-1", chunk_seq=1, generated_at_ms=1.0)
    )
    token_task = asyncio.create_task(transport.send_llm_token("hello", "resp-1"))
    await asyncio.sleep(0)
    await transport.invalidate_response("resp-1")
    control.block.set()
    audio.block.set()
    await asyncio.sleep(0)
    await token_task
    await audio_task

    assert control.sent_text == []
    assert audio.sent_bytes == []
    assert transport.queue_metrics()["tts_chunk_drop_count"] == 1
    await transport.stop()


@pytest.mark.asyncio
async def test_duplicate_turn_event_is_suppressed():
    websocket = FakeWebSocket()
    transport = BrowserTransport(websocket)
    await transport.start()

    await transport.send_turn_event("idle")
    await transport.send_turn_event("idle")
    await transport.send_turn_event("speech_start")
    await asyncio.sleep(0)
    await transport.stop()

    turn_events = [item for item in websocket.sent_text if item["type"] == "turn_event"]
    assert [item["kind"] for item in turn_events] == ["idle", "speech_start"]


@pytest.mark.asyncio
async def test_audio_chunk_is_binary_frame():
    control = FakeWebSocket()
    audio = FakeWebSocket()
    transport = BrowserTransport(control)
    await transport.start()
    await transport.bind_audio_socket(audio)

    pcm = b"\x01\x02" * 960
    await transport.send_audio_chunk(pcm, 16000, "resp-1", chunk_seq=7, generated_at_ms=12.0)
    await asyncio.sleep(0)
    await transport.stop()

    header, payload = unpack_binary_audio_message(audio.sent_bytes[0])
    assert header["type"] == "tts_chunk"
    assert header["response_id"] == "resp-1"
    assert header["chunk_seq"] == 7
    assert payload == pcm
