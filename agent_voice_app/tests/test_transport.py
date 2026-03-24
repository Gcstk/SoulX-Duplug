import asyncio
import json

import pytest

from app.transport import BrowserTransport


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.block = None

    async def send_text(self, text: str):
        if self.block is not None:
            await self.block.wait()
        await asyncio.sleep(0)
        self.sent.append(json.loads(text))


@pytest.mark.asyncio
async def test_clear_audio_preempts_media_queue():
    websocket = FakeWebSocket()
    transport = BrowserTransport(websocket)
    await transport.start()

    audio_task = asyncio.create_task(transport.send_audio("Zm9v", 24000, "resp-1"))
    clear_task = asyncio.create_task(transport.clear_audio("resp-1"))

    await asyncio.gather(audio_task, clear_task)
    await transport.stop()

    sent_types = [item["type"] for item in websocket.sent]
    assert sent_types[0] == "clear_audio"
    assert "audio" in sent_types


@pytest.mark.asyncio
async def test_invalidate_response_drops_stale_media():
    websocket = FakeWebSocket()
    transport = BrowserTransport(websocket)

    audio_task = asyncio.create_task(transport.send_audio("Zm9v", 24000, "resp-1"))
    transcript_task = asyncio.create_task(transport.send_transcript("assistant", "hello", False, "resp-1"))
    while sum(len(queue) for queue in transport._queues.values()) < 2:
        await asyncio.sleep(0)
    await transport.invalidate_response("resp-1")

    await audio_task
    await transcript_task
    assert websocket.sent == []
    assert transport.queue_metrics()["stale_media_dropped_count"] == 2
