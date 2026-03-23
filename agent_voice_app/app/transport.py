from __future__ import annotations

import json
import uuid

from fastapi import WebSocket

from .types import Phase


class BrowserTransport:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_id = f"browser-{uuid.uuid4().hex[:12]}"

    async def send_ready(self) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "ready",
                    "stream_id": self.stream_id,
                }
            )
        )

    async def send_phase(self, phase: Phase) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "phase",
                    "phase": phase.value,
                }
            )
        )

    async def send_transcript(
        self,
        speaker: str,
        text: str,
        final: bool,
        response_id: str | None = None,
    ) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "transcript",
                    "speaker": speaker,
                    "text": text,
                    "final": final,
                    "response_id": response_id,
                }
            )
        )

    async def send_audio(self, audio_b64: str, sample_rate: int, response_id: str) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "audio",
                    "audio_b64": audio_b64,
                    "sample_rate": sample_rate,
                    "response_id": response_id,
                }
            )
        )

    async def clear_audio(self, response_id: str | None = None) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "clear_audio",
                    "response_id": response_id,
                }
            )
        )

    async def send_error(self, message: str) -> None:
        await self.websocket.send_text(json.dumps({"type": "error", "message": message}))

    async def send_turn_debug(self, payload: dict) -> None:
        await self.websocket.send_text(
            json.dumps(
                {
                    "type": "turn_debug",
                    "payload": payload,
                }
            )
        )
