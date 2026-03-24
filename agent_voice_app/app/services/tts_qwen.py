from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import websockets

from ..audio import b64decode_bytes
from ..logging_utils import get_logger

logger = get_logger("tts")


async def _noop_audio(_audio: bytes) -> None:
    return None


async def _noop_done() -> None:
    return None


async def _noop_ready() -> None:
    return None


class QwenTTSService:
    def __init__(
        self,
        on_audio: Callable[[bytes], Awaitable[None]] | None = None,
        on_done: Callable[[], Awaitable[None]] | None = None,
        on_ready: Callable[[], Awaitable[None]] | None = None,
        ws_factory: Callable[..., Awaitable[Any]] | None = None,
    ):
        self._on_audio = on_audio or _noop_audio
        self._on_done = on_done or _noop_done
        self._on_ready = on_ready or _noop_ready
        self._ws_factory = ws_factory or websockets.connect

        self._api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self._model = os.getenv("QWEN_TTS_MODEL", "qwen3-tts-flash-realtime")
        self._voice = os.getenv("QWEN_TTS_VOICE", "Cherry")
        self._language = os.getenv("QWEN_TTS_LANGUAGE", "Chinese")
        self._sample_rate = int(os.getenv("QWEN_TTS_SAMPLE_RATE", "24000"))
        self._mode = os.getenv("QWEN_TTS_MODE", "commit")

        self._ws: Optional[Any] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at = 0.0
        self._first_text_sent = False
        self._broken = False
        self._lock = asyncio.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_running(self) -> bool:
        return self._running and self._ws is not None and not self._broken

    @property
    def is_broken(self) -> bool:
        return self._broken

    def bind(
        self,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
        on_ready: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_done = on_done
        self._on_ready = on_ready or _noop_ready
        logger.info("tts_bound model=%s voice=%s", self._model, self._voice)

    def reset_for_next_turn(self) -> None:
        self._first_text_sent = False
        self._on_audio = _noop_audio
        self._on_done = _noop_done
        self._on_ready = _noop_ready

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return
            if not self._api_key:
                raise ValueError("Missing DASHSCOPE_API_KEY")
            self._started_at = time.perf_counter()
            self._broken = False
            logger.info(
                "tts_start_requested model=%s voice=%s sample_rate=%s mode=%s",
                self._model,
                self._voice,
                self._sample_rate,
                self._mode,
            )
            self._ws = await self._ws_factory(
                f"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model={self._model}",
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            self._running = True
            self._first_text_sent = False
            await self._ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "mode": self._mode,
                            "voice": self._voice,
                            "language_type": self._language,
                            "response_format": "pcm",
                            "sample_rate": self._sample_rate,
                        },
                    }
                )
            )
            logger.info("tts_ready model=%s elapsed_ms=%.2f", self._model, (time.perf_counter() - self._started_at) * 1000)
            if self._receive_task is None:
                self._receive_task = asyncio.create_task(self._receive_loop())
        await self._on_ready()

    async def send(self, text: str) -> None:
        if self.is_running and text:
            if not self._first_text_sent:
                self._first_text_sent = True
                logger.info("tts_first_text_sent text_chars=%s elapsed_ms=%.2f", len(text), (time.perf_counter() - self._started_at) * 1000)
            logger.debug("tts send text_chars=%s elapsed_ms=%.2f", len(text), (time.perf_counter() - self._started_at) * 1000)
            await self._ws.send(json.dumps({"type": "input_text_buffer.append", "text": text}))

    async def flush(self) -> None:
        if self.is_running:
            logger.debug("tts flush elapsed_ms=%.2f", (time.perf_counter() - self._started_at) * 1000)
            await self._ws.send(json.dumps({"type": "input_text_buffer.commit"}))

    async def cancel(self) -> None:
        self._running = False
        logger.info("tts_cancel elapsed_ms=%.2f", (time.perf_counter() - self._started_at) * 1000 if self._started_at else 0.0)
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _receive_loop(self) -> None:
        try:
            while self._running and self._ws:
                message = await self._ws.recv()
                logger.debug("tts recv message_chars=%s elapsed_ms=%.2f", len(message), (time.perf_counter() - self._started_at) * 1000)
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._broken = True
            logger.exception("tts_receive_error elapsed_ms=%.2f", (time.perf_counter() - self._started_at) * 1000)
        finally:
            if self._broken:
                self._running = False
                self._ws = None
            self._receive_task = None

    async def _handle_message(self, message: str) -> None:
        data = json.loads(message)
        event_type = data.get("type")
        if event_type == "response.audio.delta":
            logger.debug("tts audio delta bytes_b64=%s elapsed_ms=%.2f", len(data.get("delta", "")), (time.perf_counter() - self._started_at) * 1000)
            await self._on_audio(b64decode_bytes(data.get("delta", "")))
            return
        if event_type == "response.done":
            logger.info("tts_done elapsed_ms=%.2f", (time.perf_counter() - self._started_at) * 1000)
            await self._on_done()
