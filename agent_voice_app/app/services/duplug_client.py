from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from ..logging_utils import get_logger
from ..audio import TARGET_SAMPLE_RATE, b64encode_bytes, pcm16_resample

CHUNK_DURATION_MS = 160
logger = get_logger("duplug")


class DuplugClient:
    def __init__(
        self,
        on_user_speech_start: Callable[[], Awaitable[None]],
        on_user_interim: Callable[[str], Awaitable[None]],
        on_user_turn_final: Callable[[str], Awaitable[None]],
        on_turn_idle: Callable[[], Awaitable[None]],
        on_error: Callable[[str], Awaitable[None]] | None = None,
        on_debug: Callable[[dict], Awaitable[None]] | None = None,
        ws_factory: Callable[..., Awaitable[Any]] | None = None,
        url: str | None = None,
        timeout: float | None = None,
    ):
        self._on_user_speech_start = on_user_speech_start
        self._on_user_interim = on_user_interim
        self._on_user_turn_final = on_user_turn_final
        self._on_turn_idle = on_turn_idle
        self._on_error = on_error
        self._on_debug = on_debug
        self._ws_factory = ws_factory or websockets.connect
        self._url = (url or os.getenv("DUPLUG_WS_URL", "")).strip()
        self._timeout = timeout if timeout is not None else float(os.getenv("DUPLUG_WS_TIMEOUT", "10"))

        self._ws: Optional[Any] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._running = False
        self._turn_started = False
        self._last_interim = ""
        self._session_id = f"agent-{uuid.uuid4().hex}"
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)
        self._started_at = time.perf_counter()
        self._sent_chunks = 0
        self._received_events = 0

    async def start(self) -> None:
        if self._running:
            return
        if not self._url:
            raise ValueError("Missing DUPLUG_WS_URL")
        self._ws = await self._ws_factory(
            self._url,
            open_timeout=self._timeout,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=self._timeout,
        )
        self._running = True
        self._turn_started = False
        self._last_interim = ""
        self._session_id = f"agent-{uuid.uuid4().hex}"
        self._send_queue = asyncio.Queue(maxsize=8)
        self._started_at = time.perf_counter()
        self._sent_chunks = 0
        self._received_events = 0
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._send_task = asyncio.create_task(self._send_loop())
        logger.info("duplug start session_id=%s url=%s timeout=%s", self._session_id, self._url, self._timeout)

    async def send_audio(
        self,
        pcm16_bytes: bytes,
        sample_rate: int,
        captured_at_ms: float | None = None,
        received_at_perf: float | None = None,
    ) -> None:
        if not self._running:
            return
        resampled = pcm16_resample(pcm16_bytes, sample_rate, TARGET_SAMPLE_RATE)
        if not resampled:
            return
        logger.info(
            "duplug enqueue session_id=%s input_bytes=%s input_sample_rate=%s resampled_bytes=%s queue_before=%s",
            self._session_id,
            len(pcm16_bytes),
            sample_rate,
            len(resampled),
            self._send_queue.qsize(),
        )
        packet = {
            "pcm16": resampled,
            "captured_at_ms": captured_at_ms,
            "received_at_perf": received_at_perf if received_at_perf is not None else time.perf_counter(),
        }
        while True:
            try:
                self._send_queue.put_nowait(packet)
                logger.info(
                    "duplug enqueued session_id=%s queue_after=%s captured_at_ms=%s",
                    self._session_id,
                    self._send_queue.qsize(),
                    captured_at_ms,
                )
                break
            except asyncio.QueueFull:
                try:
                    _ = self._send_queue.get_nowait()
                    self._send_queue.task_done()
                    logger.warning("duplug queue full session_id=%s dropping_oldest_chunk", self._session_id)
                except asyncio.QueueEmpty:
                    break

    async def stop(self) -> None:
        self._running = False
        logger.info(
            "duplug stop session_id=%s elapsed_ms=%.2f sent_chunks=%s received_events=%s",
            self._session_id,
            (time.perf_counter() - self._started_at) * 1000,
            self._sent_chunks,
            self._received_events,
        )
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except (asyncio.CancelledError, ConnectionClosed):
                pass
            self._send_task = None
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, ConnectionClosed):
                pass
            self._receive_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _send_loop(self) -> None:
        try:
            while self._running and self._ws:
                packet = await self._send_queue.get()
                pcm16_bytes = packet["pcm16"]
                samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                payload = {
                    "type": "audio",
                    "session_id": self._session_id,
                    "audio": b64encode_bytes(samples.astype(np.float32).tobytes()),
                }
                try:
                    self._sent_chunks += 1
                    now = time.perf_counter()
                    recv_gap_ms = (now - packet["received_at_perf"]) * 1000
                    logger.info(
                        "duplug send session_id=%s chunk_idx=%s queue_after_pop=%s samples=%s elapsed_ms=%.2f browser_to_server_ms=%s server_queue_wait_ms=%.2f",
                        self._session_id,
                        self._sent_chunks,
                        self._send_queue.qsize(),
                        len(samples),
                        (now - self._started_at) * 1000,
                        packet.get("captured_at_ms"),
                        recv_gap_ms,
                    )
                    await self._ws.send(json.dumps(payload))
                finally:
                    self._send_queue.task_done()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass
        except Exception as exc:
            if self._on_error:
                await self._on_error(f"Duplug send error: {exc}")
        finally:
            self._running = False

    async def _receive_loop(self) -> None:
        try:
            while self._running and self._ws:
                raw = await self._ws.recv()
                self._received_events += 1
                logger.info(
                    "duplug recv_raw session_id=%s event_idx=%s raw_chars=%s elapsed_ms=%.2f",
                    self._session_id,
                    self._received_events,
                    len(raw),
                    (time.perf_counter() - self._started_at) * 1000,
                )
                await self._handle_message(raw)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass
        except Exception as exc:
            if self._on_error:
                await self._on_error(f"Duplug connection error: {exc}")
        finally:
            self._running = False

    async def _handle_message(self, raw: str) -> None:
        data = json.loads(raw)
        if data.get("type") != "turn_state":
            logger.info("duplug ignore session_id=%s type=%s", self._session_id, data.get("type"))
            return
        state_data = data.get("state") or {}
        if self._on_debug:
            await self._on_debug(
                {
                    "session_id": data.get("session_id"),
                    "public_state": state_data.get("state"),
                    "text": state_data.get("text", ""),
                    "asr_segment": state_data.get("asr_segment", ""),
                    "asr_buffer": state_data.get("asr_buffer", ""),
                    "debug": state_data.get("debug", {}),
                    "queue_size": self._send_queue.qsize(),
                    "chunk_ms": CHUNK_DURATION_MS,
                }
            )
        await self._process_turn_state(state_data)

    async def _process_turn_state(self, state_data: dict) -> None:
        state = (state_data.get("state") or "").strip().lower()
        logger.info(
            "duplug process_state session_id=%s state=%s text=%r asr_segment=%r asr_buffer=%r",
            self._session_id,
            state,
            state_data.get("text", ""),
            state_data.get("asr_segment", ""),
            state_data.get("asr_buffer", ""),
        )
        if state in {"", "blank"}:
            return

        if state == "idle":
            await self._on_turn_idle()
            return

        if state == "nonidle":
            transcript = self._best_interim_text(state_data)
            if not self._turn_started:
                self._turn_started = True
                self._last_interim = ""
                await self._on_user_speech_start()
            if transcript and transcript != self._last_interim:
                self._last_interim = transcript
                await self._on_user_interim(transcript)
            return

        if state == "speak":
            final_text = self._best_final_text(state_data)
            self._turn_started = False
            self._last_interim = ""
            if final_text:
                await self._on_user_turn_final(final_text)

    @staticmethod
    def _best_interim_text(state_data: dict) -> str:
        return (state_data.get("asr_buffer") or state_data.get("asr_segment") or "").strip()

    @staticmethod
    def _best_final_text(state_data: dict) -> str:
        return (state_data.get("text") or state_data.get("asr_buffer") or state_data.get("asr_segment") or "").strip()
