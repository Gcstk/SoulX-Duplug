from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from .audio import pack_binary_audio_message, pcm16_frame_duration_ms
from .logging_utils import get_logger
from .types import Phase

logger = get_logger("transport")

P0_CONTROL = 0
P1_TEXT = 1
P2_MEDIA = 2


@dataclass
class _ControlMessage:
    payload: dict[str, Any]
    priority: int
    enqueued_at: float
    response_id: str | None = None
    future: asyncio.Future = field(default_factory=asyncio.Future)


@dataclass
class _AudioMessage:
    payload: bytes
    metadata: dict[str, Any]
    enqueued_at: float
    response_id: str | None = None


class BrowserTransport:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_id = f"browser-{uuid.uuid4().hex[:12]}"
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self._started_at = time.perf_counter()
        self._debug_sent_at = 0.0
        self._last_debug_signature: tuple[Any, ...] | None = None
        self._last_turn_event: tuple[str, str] | None = None
        self._control_queues: dict[int, deque[_ControlMessage]] = {
            P0_CONTROL: deque(),
            P1_TEXT: deque(),
        }
        self._audio_queue: deque[_AudioMessage] = deque()
        self._control_condition = asyncio.Condition()
        self._audio_condition = asyncio.Condition()
        self._control_sender_task: asyncio.Task | None = None
        self._audio_sender_task: asyncio.Task | None = None
        self._audio_ws: WebSocket | None = None
        self._running = False
        self._queue_peaks = {P0_CONTROL: 0, P1_TEXT: 0, P2_MEDIA: 0}
        self._stale_media_dropped = 0
        self._last_clear_sent_at = 0.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._control_sender_task = asyncio.create_task(self._control_send_loop())
        self._audio_sender_task = asyncio.create_task(self._audio_send_loop())

    async def stop(self) -> None:
        self._running = False
        async with self._control_condition:
            self._control_condition.notify_all()
        async with self._audio_condition:
            self._audio_condition.notify_all()
        for task in (self._control_sender_task, self._audio_sender_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._control_sender_task = None
        self._audio_sender_task = None
        self._clear_pending(RuntimeError("transport stopped"))
        self._audio_queue.clear()
        self._audio_ws = None

    async def bind_audio_socket(self, websocket: WebSocket) -> None:
        self._audio_ws = websocket
        async with self._audio_condition:
            self._audio_condition.notify_all()

    def unbind_audio_socket(self, websocket: WebSocket) -> None:
        if self._audio_ws is websocket:
            self._audio_ws = None

    async def send_ready(self) -> None:
        await self._direct_send(
            {
                "type": "ready",
                "stream_id": self.stream_id,
                "session_id": self.session_id,
            }
        )

    async def send_phase(self, phase: Phase) -> None:
        await self._enqueue_control({"type": "session_control", "event": "phase", "phase": phase.value}, P0_CONTROL)

    async def send_turn_event(self, kind: str, text: str = "") -> None:
        signature = (kind, text if kind == "complete" else "")
        if self._last_turn_event == signature:
            return
        self._last_turn_event = signature
        await self._enqueue_control({"type": "turn_event", "kind": kind, "text": text}, P0_CONTROL)

    async def send_interrupt(self, kind: str, response_id: str | None = None) -> None:
        await self._enqueue_control({"type": "interrupt", "kind": kind, "response_id": response_id}, P0_CONTROL, response_id=response_id)

    async def send_asr_partial(self, text: str) -> None:
        await self._enqueue_control({"type": "asr_partial", "text": text}, P1_TEXT)

    async def send_asr_final(self, text: str) -> None:
        await self._enqueue_control({"type": "asr_final", "text": text}, P1_TEXT)

    async def send_llm_token(self, token: str, response_id: str) -> None:
        await self._enqueue_control({"type": "llm_token", "text": token, "response_id": response_id}, P1_TEXT, response_id=response_id)

    async def send_llm_final(self, text: str, response_id: str) -> None:
        await self._enqueue_control({"type": "llm_final", "text": text, "response_id": response_id}, P1_TEXT, response_id=response_id)

    async def send_audio_chunk(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        response_id: str,
        *,
        chunk_seq: int,
        generated_at_ms: float,
    ) -> None:
        metadata = {
            "type": "tts_chunk",
            "response_id": response_id,
            "chunk_seq": chunk_seq,
            "generated_at_ms": generated_at_ms,
            "sample_rate": sample_rate,
            "duration_ms": round(pcm16_frame_duration_ms(pcm_bytes, sample_rate), 2),
        }
        msg = _AudioMessage(
            payload=pack_binary_audio_message(metadata, pcm_bytes),
            metadata=metadata,
            enqueued_at=time.perf_counter(),
            response_id=response_id,
        )
        async with self._audio_condition:
            self._audio_queue.append(msg)
            self._queue_peaks[P2_MEDIA] = max(self._queue_peaks[P2_MEDIA], len(self._audio_queue))
            logger.info(
                "tts_chunk_enqueued stream_id=%s session_id=%s response_id=%s priority=%s chunk_ms=%.2f queue_depth=%s",
                self.stream_id,
                self.session_id,
                response_id,
                P2_MEDIA,
                metadata["duration_ms"],
                len(self._audio_queue),
            )
            self._audio_condition.notify()

    async def clear_audio(self, response_id: str | None = None) -> None:
        logger.info(
            "interrupt_clear_enqueued stream_id=%s session_id=%s response_id=%s",
            self.stream_id,
            self.session_id,
            response_id,
        )
        await self.invalidate_response(response_id)
        await self.send_interrupt("clear_audio", response_id)

    async def send_error(self, message: str) -> None:
        await self._enqueue_control({"type": "session_control", "event": "error", "message": message}, P0_CONTROL)

    async def send_metrics(self, payload: dict[str, Any]) -> None:
        enriched = {
            **payload,
            "control_queue_peak": max(self._queue_peaks[P0_CONTROL], self._queue_peaks[P1_TEXT]),
            "audio_queue_peak": self._queue_peaks[P2_MEDIA],
            "tts_chunk_drop_count": self._stale_media_dropped,
        }
        await self._enqueue_control({"type": "session_control", "event": "metrics", "payload": enriched}, P1_TEXT)

    async def send_turn_debug(self, payload: dict[str, Any]) -> None:
        now = time.perf_counter()
        signature = (
            payload.get("public_state"),
            payload.get("text", ""),
            payload.get("asr_segment", ""),
            payload.get("asr_buffer", ""),
        )
        if self._last_debug_signature == signature and (now - self._debug_sent_at) < 0.25:
            return
        self._last_debug_signature = signature
        self._debug_sent_at = now
        await self._enqueue_control(
            {
                "type": "session_control",
                "event": "turn_debug",
                "payload": {
                    "public_state": payload.get("public_state"),
                    "text": payload.get("text", ""),
                    "asr_segment": payload.get("asr_segment", ""),
                    "asr_buffer": payload.get("asr_buffer", ""),
                    "queue_size": payload.get("queue_size"),
                    "chunk_ms": payload.get("chunk_ms"),
                    "received_event_idx": payload.get("received_event_idx"),
                    "server_elapsed_ms": round((time.perf_counter() - self._started_at) * 1000, 2),
                },
            },
            P0_CONTROL,
        )

    async def invalidate_response(self, response_id: str | None) -> None:
        if not response_id:
            return
        async with self._control_condition:
            for priority in (P1_TEXT,):
                queue = self._control_queues[priority]
                kept: deque[_ControlMessage] = deque()
                while queue:
                    item = queue.popleft()
                    if item.response_id == response_id:
                        if not item.future.done():
                            item.future.set_result(False)
                        continue
                    kept.append(item)
                self._control_queues[priority] = kept
        async with self._audio_condition:
            kept_audio: deque[_AudioMessage] = deque()
            while self._audio_queue:
                item = self._audio_queue.popleft()
                if item.response_id == response_id:
                    self._stale_media_dropped += 1
                    logger.info(
                        "interrupt_audio_dropped stream_id=%s session_id=%s response_id=%s chunk_seq=%s",
                        self.stream_id,
                        self.session_id,
                        response_id,
                        item.metadata.get("chunk_seq"),
                    )
                    continue
                kept_audio.append(item)
            self._audio_queue = kept_audio

    def queue_metrics(self) -> dict[str, Any]:
        return {
            "control_queue_peak": max(self._queue_peaks[P0_CONTROL], self._queue_peaks[P1_TEXT]),
            "audio_queue_peak": self._queue_peaks[P2_MEDIA],
            "tts_chunk_drop_count": self._stale_media_dropped,
        }

    @property
    def last_clear_sent_at(self) -> float:
        return self._last_clear_sent_at

    async def _enqueue_control(
        self,
        payload: dict[str, Any],
        priority: int,
        *,
        response_id: str | None = None,
    ) -> bool:
        item = _ControlMessage(payload=payload, priority=priority, enqueued_at=time.perf_counter(), response_id=response_id)
        async with self._control_condition:
            queue = self._control_queues[priority]
            queue.append(item)
            self._queue_peaks[priority] = max(self._queue_peaks[priority], len(queue))
            logger.info(
                "control_enqueue stream_id=%s session_id=%s priority=%s message_type=%s queue_depth=%s",
                self.stream_id,
                self.session_id,
                priority,
                payload.get("type"),
                len(queue),
            )
            self._control_condition.notify()
        return await item.future

    async def _control_send_loop(self) -> None:
        try:
            while self._running:
                async with self._control_condition:
                    while self._running and not any(self._control_queues[p] for p in (P0_CONTROL, P1_TEXT)):
                        await self._control_condition.wait()
                    if not self._running:
                        break
                    item = self._pop_control()
                await self._send_control(item)
        except asyncio.CancelledError:
            raise

    async def _audio_send_loop(self) -> None:
        try:
            while self._running:
                async with self._audio_condition:
                    while self._running and (self._audio_ws is None or not self._audio_queue):
                        await self._audio_condition.wait()
                    if not self._running:
                        break
                    item = self._audio_queue.popleft()
                    audio_ws = self._audio_ws
                if audio_ws is None:
                    continue
                # Audio is isolated on its own websocket so queued media cannot
                # head-of-line block interrupts or state transitions.
                age_ms = (time.perf_counter() - item.enqueued_at) * 1000
                item.metadata["chunk_age_ms"] = round(age_ms, 2)
                await audio_ws.send_bytes(item.payload)
                logger.debug(
                    "audio_send stream_id=%s session_id=%s response_id=%s chunk_seq=%s chunk_age_ms=%.2f",
                    self.stream_id,
                    self.session_id,
                    item.response_id,
                    item.metadata.get("chunk_seq"),
                    age_ms,
                )
        except asyncio.CancelledError:
            raise

    def _pop_control(self) -> _ControlMessage:
        for priority in (P0_CONTROL, P1_TEXT):
            queue = self._control_queues[priority]
            if queue:
                return queue.popleft()
        raise RuntimeError("no control items")

    async def _send_control(self, item: _ControlMessage) -> None:
        queue_wait_ms = (time.perf_counter() - item.enqueued_at) * 1000
        await self.websocket.send_text(json.dumps(item.payload))
        logger.info(
            "control_send stream_id=%s session_id=%s priority=%s message_type=%s queue_wait_ms=%.2f",
            self.stream_id,
            self.session_id,
            item.priority,
            item.payload.get("type"),
            queue_wait_ms,
        )
        if item.payload.get("type") == "interrupt" and item.payload.get("kind") == "clear_audio":
            self._last_clear_sent_at = time.perf_counter()
            logger.info(
                "interrupt_clear_sent stream_id=%s session_id=%s response_id=%s",
                self.stream_id,
                self.session_id,
                item.payload.get("response_id"),
            )
        if not item.future.done():
            item.future.set_result(True)

    async def _direct_send(self, payload: dict[str, Any]) -> None:
        await self.websocket.send_text(json.dumps(payload))

    def _clear_pending(self, exc: Exception) -> None:
        for queue in self._control_queues.values():
            while queue:
                item = queue.popleft()
                if not item.future.done():
                    item.future.set_exception(exc)
