from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from .logging_utils import get_logger
from .types import Phase

logger = get_logger("transport")


P0_CONTROL = 0
P1_TRANSCRIPT = 1
P2_MEDIA = 2


@dataclass
class _QueuedMessage:
    payload: dict[str, Any]
    priority: int
    enqueued_at: float
    response_id: str | None = None
    future: asyncio.Future = field(default_factory=asyncio.Future)


class BrowserTransport:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.stream_id = f"browser-{uuid.uuid4().hex[:12]}"
        self.session_id = f"session-{uuid.uuid4().hex[:12]}"
        self._started_at = time.perf_counter()
        self._debug_sent_at = 0.0
        self._last_debug_signature: tuple[Any, ...] | None = None
        self._queues: dict[int, deque[_QueuedMessage]] = {
            P0_CONTROL: deque(),
            P1_TRANSCRIPT: deque(),
            P2_MEDIA: deque(),
        }
        self._condition = asyncio.Condition()
        self._sender_task: asyncio.Task | None = None
        self._running = False
        self._queue_peaks = {P0_CONTROL: 0, P1_TRANSCRIPT: 0, P2_MEDIA: 0}
        self._stale_media_dropped = 0
        self._last_clear_sent_at = 0.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._sender_task = asyncio.create_task(self._send_loop())

    async def stop(self) -> None:
        self._running = False
        async with self._condition:
            self._condition.notify_all()
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        self._clear_pending(RuntimeError("transport stopped"))

    async def send_ready(self) -> None:
        await self._direct_send(
            {
                "type": "ready",
                "stream_id": self.stream_id,
                "session_id": self.session_id,
            }
        )

    async def send_phase(self, phase: Phase) -> None:
        await self._enqueue_and_wait({"type": "phase", "phase": phase.value}, P0_CONTROL)

    async def send_turn_event(self, kind: str, text: str = "") -> None:
        await self._enqueue_and_wait({"type": "turn_event", "kind": kind, "text": text}, P0_CONTROL)

    async def send_transcript(
        self,
        speaker: str,
        text: str,
        final: bool,
        response_id: str | None = None,
    ) -> None:
        await self._enqueue_and_wait(
            {
                "type": "transcript",
                "speaker": speaker,
                "text": text,
                "final": final,
                "response_id": response_id,
            },
            P1_TRANSCRIPT,
            response_id=response_id,
        )

    async def send_audio(self, audio_b64: str, sample_rate: int, response_id: str) -> None:
        await self._enqueue_and_wait(
            {
                "type": "audio",
                "audio_b64": audio_b64,
                "sample_rate": sample_rate,
                "response_id": response_id,
            },
            P2_MEDIA,
            response_id=response_id,
        )

    async def clear_audio(self, response_id: str | None = None) -> None:
        logger.info(
            "interrupt_clear_enqueued stream_id=%s session_id=%s response_id=%s",
            self.stream_id,
            self.session_id,
            response_id,
        )
        await self._enqueue_and_wait(
            {"type": "clear_audio", "response_id": response_id},
            P0_CONTROL,
            response_id=response_id,
        )

    async def send_error(self, message: str) -> None:
        await self._enqueue_and_wait({"type": "error", "message": message}, P0_CONTROL)

    async def send_metrics(self, payload: dict[str, Any]) -> None:
        enriched = {
            **payload,
            "control_queue_peak_p0": self._queue_peaks[P0_CONTROL],
            "control_queue_peak_p1": self._queue_peaks[P1_TRANSCRIPT],
            "control_queue_peak_p2": self._queue_peaks[P2_MEDIA],
            "stale_media_dropped_count": self._stale_media_dropped,
            "interrupt_detect_to_clear_sent_ms": payload.get("interrupt_detect_to_clear_sent_ms"),
        }
        await self._enqueue_and_wait({"type": "metrics", "payload": enriched}, P1_TRANSCRIPT)

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
        summary = {
            "public_state": payload.get("public_state"),
            "text": payload.get("text", ""),
            "asr_segment": payload.get("asr_segment", ""),
            "asr_buffer": payload.get("asr_buffer", ""),
            "queue_size": payload.get("queue_size"),
            "chunk_ms": payload.get("chunk_ms"),
            "received_event_idx": payload.get("received_event_idx"),
            "server_elapsed_ms": round((time.perf_counter() - self._started_at) * 1000, 2),
        }
        await self._enqueue_and_wait({"type": "turn_debug", "payload": summary}, P0_CONTROL)

    async def invalidate_response(self, response_id: str | None) -> None:
        if not response_id:
            return
        async with self._condition:
            for priority in (P1_TRANSCRIPT, P2_MEDIA):
                queue = self._queues[priority]
                kept: deque[_QueuedMessage] = deque()
                while queue:
                    item = queue.popleft()
                    if item.response_id == response_id:
                        self._stale_media_dropped += 1
                        if not item.future.done():
                            item.future.set_result(False)
                        logger.info(
                            "control_drop_stale_media stream_id=%s session_id=%s response_id=%s priority=%s type=%s",
                            self.stream_id,
                            self.session_id,
                            response_id,
                            priority,
                            item.payload.get("type"),
                        )
                        continue
                    kept.append(item)
                self._queues[priority] = kept

    def queue_metrics(self) -> dict[str, Any]:
        return {
            "control_queue_peak_p0": self._queue_peaks[P0_CONTROL],
            "control_queue_peak_p1": self._queue_peaks[P1_TRANSCRIPT],
            "control_queue_peak_p2": self._queue_peaks[P2_MEDIA],
            "stale_media_dropped_count": self._stale_media_dropped,
        }

    @property
    def last_clear_sent_at(self) -> float:
        return self._last_clear_sent_at

    async def _enqueue_and_wait(
        self,
        payload: dict[str, Any],
        priority: int,
        *,
        response_id: str | None = None,
    ) -> bool:
        item = _QueuedMessage(
            payload=payload,
            priority=priority,
            enqueued_at=time.perf_counter(),
            response_id=response_id or payload.get("response_id"),
        )
        async with self._condition:
            queue = self._queues[priority]
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
            if priority == P0_CONTROL and self._queues[P2_MEDIA]:
                logger.info(
                    "control_preempt_media stream_id=%s session_id=%s message_type=%s pending_media=%s",
                    self.stream_id,
                    self.session_id,
                    payload.get("type"),
                    len(self._queues[P2_MEDIA]),
                )
            self._condition.notify()
        return await item.future

    async def _send_loop(self) -> None:
        try:
            while self._running:
                async with self._condition:
                    while self._running and not any(self._queues[p] for p in (P0_CONTROL, P1_TRANSCRIPT, P2_MEDIA)):
                        await self._condition.wait()
                    if not self._running:
                        break
                    item = self._pop_next()
                await self._send_item(item)
        except asyncio.CancelledError:
            raise

    def _pop_next(self) -> _QueuedMessage:
        for priority in (P0_CONTROL, P1_TRANSCRIPT, P2_MEDIA):
            queue = self._queues[priority]
            if queue:
                return queue.popleft()
        raise RuntimeError("no queued items")

    async def _send_item(self, item: _QueuedMessage) -> None:
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
        if item.payload.get("type") == "clear_audio":
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
        for queue in self._queues.values():
            while queue:
                item = queue.popleft()
                if not item.future.done():
                    item.future.set_exception(exc)
