from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Callable

from .tts_qwen import QwenTTSService
from ..logging_utils import get_logger

logger = get_logger("tts_pool")


@dataclass
class _Entry:
    tts: QwenTTSService
    idle_since: float


class QwenTTSPool:
    def __init__(
        self,
        pool_size: int | None = None,
        ttl: float | None = None,
        service_factory: Callable[[], QwenTTSService] = QwenTTSService,
    ):
        self._pool_size = max(1, pool_size if pool_size is not None else int(os.getenv("TTS_POOL_SIZE", "4")))
        self._ttl = max(0.1, ttl if ttl is not None else float(os.getenv("TTS_POOL_TTL", "8.0")))
        self._service_factory = service_factory
        self._ready: list[_Entry] = []
        self._running = False
        self._fill_task: asyncio.Task | None = None
        self._fill_event = asyncio.Event()
        self.pool_hit = 0
        self.pool_miss = 0
        self.warmup_count = 0

    @property
    def available(self) -> int:
        return len(self._ready)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        logger.info("tts_pool_started pool_size=%s ttl=%s", self._pool_size, self._ttl)
        self._fill_task = asyncio.create_task(self._fill_loop())
        self._trigger_fill()

    async def stop(self) -> None:
        self._running = False
        self._fill_event.set()
        if self._fill_task:
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
            self._fill_task = None
        for entry in self._ready:
            await entry.tts.cancel()
        self._ready.clear()

    async def get(self) -> tuple[QwenTTSService, bool]:
        while self._ready:
            entry = self._ready.pop(0)
            age = time.monotonic() - entry.idle_since
            if age < self._ttl and entry.tts.is_running and not entry.tts.is_broken:
                self.pool_hit += 1
                logger.info("tts_pool_hit available=%s idle_ms=%s", len(self._ready), int(age * 1000))
                self._trigger_fill()
                return entry.tts, True
            logger.info("tts_pool_evict_stale idle_ms=%s", int(age * 1000))
            await entry.tts.cancel()

        self.pool_miss += 1
        logger.info("tts_pool_miss available=0")
        tts = self._service_factory()
        await tts.start()
        self._trigger_fill()
        return tts, False

    async def release(self, tts: QwenTTSService, *, broken: bool = False) -> None:
        if broken or tts.is_broken or not tts.is_running:
            await tts.cancel()
            self._trigger_fill()
            return
        tts.reset_for_next_turn()
        if len(self._ready) >= self._pool_size:
            await tts.cancel()
            return
        self._ready.append(_Entry(tts=tts, idle_since=time.monotonic()))
        self._trigger_fill()

    def _trigger_fill(self) -> None:
        self._fill_event.set()

    async def _fill_loop(self) -> None:
        try:
            while self._running:
                await self._evict_stale()
                while self._running and len(self._ready) < self._pool_size:
                    tts = self._service_factory()
                    try:
                        await tts.start()
                        tts.reset_for_next_turn()
                        self._ready.append(_Entry(tts=tts, idle_since=time.monotonic()))
                        self.warmup_count += 1
                        logger.info("tts_pool_refill available=%s target=%s", len(self._ready), self._pool_size)
                    except Exception:
                        logger.exception("tts_pool_refill_failed")
                        await asyncio.sleep(1.0)
                        break
                self._fill_event.clear()
                try:
                    await asyncio.wait_for(self._fill_event.wait(), timeout=self._ttl / 2)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    async def _evict_stale(self) -> None:
        fresh: list[_Entry] = []
        now = time.monotonic()
        for entry in self._ready:
            age = now - entry.idle_since
            if age < self._ttl and entry.tts.is_running and not entry.tts.is_broken:
                fresh.append(entry)
                continue
            logger.info("tts_pool_evict_stale idle_ms=%s", int(age * 1000))
            await entry.tts.cancel()
        self._ready = fresh
