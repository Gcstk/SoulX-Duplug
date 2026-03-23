from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Optional

from openai import AsyncOpenAI

from ..logging_utils import get_logger


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful Chinese voice assistant. "
    "Keep replies concise, natural, and easy to speak aloud."
)
logger = get_logger("llm")


class LLMService:
    def __init__(
        self,
        on_token: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
    ):
        self._on_token = on_token
        self._on_done = on_done
        self._client = client or AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", None),
        )
        self._model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._system_prompt = system_prompt or os.getenv("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self._history: list[dict[str, str]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at = 0.0

    @property
    def history(self) -> list[dict[str, str]]:
        return self._history.copy()

    def clear_history(self) -> None:
        self._history = []

    async def start_turn(self, user_text: str) -> None:
        if self._running:
            await self.cancel()
        self._history.append({"role": "user", "content": user_text})
        self._running = True
        self._started_at = time.perf_counter()
        logger.info("llm start model=%s user_chars=%s history_len=%s", self._model, len(user_text), len(self._history))
        self._task = asyncio.create_task(self._run())

    async def cancel(self) -> None:
        self._running = False
        logger.info("llm cancel model=%s elapsed_ms=%.2f", self._model, (time.perf_counter() - self._started_at) * 1000 if self._started_at else 0.0)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        assistant_text = ""
        try:
            first_token_logged = False
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": self._system_prompt}, *self._history],
                stream=True,
                temperature=0.7,
                max_tokens=400,
            )
            async for chunk in stream:
                if not self._running:
                    break
                delta = chunk.choices[0].delta if chunk.choices else None
                token = delta.content if delta and delta.content else ""
                if token:
                    if not first_token_logged:
                        first_token_logged = True
                        logger.info("llm first upstream token model=%s elapsed_ms=%.2f token=%r", self._model, (time.perf_counter() - self._started_at) * 1000, token)
                    assistant_text += token
                    await self._on_token(token)
            if self._running and assistant_text:
                self._history.append({"role": "assistant", "content": assistant_text})
            logger.info("llm complete model=%s elapsed_ms=%.2f assistant_chars=%s", self._model, (time.perf_counter() - self._started_at) * 1000, len(assistant_text))
            await self._on_done()
        except asyncio.CancelledError:
            logger.info("llm task cancelled model=%s elapsed_ms=%.2f partial_chars=%s", self._model, (time.perf_counter() - self._started_at) * 1000, len(assistant_text))
            raise
        except Exception:
            logger.exception("llm error model=%s elapsed_ms=%.2f", self._model, (time.perf_counter() - self._started_at) * 1000)
            await self._on_done()
        finally:
            self._running = False
            self._task = None
