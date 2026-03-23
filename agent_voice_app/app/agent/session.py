from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable

from ..logging_utils import get_logger
from ..services.duplug_client import DuplugClient
from ..services.llm import LLMService
from ..services.tts_qwen import QwenTTSService
from ..transport import BrowserTransport
from ..types import ActiveResponse, Phase, SessionState

logger = get_logger("session")


class AgentSession:
    def __init__(
        self,
        transport: BrowserTransport,
        duplug_client_factory: Callable[..., DuplugClient] = DuplugClient,
        llm_factory: Callable[..., LLMService] = LLMService,
        tts_factory: Callable[..., QwenTTSService] = QwenTTSService,
    ):
        self.transport = transport
        self._duplug_client_factory = duplug_client_factory
        self._llm_factory = llm_factory
        self._tts_factory = tts_factory

        self.state = SessionState()
        self._llm = self._llm_factory(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )
        self._tts: QwenTTSService | None = None
        self._duplug = self._duplug_client_factory(
            on_user_speech_start=self._on_user_speech_start,
            on_user_interim=self._on_user_interim,
            on_user_turn_final=self._on_user_turn_final,
            on_turn_idle=self._on_turn_idle,
            on_error=self._on_duplug_error,
            on_debug=self._on_duplug_debug,
        )
        self._lock = asyncio.Lock()
        self._started_at = time.perf_counter()

    @property
    def _stream_id(self) -> str:
        return getattr(self.transport, "stream_id", "unknown")

    async def start(self) -> None:
        await self._duplug.start()
        logger.info("session transport_started stream_id=%s phase=%s", self._stream_id, self.state.phase.value)
        await self.transport.send_phase(self.state.phase)

    async def stop(self) -> None:
        logger.info(
            "session stop stream_id=%s elapsed_ms=%.2f phase=%s",
            self._stream_id,
            (time.perf_counter() - self._started_at) * 1000,
            self.state.phase.value,
        )
        await self._cancel_active_response(send_clear=False)
        await self._duplug.stop()

    async def handle_audio(self, pcm16_bytes: bytes, sample_rate: int) -> None:
        logger.info(
            "session handle_audio stream_id=%s bytes=%s sample_rate=%s phase=%s",
            self._stream_id,
            len(pcm16_bytes),
            sample_rate,
            self.state.phase.value,
        )
        await self._duplug.send_audio(pcm16_bytes, sample_rate)

    async def _on_user_speech_start(self) -> None:
        logger.info("user speech start stream_id=%s phase=%s", self._stream_id, self.state.phase.value)
        async with self._lock:
            if self.state.phase == Phase.RESPONDING:
                logger.info(
                    "barge-in detected stream_id=%s response_id=%s",
                    self._stream_id,
                    self.state.active_response.response_id if self.state.active_response else None,
                )
                await self._cancel_active_response(send_clear=True)
                self.state.phase = Phase.LISTENING
                await self.transport.send_phase(self.state.phase)

    async def _on_user_interim(self, text: str) -> None:
        self.state.user_live_text = text
        logger.info("user interim stream_id=%s chars=%s text=%r", self._stream_id, len(text), text)
        await self.transport.send_transcript("user", text, False)

    async def _on_user_turn_final(self, text: str) -> None:
        async with self._lock:
            self.state.user_live_text = text
            logger.info("user final stream_id=%s chars=%s text=%r", self._stream_id, len(text), text)
            await self.transport.send_transcript("user", text, True)
            await self._start_response(text)

    async def _on_turn_idle(self) -> None:
        return None

    async def _on_duplug_error(self, message: str) -> None:
        logger.error("duplug error stream_id=%s message=%s", self._stream_id, message)
        await self.transport.send_error(message)

    async def _on_duplug_debug(self, payload: dict) -> None:
        logger.info(
            "duplug state stream_id=%s state=%s text=%r asr_segment=%r asr_buffer=%r queue_size=%s",
            self._stream_id,
            payload.get("public_state"),
            payload.get("text", ""),
            payload.get("asr_segment", ""),
            payload.get("asr_buffer", ""),
            payload.get("queue_size"),
        )
        await self.transport.send_turn_debug(payload)

    async def _start_response(self, text: str) -> None:
        await self._cancel_active_response(send_clear=False)
        response = ActiveResponse(response_id=uuid.uuid4().hex)
        self.state.active_response = response
        self.state.phase = Phase.RESPONDING
        self._tts = self._tts_factory(
            on_audio=lambda audio_b64: self._on_tts_audio(response.response_id, audio_b64),
            on_done=lambda: self._on_tts_done(response.response_id),
        )
        response.started_at = time.perf_counter()
        logger.info(
            "response start stream_id=%s response_id=%s user_chars=%s",
            self._stream_id,
            response.response_id,
            len(text),
        )
        await self._tts.start()
        await self.transport.send_phase(self.state.phase)
        await self._llm.start_turn(text)

    async def _cancel_active_response(self, send_clear: bool) -> None:
        response = self.state.active_response
        if response:
            response.cancelled = True
            logger.info(
                "response cancel stream_id=%s response_id=%s send_clear=%s assistant_chars=%s",
                self._stream_id,
                response.response_id,
                send_clear,
                len(response.assistant_text),
            )
            if send_clear:
                await self.transport.clear_audio(response.response_id)
        await self._llm.cancel()
        if self._tts:
            await self._tts.cancel()
            self._tts = None
        self.state.active_response = None

    async def _on_llm_token(self, token: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled:
            return
        if not getattr(response, "first_token_logged", False):
            response.first_token_logged = True
            logger.info(
                "llm first_token stream_id=%s response_id=%s elapsed_ms=%.2f token=%r",
                self._stream_id,
                response.response_id,
                (time.perf_counter() - response.started_at) * 1000,
                token,
            )
        response.assistant_text += token
        logger.info(
            "llm token stream_id=%s response_id=%s token_chars=%s assistant_chars=%s",
            self._stream_id,
            response.response_id,
            len(token),
            len(response.assistant_text),
        )
        await self.transport.send_transcript(
            "assistant",
            response.assistant_text,
            False,
            response.response_id,
        )
        if self._tts:
            await self._tts.send(token)

    async def _on_llm_done(self) -> None:
        response = self.state.active_response
        if not response or response.cancelled:
            return
        response.llm_done = True
        logger.info(
            "llm done stream_id=%s response_id=%s elapsed_ms=%.2f assistant_chars=%s",
            self._stream_id,
            response.response_id,
            (time.perf_counter() - response.started_at) * 1000,
            len(response.assistant_text),
        )
        if self._tts:
            await self._tts.flush()
        else:
            await self._finish_response_if_ready(response.response_id)

    async def _on_tts_audio(self, response_id: str, audio_b64: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled or response.response_id != response_id:
            return
        if self._tts:
            if not getattr(response, "first_audio_logged", False):
                response.first_audio_logged = True
                logger.info(
                    "tts first_audio stream_id=%s response_id=%s elapsed_ms=%.2f bytes_b64=%s",
                    self._stream_id,
                    response_id,
                    (time.perf_counter() - response.started_at) * 1000,
                    len(audio_b64),
                )
            logger.info(
                "tts audio stream_id=%s response_id=%s bytes_b64=%s",
                self._stream_id,
                response_id,
                len(audio_b64),
            )
            await self.transport.send_audio(audio_b64, self._tts.sample_rate, response_id)

    async def _on_tts_done(self, response_id: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled or response.response_id != response_id:
            return
        response.tts_done = True
        logger.info(
            "tts done stream_id=%s response_id=%s elapsed_ms=%.2f",
            self._stream_id,
            response_id,
            (time.perf_counter() - response.started_at) * 1000,
        )
        await self._finish_response_if_ready(response_id)

    async def _finish_response_if_ready(self, response_id: str) -> None:
        response = self.state.active_response
        if not response or response.response_id != response_id:
            return
        if not response.llm_done or not response.tts_done:
            return
        logger.info(
            "response complete stream_id=%s response_id=%s total_ms=%.2f assistant_chars=%s",
            self._stream_id,
            response.response_id,
            (time.perf_counter() - response.started_at) * 1000,
            len(response.assistant_text),
        )
        await self.transport.send_transcript(
            "assistant",
            response.assistant_text,
            True,
            response.response_id,
        )
        if self._tts:
            await self._tts.cancel()
            self._tts = None
        self.state.active_response = None
        self.state.phase = Phase.LISTENING
        await self.transport.send_phase(self.state.phase)
