from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable

from ..logging_utils import get_logger
from ..services.duplug_client import DuplugClient
from ..services.llm import LLMService
from ..services.tts_qwen import QwenTTSService
from ..services.tts_qwen_pool import QwenTTSPool
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
        tts_pool: QwenTTSPool | None = None,
    ):
        self.transport = transport
        self._duplug_client_factory = duplug_client_factory
        self._llm_factory = llm_factory
        self._tts_factory = tts_factory
        self._tts_pool = tts_pool

        self.state = SessionState()
        self._llm = self._llm_factory(
            on_token=self._on_llm_token,
            on_done=self._on_llm_done,
        )
        self._tts: QwenTTSService | None = None
        self._tts_acquire_task: asyncio.Task | None = None
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
        self._active_turn_started_at = 0.0
        self._turn_last_captured_at_ms = 0.0
        self._turn_last_audio_seq = 0
        self._last_ingress_perf = 0.0

    @property
    def _stream_id(self) -> str:
        return getattr(self.transport, "stream_id", "unknown")

    async def start(self) -> None:
        await self._duplug.start()
        logger.info(
            "session transport_started stream_id=%s session_id=%s phase=%s",
            self._stream_id,
            self.transport.session_id,
            self.state.phase.value,
        )
        await self.transport.send_phase(self.state.phase)

    async def stop(self) -> None:
        logger.info(
            "session stop stream_id=%s session_id=%s elapsed_ms=%.2f phase=%s",
            self._stream_id,
            self.transport.session_id,
            (time.perf_counter() - self._started_at) * 1000,
            self.state.phase.value,
        )
        await self._cancel_active_response(send_clear=False)
        await self._duplug.stop()

    async def handle_audio(
        self,
        pcm16_bytes: bytes,
        sample_rate: int,
        captured_at_ms: float | None = None,
        received_at_perf: float | None = None,
        received_at_wall_ms: float | None = None,
        seq: int | None = None,
    ) -> None:
        now_perf = received_at_perf if received_at_perf is not None else time.perf_counter()
        now_wall_ms = received_at_wall_ms if received_at_wall_ms is not None else time.time() * 1000
        capture_to_server_ms = None
        if captured_at_ms:
            capture_to_server_ms = now_wall_ms - captured_at_ms
            self._turn_last_captured_at_ms = captured_at_ms
        uplink_gap_ms = None
        if self._last_ingress_perf:
            uplink_gap_ms = (now_perf - self._last_ingress_perf) * 1000
        self._last_ingress_perf = now_perf
        if seq is not None:
            self._turn_last_audio_seq = seq
        logger.info(
            "audio_ingress stream_id=%s session_id=%s seq=%s bytes=%s sample_rate=%s phase=%s capture_to_server_ms=%s uplink_gap_ms=%s",
            self._stream_id,
            self.transport.session_id,
            seq,
            len(pcm16_bytes),
            sample_rate,
            self.state.phase.value,
            f"{capture_to_server_ms:.2f}" if capture_to_server_ms is not None else None,
            f"{uplink_gap_ms:.2f}" if uplink_gap_ms is not None else None,
        )
        await self._duplug.send_audio(
            pcm16_bytes,
            sample_rate,
            captured_at_ms=captured_at_ms,
            received_at_perf=now_perf,
            received_at_wall_ms=now_wall_ms,
            seq=seq,
        )

    async def _on_user_speech_start(self) -> None:
        self._active_turn_started_at = time.perf_counter()
        logger.info("turn_nonidle stream_id=%s session_id=%s phase=%s", self._stream_id, self.transport.session_id, self.state.phase.value)
        await self.transport.send_turn_event("nonidle")
        async with self._lock:
            if self.state.phase == Phase.RESPONDING:
                response = self.state.active_response
                if response:
                    response.barge_in_at = time.perf_counter()
                logger.info(
                    "barge_in stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f",
                    self._stream_id,
                    self.transport.session_id,
                    response.response_id if response else None,
                    (time.perf_counter() - response.response_started_at) * 1000 if response else 0.0,
                )
                await self._cancel_active_response(send_clear=True)
                self.state.phase = Phase.LISTENING
                await self.transport.send_phase(self.state.phase)

    async def _on_user_interim(self, text: str) -> None:
        self.state.user_live_text = text
        logger.debug("user interim stream_id=%s chars=%s text=%r", self._stream_id, len(text), text)
        await self.transport.send_transcript("user", text, False)

    async def _on_user_turn_final(self, text: str) -> None:
        async with self._lock:
            self.state.user_live_text = text
            now = time.perf_counter()
            now_wall_ms = time.time() * 1000
            asr_turn_ms = (now - self._active_turn_started_at) * 1000 if self._active_turn_started_at else 0.0
            capture_to_asr_final_ms = None
            if self._turn_last_captured_at_ms:
                capture_to_asr_final_ms = now_wall_ms - self._turn_last_captured_at_ms
            logger.info(
                "turn_complete stream_id=%s session_id=%s seq=%s",
                self._stream_id,
                self.transport.session_id,
                self._turn_last_audio_seq,
            )
            await self.transport.send_turn_event("complete", text)
            logger.info(
                "asr_final stream_id=%s session_id=%s seq=%s chars=%s asr_turn_ms=%.2f capture_to_asr_final_ms=%s text=%r",
                self._stream_id,
                self.transport.session_id,
                self._turn_last_audio_seq,
                len(text),
                asr_turn_ms,
                f"{capture_to_asr_final_ms:.2f}" if capture_to_asr_final_ms is not None else None,
                text,
            )
            await self.transport.send_transcript("user", text, True)
            await self._start_response(
                text,
                asr_final_at=now,
                asr_final_wall_ms=now_wall_ms,
                capture_to_asr_final_ms=capture_to_asr_final_ms,
            )

    async def _on_turn_idle(self) -> None:
        await self.transport.send_turn_event("idle")

    async def _on_duplug_error(self, message: str) -> None:
        logger.error("duplug error stream_id=%s session_id=%s message=%s", self._stream_id, self.transport.session_id, message)
        await self.transport.send_error(message)

    async def _on_duplug_debug(self, payload: dict) -> None:
        logger.debug(
            "duplug state stream_id=%s state=%s text=%r asr_segment=%r asr_buffer=%r queue_size=%s",
            self._stream_id,
            payload.get("public_state"),
            payload.get("text", ""),
            payload.get("asr_segment", ""),
            payload.get("asr_buffer", ""),
            payload.get("queue_size"),
        )
        await self.transport.send_turn_debug(payload)

    async def _start_response(
        self,
        text: str,
        *,
        asr_final_at: float,
        asr_final_wall_ms: float,
        capture_to_asr_final_ms: float | None,
    ) -> None:
        await self._cancel_active_response(send_clear=False)
        response = ActiveResponse(response_id=uuid.uuid4().hex)
        response.response_started_at = time.perf_counter()
        response.response_started_wall_ms = time.time() * 1000
        response.asr_final_at = asr_final_at
        response.asr_final_wall_ms = asr_final_wall_ms
        self.state.active_response = response
        self.state.phase = Phase.RESPONDING
        logger.info(
            "response_start stream_id=%s session_id=%s response_id=%s user_chars=%s asr_final_to_response_start_ms=%.2f",
            self._stream_id,
            self.transport.session_id,
            response.response_id,
            len(text),
            (response.response_started_at - asr_final_at) * 1000,
        )
        await self.transport.send_phase(self.state.phase)

        response.tts_start_requested_at = time.perf_counter()
        self._tts_acquire_task = asyncio.create_task(self._acquire_tts(response.response_id))
        response.llm_dispatch_at = time.perf_counter()
        logger.info(
            "llm_dispatch stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f",
            self._stream_id,
            self.transport.session_id,
            response.response_id,
            (response.llm_dispatch_at - response.response_started_at) * 1000,
        )
        await self._llm.start_turn(text)
        response.metadata = {
            "capture_to_asr_final_ms": capture_to_asr_final_ms,
        }

    async def _acquire_tts(self, response_id: str) -> None:
        response = self.state.active_response
        if not response or response.response_id != response_id:
            return
        try:
            if self._tts_pool:
                tts, hit = await self._tts_pool.get()
                self._tts = tts
                response.metadata["tts_pool_hit"] = hit
                response.tts_pool_acquired_at = time.perf_counter()
                logger.info(
                    "tts_pool_acquired stream_id=%s session_id=%s response_id=%s hit=%s elapsed_ms=%.2f",
                    self._stream_id,
                    self.transport.session_id,
                    response_id,
                    hit,
                    (response.tts_pool_acquired_at - response.response_started_at) * 1000,
                )
                tts.bind(
                    on_audio=lambda audio_b64: self._on_tts_audio(response_id, audio_b64),
                    on_done=lambda: self._on_tts_done(response_id),
                    on_ready=lambda: self._on_tts_ready(response_id),
                )
                await self._on_tts_ready(response_id)
                return

            self._tts = self._tts_factory(
                on_audio=lambda audio_b64: self._on_tts_audio(response_id, audio_b64),
                on_done=lambda: self._on_tts_done(response_id),
                on_ready=lambda: self._on_tts_ready(response_id),
            )
            await self._tts.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            response = self.state.active_response
            if response and response.response_id == response_id:
                response.tts_failed = True
            logger.exception(
                "tts_start_failed stream_id=%s session_id=%s response_id=%s",
                self._stream_id,
                self.transport.session_id,
                response_id,
            )

    async def _cancel_active_response(self, send_clear: bool) -> None:
        response = self.state.active_response
        clear_started_at = time.perf_counter() if response else 0.0
        if response:
            response.cancelled = True
            logger.info(
                "response_cancelled stream_id=%s session_id=%s response_id=%s send_clear=%s assistant_chars=%s",
                self._stream_id,
                self.transport.session_id,
                response.response_id,
                send_clear,
                len(response.assistant_text),
            )
            await self.transport.invalidate_response(response.response_id)
            if send_clear:
                await self.transport.clear_audio(response.response_id)
                if response.barge_in_at:
                    response.metadata["interrupt_detect_to_clear_sent_ms"] = (
                        (self.transport.last_clear_sent_at - response.barge_in_at) * 1000
                    )
                else:
                    response.metadata["interrupt_detect_to_clear_sent_ms"] = (
                        (self.transport.last_clear_sent_at - clear_started_at) * 1000
                    )
        await self._llm.cancel()
        if self._tts_acquire_task:
            self._tts_acquire_task.cancel()
            try:
                await self._tts_acquire_task
            except asyncio.CancelledError:
                pass
            self._tts_acquire_task = None
        if self._tts:
            if self._tts_pool:
                await self._tts_pool.release(
                    self._tts,
                    broken=self._tts.is_broken or bool(response and not response.tts_done),
                )
            else:
                await self._tts.cancel()
            self._tts = None
        self.state.active_response = None

    async def _on_tts_ready(self, response_id: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled or response.response_id != response_id or not self._tts:
            return
        response.tts_ready = True
        response.tts_ready_at = time.perf_counter()
        logger.info(
            "tts_ready stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f pending_tokens=%s",
            self._stream_id,
            self.transport.session_id,
            response_id,
            (response.tts_ready_at - response.response_started_at) * 1000,
            len(response.pending_tts_tokens),
        )
        pending = list(response.pending_tts_tokens)
        response.pending_tts_tokens.clear()
        for token in pending:
            await self._tts.send(token)
        if response.llm_done:
            await self._tts.flush()

    async def _on_llm_token(self, token: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled:
            return
        if not response.first_token_logged:
            response.first_token_logged = True
            response.llm_first_token_at = time.perf_counter()
            logger.info(
                "llm_first_token stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f token=%r",
                self._stream_id,
                self.transport.session_id,
                response.response_id,
                (response.llm_first_token_at - response.response_started_at) * 1000,
                token,
            )
        response.assistant_text += token
        logger.debug(
            "llm token stream_id=%s session_id=%s response_id=%s token_chars=%s assistant_chars=%s",
            self._stream_id,
            self.transport.session_id,
            response.response_id,
            len(token),
            len(response.assistant_text),
        )
        await self.transport.send_transcript("assistant", response.assistant_text, False, response.response_id)
        if response.tts_ready and self._tts:
            await self._tts.send(token)
        else:
            response.pending_tts_tokens.append(token)

    async def _on_llm_done(self) -> None:
        response = self.state.active_response
        if not response or response.cancelled:
            return
        response.llm_done = True
        logger.info(
            "llm_done stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f assistant_chars=%s",
            self._stream_id,
            self.transport.session_id,
            response.response_id,
            (time.perf_counter() - response.response_started_at) * 1000,
            len(response.assistant_text),
        )
        if response.tts_ready and self._tts:
            await self._tts.flush()
        elif response.tts_failed:
            await self._finish_response_if_ready(response.response_id)

    async def _on_tts_audio(self, response_id: str, audio_b64: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled or response.response_id != response_id or not self._tts:
            return
        if not response.first_audio_logged:
            response.first_audio_logged = True
            response.tts_first_audio_at = time.perf_counter()
            logger.info(
                "tts_first_audio stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f bytes_b64=%s",
                self._stream_id,
                self.transport.session_id,
                response_id,
                (response.tts_first_audio_at - response.response_started_at) * 1000,
                len(audio_b64),
            )
        logger.debug(
            "tts audio stream_id=%s session_id=%s response_id=%s bytes_b64=%s",
            self._stream_id,
            self.transport.session_id,
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
            "tts_done stream_id=%s session_id=%s response_id=%s elapsed_ms=%.2f",
            self._stream_id,
            self.transport.session_id,
            response_id,
            (time.perf_counter() - response.response_started_at) * 1000,
        )
        await self._finish_response_if_ready(response_id)

    async def _finish_response_if_ready(self, response_id: str) -> None:
        response = self.state.active_response
        if not response or response.cancelled or response.response_id != response_id:
            return
        if not response.llm_done:
            return
        if not response.tts_done and not response.tts_failed:
            return

        await self.transport.send_transcript("assistant", response.assistant_text, True, response_id)

        metrics = {
            "response_id": response.response_id,
            "capture_to_asr_final_ms": self._round_ms(response.metadata.get("capture_to_asr_final_ms")),
            "turn_complete_to_response_start_ms": self._round_ms((response.response_started_at - response.asr_final_at) * 1000),
            "response_start_to_llm_dispatch_ms": self._round_ms((response.llm_dispatch_at - response.response_started_at) * 1000),
            "llm_dispatch_to_first_token_ms": self._round_ms(
                (response.llm_first_token_at - response.llm_dispatch_at) * 1000 if response.llm_first_token_at else None
            ),
            "tts_pool_acquire_ms": self._round_ms(
                (response.tts_pool_acquired_at - response.response_started_at) * 1000 if response.tts_pool_acquired_at else None
            ),
            "tts_pool_hit": response.metadata.get("tts_pool_hit"),
            "first_token_to_first_audio_ms": self._round_ms(
                (response.tts_first_audio_at - response.llm_first_token_at) * 1000
                if response.llm_first_token_at and response.tts_first_audio_at
                else None
            ),
            "response_start_to_first_audio_ms": self._round_ms(
                (response.tts_first_audio_at - response.response_started_at) * 1000 if response.tts_first_audio_at else None
            ),
            "interrupt_detect_to_clear_sent_ms": self._round_ms(response.metadata.get("interrupt_detect_to_clear_sent_ms")),
            "response_total_ms": self._round_ms((time.perf_counter() - response.response_started_at) * 1000),
        }
        metrics.update(self.transport.queue_metrics())
        logger.info(
            "response_complete stream_id=%s session_id=%s response_id=%s metrics=%s",
            self._stream_id,
            self.transport.session_id,
            response.response_id,
            metrics,
        )
        await self.transport.send_metrics(metrics)
        if self._tts:
            if self._tts_pool:
                await self._tts_pool.release(self._tts, broken=self._tts.is_broken)
            else:
                await self._tts.cancel()
            self._tts = None
        self._tts_acquire_task = None
        self.state.phase = Phase.LISTENING
        self.state.active_response = None
        await self.transport.send_phase(self.state.phase)

    @staticmethod
    def _round_ms(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 2)
