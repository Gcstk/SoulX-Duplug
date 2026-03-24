from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.agent.session import AgentSession
from app.audio import unpack_binary_audio_message
from app.logging_utils import get_logger
from app.services.tts_qwen_pool import QwenTTSPool
from app.transport import BrowserTransport

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
logger = get_logger("server")

app = FastAPI(title="SoulX Agent Voice App", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    def add(self, session: AgentSession) -> None:
        self._sessions[session.transport.session_id] = session

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def pop(self, session_id: str) -> AgentSession | None:
        return self._sessions.pop(session_id, None)


registry = SessionRegistry()
tts_pool = QwenTTSPool()


@app.on_event("startup")
async def startup_event() -> None:
    if os.getenv("TTS_POOL_PREWARM_ON_START", "true").lower() == "true":
        await tts_pool.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await tts_pool.stop()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/web", status_code=307)


@app.get("/web")
async def web() -> FileResponse:
    return FileResponse(STATIC_DIR / "browser_agent.html")


@app.websocket("/ws/control")
async def control_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    transport = BrowserTransport(websocket)
    await transport.start()
    session = AgentSession(transport, tts_pool=tts_pool)
    registry.add(session)
    await transport.send_ready()
    ws_opened_at = time.perf_counter()
    logger.info(
        "browser_session_started stream_id=%s session_id=%s client=%s",
        transport.stream_id,
        transport.session_id,
        getattr(websocket, "client", None),
    )

    started = False

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            message_type = data.get("type")
            logger.debug(
                "browser control message stream_id=%s session_id=%s type=%s elapsed_ms=%.2f",
                transport.stream_id,
                transport.session_id,
                message_type,
                (time.perf_counter() - ws_opened_at) * 1000,
            )

            if message_type == "start":
                try:
                    await session.start()
                    started = True
                    logger.info(
                        "browser control started stream_id=%s session_id=%s sample_rate=%s",
                        transport.stream_id,
                        transport.session_id,
                        data.get("sample_rate"),
                    )
                except Exception as exc:
                    logger.exception(
                        "session start failed stream_id=%s session_id=%s error=%s",
                        transport.stream_id,
                        transport.session_id,
                        exc,
                    )
                    await transport.send_error(f"session start failed: {exc}")
                continue

            if message_type == "stop":
                logger.info(
                    "browser requested stop stream_id=%s session_id=%s",
                    transport.stream_id,
                    transport.session_id,
                )
                break

            await transport.send_error(f"unsupported message type: {message_type}")

    except WebSocketDisconnect:
        logger.info(
            "browser control disconnected stream_id=%s session_id=%s",
            transport.stream_id,
            transport.session_id,
        )
    finally:
        registry.pop(transport.session_id)
        logger.info(
            "browser session stopping stream_id=%s session_id=%s total_elapsed_ms=%.2f",
            transport.stream_id,
            transport.session_id,
            (time.perf_counter() - ws_opened_at) * 1000,
        )
        await session.stop()
        await transport.stop()


@app.websocket("/ws/uplink")
async def uplink_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    opened_at = time.perf_counter()
    session: AgentSession | None = None
    stream_id = "unknown"
    session_id = "unknown"
    first_audio_logged = False
    last_audio_received_at = 0.0

    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                if data.get("type") != "bind":
                    await websocket.send_text(json.dumps({"type": "error", "message": "first uplink message must be bind"}))
                    continue
                session_id = str(data.get("session_id") or "")
                session = registry.get(session_id)
                if not session:
                    await websocket.send_text(json.dumps({"type": "error", "message": "unknown session_id"}))
                    continue
                stream_id = session.transport.stream_id
                logger.info(
                    "browser_uplink_connected stream_id=%s session_id=%s elapsed_ms=%.2f",
                    stream_id,
                    session_id,
                    (time.perf_counter() - opened_at) * 1000,
                )
                continue

            if not session:
                await websocket.send_text(json.dumps({"type": "error", "message": "uplink not bound"}))
                continue

            raw = message.get("bytes")
            if raw is None:
                continue

            header, pcm_bytes = unpack_binary_audio_message(raw)
            if header.get("type") != "audio_in":
                await websocket.send_text(json.dumps({"type": "error", "message": "unsupported uplink frame"}))
                continue

            now = time.perf_counter()
            if not first_audio_logged:
                first_audio_logged = True
                logger.info(
                    "browser_audio_first_chunk stream_id=%s session_id=%s elapsed_ms=%.2f seq=%s",
                    stream_id,
                    session_id,
                    (now - opened_at) * 1000,
                    header.get("seq"),
                )
            if last_audio_received_at:
                uplink_gap_ms = (now - last_audio_received_at) * 1000
                if uplink_gap_ms > 200:
                    logger.info(
                        "browser_audio_gap_anomaly stream_id=%s session_id=%s gap_ms=%.2f seq=%s",
                        stream_id,
                        session_id,
                        uplink_gap_ms,
                        header.get("seq"),
                    )
            last_audio_received_at = now

            await session.handle_audio(
                pcm_bytes,
                int(header.get("sample_rate", 16000)),
                captured_at_ms=float(header.get("captured_at_ms", 0.0) or 0.0),
                received_at_perf=now,
                received_at_wall_ms=time.time() * 1000,
                seq=int(header.get("seq", 0) or 0),
            )

    except WebSocketDisconnect:
        logger.info(
            "browser uplink disconnected stream_id=%s session_id=%s total_elapsed_ms=%.2f",
            stream_id,
            session_id,
            (time.perf_counter() - opened_at) * 1000,
        )


@app.websocket("/ws/downlink-audio")
async def downlink_audio_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    opened_at = time.perf_counter()
    session: AgentSession | None = None
    stream_id = "unknown"
    session_id = "unknown"
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            if data.get("type") != "bind":
                await websocket.send_text(json.dumps({"type": "error", "message": "first downlink message must be bind"}))
                continue
            session_id = str(data.get("session_id") or "")
            session = registry.get(session_id)
            if not session:
                await websocket.send_text(json.dumps({"type": "error", "message": "unknown session_id"}))
                continue
            stream_id = session.transport.stream_id
            await session.transport.bind_audio_socket(websocket)
            logger.info(
                "browser_downlink_audio_connected stream_id=%s session_id=%s elapsed_ms=%.2f",
                stream_id,
                session_id,
                (time.perf_counter() - opened_at) * 1000,
            )
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"]:
                    data = json.loads(message["text"])
                    if data.get("type") == "playback_stop_ack":
                        logger.info(
                            "playback_stop_ack stream_id=%s session_id=%s response_id=%s elapsed_ms=%s",
                            stream_id,
                            session_id,
                            data.get("response_id"),
                            data.get("elapsed_ms"),
                        )
                elif message.get("bytes") is not None:
                    continue
    except WebSocketDisconnect:
        if session is not None:
            session.transport.unbind_audio_socket(websocket)
        logger.info(
            "browser downlink audio disconnected stream_id=%s session_id=%s total_elapsed_ms=%.2f",
            stream_id,
            session_id,
            (time.perf_counter() - opened_at) * 1000,
        )
