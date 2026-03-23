from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.agent.session import AgentSession
from app.audio import b64decode_bytes
from app.logging_utils import get_logger
from app.transport import BrowserTransport

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
logger = get_logger("server")

app = FastAPI(title="SoulX Agent Voice App", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/web", status_code=307)


@app.get("/web")
async def web() -> FileResponse:
    return FileResponse(STATIC_DIR / "browser_agent.html")


@app.websocket("/ws/browser")
async def browser_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    transport = BrowserTransport(websocket)
    session = AgentSession(transport)
    await transport.send_ready()
    ws_opened_at = time.perf_counter()
    logger.info("browser websocket accepted stream_id=%s client=%s", transport.stream_id, getattr(websocket, "client", None))

    started = False
    audio_message_count = 0

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            message_type = data.get("type")
            logger.info(
                "browser message stream_id=%s type=%s elapsed_ms=%.2f",
                transport.stream_id,
                message_type,
                (time.perf_counter() - ws_opened_at) * 1000,
            )

            if message_type == "start":
                try:
                    await session.start()
                    started = True
                    logger.info("session started stream_id=%s sample_rate=%s", transport.stream_id, data.get("sample_rate"))
                except Exception as exc:
                    logger.exception("session start failed stream_id=%s error=%s", transport.stream_id, exc)
                    await transport.send_error(f"session start failed: {exc}")
                continue

            if message_type == "audio":
                if not started:
                    await transport.send_error("session not started")
                    continue
                if data.get("encoding") != "pcm16":
                    await transport.send_error("audio encoding must be pcm16")
                    continue
                sample_rate = int(data.get("sample_rate", 16000))
                captured_at_ms = data.get("captured_at_ms")
                pcm_bytes = b64decode_bytes(data.get("audio_b64", ""))
                audio_message_count += 1
                now_perf = time.perf_counter()
                logger.info(
                    "browser audio received stream_id=%s idx=%s sample_rate=%s bytes=%s elapsed_ms=%.2f captured_at_ms=%s",
                    transport.stream_id,
                    audio_message_count,
                    sample_rate,
                    len(pcm_bytes),
                    (now_perf - ws_opened_at) * 1000,
                    captured_at_ms,
                )
                await session.handle_audio(pcm_bytes, sample_rate, captured_at_ms=captured_at_ms, received_at_perf=now_perf)
                continue

            if message_type == "stop":
                logger.info("browser requested stop stream_id=%s", transport.stream_id)
                break

            await transport.send_error(f"unsupported message type: {message_type}")

    except WebSocketDisconnect:
        logger.info("browser websocket disconnected stream_id=%s", transport.stream_id)
    finally:
        logger.info(
            "browser session stopping stream_id=%s total_elapsed_ms=%.2f audio_messages=%s",
            transport.stream_id,
            (time.perf_counter() - ws_opened_at) * 1000,
            audio_message_count,
        )
        await session.stop()
