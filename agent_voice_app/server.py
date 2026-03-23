from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.agent.session import AgentSession
from app.audio import b64decode_bytes
from app.transport import BrowserTransport

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

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

    started = False

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            message_type = data.get("type")

            if message_type == "start":
                try:
                    await session.start()
                    started = True
                except Exception as exc:
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
                pcm_bytes = b64decode_bytes(data.get("audio_b64", ""))
                await session.handle_audio(pcm_bytes, sample_rate)
                continue

            if message_type == "stop":
                break

            await transport.send_error(f"unsupported message type: {message_type}")

    except WebSocketDisconnect:
        pass
    finally:
        await session.stop()
