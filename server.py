import os
import base64
import json
import time
import asyncio
from typing import Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from service.engine import TurnTakingEngine
from service.session import TurnSession
from service.model import load_turn_model

SESSION_TTL_SEC = 60  # Recycle if no audio for 60 seconds
GC_INTERVAL_SEC = 10  # Session GC interval

# Global state
sessions: Dict[str, TurnSession] = {}

LOG_LEVEL_RANK = {"quiet": 0, "basic": 1, "debug": 2}
SERVER_LOG_LEVEL = os.getenv("TURN_SERVER_LOG_LEVEL", "basic").strip().lower()
if SERVER_LOG_LEVEL not in LOG_LEVEL_RANK:
    SERVER_LOG_LEVEL = "basic"


def should_log(required_level: str) -> bool:
    return LOG_LEVEL_RANK[SERVER_LOG_LEVEL] >= LOG_LEVEL_RANK[required_level]


def log(message: str, level: str = "basic"):
    if should_log(level):
        print(message)


# FastAPI Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    log("[TurnTaking] loading model ...", "basic")
    app.state.model = load_turn_model(
        config_path=os.path.join(os.path.dirname(__file__), "config/config.yaml")
    )
    log("[TurnTaking] model loaded", "basic")

    gc_task = asyncio.create_task(session_gc_loop())

    yield

    gc_task.cancel()
    log("[TurnTaking] shutdown", "basic")


app = FastAPI(lifespan=lifespan)


async def session_gc_loop():
    while True:
        now = time.time()
        expired = []

        for sid, sess in sessions.items():
            if getattr(sess, "processing", False):
                continue
            if now - sess.last_active_ts > SESSION_TTL_SEC:
                expired.append(sid)

        for sid in expired:
            log(f"[TurnTaking] GC session {sid}", "debug")
            del sessions[sid]

        await asyncio.sleep(GC_INTERVAL_SEC)


@app.websocket("/turn")
async def turn_ws(ws: WebSocket):
    await ws.accept()
    log("[TurnTaking] websocket connected", "basic")

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            message_type = data.get("type")

            if message_type == "reset":
                session_id = data.get("session_id")
                if session_id in sessions:
                    del sessions[session_id]
                    log(f"[TurnTaking] reset session {session_id}", "debug")
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "session_reset",
                            "session_id": session_id,
                            "ok": True,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if message_type != "audio":
                continue

            session_id = data["session_id"]

            if session_id not in sessions:
                engine = TurnTakingEngine(model=ws.app.state.model)
                sessions[session_id] = TurnSession(engine)
                log(f"[TurnTaking] new session {session_id}", "debug")

            session = sessions[session_id]
            session.touch()  # Update timestamp

            try:
                audio = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.float32)
            except Exception:
                continue

            chunk_len = len(audio)
            t_start = time.time()
            log(
                f"[TurnTaking] recv audio session={session_id} samples={chunk_len} processing={getattr(session, 'processing', False)}"
                ,
                "debug",
            )
            state = session.feed_audio(audio)
            elapsed = time.time() - t_start

            if state is not None:
                public_state = state.get("state")
                debug = state.get("debug", {})
                log(
                    "[TurnTaking] send state "
                    f"session={session_id} public={public_state} "
                    f"internal={debug.get('internal_state')} "
                    f"hint={debug.get('eval_label_hint')} "
                    f"delta={debug.get('delta_text', '')[:60]} "
                    f"cascade={debug.get('cascade_text', '')[:60]} "
                    f"text={state.get('text', '')[:60]} "
                    f"elapsed={elapsed:.3f}s"
                    ,
                    "debug",
                )
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "turn_state",
                            "session_id": session_id,
                            "state": state,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    )
                )

    except WebSocketDisconnect:
        log("[TurnTaking] websocket disconnected", "basic")

    except Exception as e:
        print(f"[TurnTaking] websocket error: {e}")
