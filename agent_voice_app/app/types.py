from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(str, Enum):
    LISTENING = "LISTENING"
    RESPONDING = "RESPONDING"


@dataclass
class ActiveResponse:
    response_id: str
    assistant_text: str = ""
    llm_done: bool = False
    tts_done: bool = False
    cancelled: bool = False
    response_started_at: float = 0.0
    response_started_wall_ms: float = 0.0
    tts_start_requested_at: float = 0.0
    tts_pool_acquired_at: float = 0.0
    tts_ready_at: float = 0.0
    llm_dispatch_at: float = 0.0
    llm_first_token_at: float = 0.0
    tts_first_audio_at: float = 0.0
    asr_final_at: float = 0.0
    asr_final_wall_ms: float = 0.0
    barge_in_at: float = 0.0
    pending_tts_tokens: list[str] = field(default_factory=list)
    tts_ready: bool = False
    tts_failed: bool = False
    first_token_logged: bool = False
    first_audio_logged: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class SessionState:
    phase: Phase = Phase.LISTENING
    user_live_text: str = ""
    active_response: Optional[ActiveResponse] = None
    metadata: dict = field(default_factory=dict)
