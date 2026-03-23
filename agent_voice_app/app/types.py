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
    started_at: float = 0.0
    first_token_logged: bool = False
    first_audio_logged: bool = False


@dataclass
class SessionState:
    phase: Phase = Phase.LISTENING
    user_live_text: str = ""
    active_response: Optional[ActiveResponse] = None
    metadata: dict = field(default_factory=dict)
