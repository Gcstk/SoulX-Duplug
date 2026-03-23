import pytest

from app.agent.session import AgentSession


class FakeTransport:
    def __init__(self):
        self.phases = []
        self.transcripts = []
        self.audio = []
        self.clears = []

    async def send_phase(self, phase):
        self.phases.append(phase.value)

    async def send_transcript(self, speaker, text, final, response_id=None):
        self.transcripts.append((speaker, text, final, response_id))

    async def send_audio(self, audio_b64, sample_rate, response_id):
        self.audio.append((audio_b64, sample_rate, response_id))

    async def clear_audio(self, response_id=None):
        self.clears.append(response_id)


class FakeDuplug:
    def __init__(self, **callbacks):
        self.callbacks = callbacks

    async def start(self):
        return None

    async def stop(self):
        return None

    async def send_audio(self, pcm16_bytes, sample_rate):
        self.last_audio = (pcm16_bytes, sample_rate)


class FakeLLM:
    def __init__(self, on_token, on_done):
        self.on_token = on_token
        self.on_done = on_done
        self.cancelled = 0

    async def start_turn(self, user_text: str):
        self.last_user_text = user_text

    async def cancel(self):
        self.cancelled += 1


class FakeTTS:
    def __init__(self, on_audio, on_done):
        self.on_audio = on_audio
        self.on_done = on_done
        self.sent = []
        self.cancelled = 0
        self.sample_rate = 24000

    async def start(self):
        return None

    async def send(self, text: str):
        self.sent.append(text)

    async def flush(self):
        return None

    async def cancel(self):
        self.cancelled += 1


@pytest.mark.asyncio
async def test_interrupt_clears_current_response():
    transport = FakeTransport()
    session = AgentSession(
        transport=transport,
        duplug_client_factory=FakeDuplug,
        llm_factory=FakeLLM,
        tts_factory=FakeTTS,
    )

    await session.start()
    await session._on_user_turn_final("你好")
    active_id = session.state.active_response.response_id

    await session._on_user_speech_start()

    assert active_id in transport.clears
    assert session.state.active_response is None
    assert session.state.phase.value == "LISTENING"


@pytest.mark.asyncio
async def test_stale_audio_is_ignored_after_cancel():
    transport = FakeTransport()
    session = AgentSession(
        transport=transport,
        duplug_client_factory=FakeDuplug,
        llm_factory=FakeLLM,
        tts_factory=FakeTTS,
    )

    await session.start()
    await session._on_user_turn_final("你好")
    active_id = session.state.active_response.response_id
    await session._cancel_active_response(send_clear=True)
    await session._on_tts_audio(active_id, "Zm9v")

    assert transport.audio == []
