import asyncio

import pytest

from app.agent.session import AgentSession


class FakeTransport:
    def __init__(self):
        self.stream_id = "browser-test"
        self.session_id = "session-test"
        self.phases = []
        self.transcripts = []
        self.audio = []
        self.clears = []
        self.metrics = []
        self.errors = []
        self.turn_events = []
        self.invalidated = []
        self.last_clear_sent_at = 0.0

    async def send_phase(self, phase):
        self.phases.append(phase.value)

    async def send_transcript(self, speaker, text, final, response_id=None):
        self.transcripts.append((speaker, text, final, response_id))

    async def send_audio(self, audio_b64, sample_rate, response_id):
        self.audio.append((audio_b64, sample_rate, response_id))

    async def clear_audio(self, response_id=None):
        self.clears.append(response_id)
        self.last_clear_sent_at = asyncio.get_running_loop().time()

    async def send_metrics(self, payload):
        self.metrics.append(payload)

    async def send_error(self, message):
        self.errors.append(message)

    async def send_turn_debug(self, payload):
        return None

    async def send_turn_event(self, kind, text=""):
        self.turn_events.append((kind, text))

    async def invalidate_response(self, response_id):
        self.invalidated.append(response_id)

    def queue_metrics(self):
        return {
            "control_queue_peak_p0": 0,
            "control_queue_peak_p1": 0,
            "control_queue_peak_p2": 0,
            "stale_media_dropped_count": 0,
        }


class FakeDuplug:
    def __init__(self, **callbacks):
        self.callbacks = callbacks

    async def start(self):
        return None

    async def stop(self):
        return None

    async def send_audio(self, pcm16_bytes, sample_rate, **kwargs):
        self.last_audio = (pcm16_bytes, sample_rate, kwargs)


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
    instances = []

    def __init__(self, on_audio, on_done, on_ready=None):
        self.on_audio = on_audio
        self.on_done = on_done
        self.on_ready = on_ready
        self.sent = []
        self.cancelled = 0
        self.sample_rate = 24000
        self.started = False
        FakeTTS.instances.append(self)

    async def start(self):
        self.started = True
        if self.on_ready:
            await self.on_ready()

    async def send(self, text: str):
        self.sent.append(text)

    async def flush(self):
        return None

    async def cancel(self):
        self.cancelled += 1

    @property
    def is_broken(self):
        return False

    @property
    def is_running(self):
        return True

    def bind(self, on_audio, on_done, on_ready=None):
        self.on_audio = on_audio
        self.on_done = on_done
        self.on_ready = on_ready

    def reset_for_next_turn(self):
        return None


class FakePool:
    def __init__(self, tts_factory):
        self.tts_factory = tts_factory
        self.released = []
        self.hit = True

    async def get(self):
        return self.tts_factory(None, None, None), self.hit

    async def release(self, tts, broken=False):
        self.released.append((tts, broken))


@pytest.mark.asyncio
async def test_interrupt_clears_current_response():
    FakeTTS.instances.clear()
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
    assert active_id in transport.invalidated
    assert session.state.active_response is None
    assert session.state.phase.value == "LISTENING"


@pytest.mark.asyncio
async def test_stale_audio_is_ignored_after_cancel():
    FakeTTS.instances.clear()
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


class SlowReadyTTS(FakeTTS):
    def __init__(self, on_audio, on_done, on_ready=None):
        super().__init__(on_audio, on_done, on_ready=on_ready)
        self.ready_gate = None

    async def start(self):
        self.started = True
        self.ready_gate = self.ready_gate or asyncio.Event()
        await self.ready_gate.wait()
        if self.on_ready:
            await self.on_ready()


@pytest.mark.asyncio
async def test_llm_dispatch_does_not_wait_for_tts_ready():
    transport = FakeTransport()
    pool = FakePool(SlowReadyTTS)
    session = AgentSession(
        transport=transport,
        duplug_client_factory=FakeDuplug,
        llm_factory=FakeLLM,
        tts_factory=SlowReadyTTS,
        tts_pool=pool,
    )

    await session.start()
    await session._on_user_turn_final("你好")

    assert session.state.active_response is not None
    assert session._llm.last_user_text == "你好"
    assert session.state.active_response.pending_tts_tokens == []


@pytest.mark.asyncio
async def test_tokens_buffer_until_tts_ready_then_flush():
    transport = FakeTransport()
    session = AgentSession(
        transport=transport,
        duplug_client_factory=FakeDuplug,
        llm_factory=FakeLLM,
        tts_factory=SlowReadyTTS,
    )

    await session.start()
    await session._on_user_turn_final("你好")
    response = session.state.active_response
    while session._tts is None:
        await asyncio.sleep(0)
    tts = session._tts

    await session._on_llm_token("你")
    await session._on_llm_token("好")

    assert response.pending_tts_tokens == ["你", "好"]
    assert tts.sent == []

    tts.ready_gate.set()
    await session._tts_acquire_task

    assert response.pending_tts_tokens == []
    assert tts.sent == ["你", "好"]


@pytest.mark.asyncio
async def test_complete_metrics_include_pool_fields():
    transport = FakeTransport()
    pool = FakePool(FakeTTS)
    session = AgentSession(
        transport=transport,
        duplug_client_factory=FakeDuplug,
        llm_factory=FakeLLM,
        tts_factory=FakeTTS,
        tts_pool=pool,
    )

    await session.start()
    await session._on_user_turn_final("你好")
    response_id = session.state.active_response.response_id
    await session._on_llm_token("好")
    await session._on_llm_done()
    await session._on_tts_audio(response_id, "Zm9v")
    await session._on_tts_done(response_id)

    assert transport.metrics
    assert "tts_pool_acquire_ms" in transport.metrics[-1]
    assert "tts_pool_hit" in transport.metrics[-1]
