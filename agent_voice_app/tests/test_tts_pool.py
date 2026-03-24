import pytest

from app.services.tts_qwen_pool import QwenTTSPool


class FakePooledTTS:
    def __init__(self):
        self.started = 0
        self.cancelled = 0
        self.bound = 0
        self.running = False
        self.broken = False

    async def start(self):
        self.started += 1
        self.running = True

    async def cancel(self):
        self.cancelled += 1
        self.running = False

    @property
    def is_running(self):
        return self.running

    @property
    def is_broken(self):
        return self.broken

    def reset_for_next_turn(self):
        return None

    def bind(self, on_audio, on_done, on_ready=None):
        self.bound += 1


@pytest.mark.asyncio
async def test_tts_pool_hit_returns_warm_connection():
    created = []

    def factory():
        tts = FakePooledTTS()
        created.append(tts)
        return tts

    pool = QwenTTSPool(pool_size=1, ttl=8.0, service_factory=factory)
    await pool.start()
    tts1, hit1 = await pool.get()
    assert hit1 is False
    assert pool.pool_miss == 1
    await pool.release(tts1, broken=False)
    tts2, hit = await pool.get()
    assert hit is True
    assert pool.pool_hit == 1
    assert tts2 in created
    await pool.stop()


@pytest.mark.asyncio
async def test_tts_pool_release_refills():
    created = []

    def factory():
        tts = FakePooledTTS()
        created.append(tts)
        return tts

    pool = QwenTTSPool(pool_size=1, ttl=8.0, service_factory=factory)
    await pool.start()
    tts, _ = await pool.get()
    await pool.release(tts, broken=False)
    assert pool.available >= 1
    await pool.stop()
