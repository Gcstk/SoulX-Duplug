from __future__ import annotations

import os
import time
import asyncio

import pytest

from app.services.llm import LLMService


pytestmark = pytest.mark.integration


def _missing_env() -> list[str]:
    required = ["LLM_API_KEY", "LLM_MODEL"]
    missing = [name for name in required if not os.getenv(name)]
    return missing


@pytest.mark.asyncio
async def test_llm_real_latency_smoke():
    if os.getenv("RUN_LLM_INTEGRATION") != "1":
        pytest.skip("set RUN_LLM_INTEGRATION=1 to run real LLM integration test")

    missing = _missing_env()
    if missing:
        pytest.skip(f"missing env vars: {', '.join(missing)}")

    tokens: list[str] = []
    events: dict[str, float] = {}

    async def on_token(token: str) -> None:
        if "first_token" not in events:
            events["first_token"] = time.perf_counter()
        tokens.append(token)

    async def on_done() -> None:
        events["done"] = time.perf_counter()

    service = LLMService(on_token=on_token, on_done=on_done)
    prompt = os.getenv("LLM_INTEGRATION_PROMPT", "请用一句中文简短介绍你自己。")

    started_at = time.perf_counter()
    await service.start_turn(prompt)

    timeout_s = float(os.getenv("LLM_INTEGRATION_TIMEOUT", "30"))
    try:
        await asyncio.wait_for(service._task, timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        await service.cancel()
        raise AssertionError(f"LLM integration test timed out after {timeout_s}s") from exc

    assert tokens, "LLM returned no tokens"

    first_token_ms = ((events["first_token"] - started_at) * 1000) if "first_token" in events else None
    total_ms = ((events["done"] - started_at) * 1000) if "done" in events else None
    text = "".join(tokens)

    print(
        "\nLLM integration metrics:",
        {
            "model": os.getenv("LLM_MODEL"),
            "prompt_chars": len(prompt),
            "output_chars": len(text),
            "first_token_ms": round(first_token_ms, 2) if first_token_ms is not None else None,
            "total_ms": round(total_ms, 2) if total_ms is not None else None,
            "preview": text[:120],
        },
    )

    assert first_token_ms is not None
