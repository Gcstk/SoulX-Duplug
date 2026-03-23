from __future__ import annotations

import base64

try:
    import audioop
except ModuleNotFoundError:  # pragma: no cover
    import audioop_lts as audioop

PCM_WIDTH = 2
TARGET_SAMPLE_RATE = 16000


def b64encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def pcm16_resample(pcm_bytes: bytes, sample_rate: int, target_sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate == target_sample_rate:
        return pcm_bytes
    converted, _ = audioop.ratecv(
        pcm_bytes,
        PCM_WIDTH,
        1,
        sample_rate,
        target_sample_rate,
        None,
    )
    return converted
