from __future__ import annotations

import base64
import json
import struct

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


def pack_binary_audio_message(metadata: dict, payload: bytes) -> bytes:
    # Binary audio frames use a tiny JSON header plus raw PCM payload to avoid
    # base64 overhead while still keeping the wire format easy to inspect.
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(header)) + header + payload


def unpack_binary_audio_message(frame: bytes) -> tuple[dict, bytes]:
    if len(frame) < 4:
        raise ValueError("frame too short")
    (header_len,) = struct.unpack("!I", frame[:4])
    if header_len <= 0 or len(frame) < 4 + header_len:
        raise ValueError("invalid header length")
    header = json.loads(frame[4 : 4 + header_len].decode("utf-8"))
    return header, frame[4 + header_len :]


def pcm16_frame_duration_ms(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> float:
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("sample_rate and channels must be positive")
    samples = len(pcm_bytes) / (PCM_WIDTH * channels)
    return (samples / sample_rate) * 1000


def chunk_pcm16(pcm_bytes: bytes, sample_rate: int, max_chunk_ms: float, channels: int = 1) -> list[bytes]:
    if not pcm_bytes:
        return []
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be positive")
    samples_per_chunk = max(1, int(sample_rate * (max_chunk_ms / 1000)))
    bytes_per_chunk = samples_per_chunk * PCM_WIDTH * channels
    return [pcm_bytes[idx : idx + bytes_per_chunk] for idx in range(0, len(pcm_bytes), bytes_per_chunk)]
