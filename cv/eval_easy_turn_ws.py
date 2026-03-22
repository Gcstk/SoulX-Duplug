from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
import time
import uuid
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.metrics import (
    compute_cer,
    compute_wer,
    levenshtein_distance,
    normalize_text,
    strip_tags,
    char_tokens,
    word_tokens,
)


LABEL_ORDER = ["BACKCHANNEL", "COMPLETE", "INCOMPLETE", "WAIT"]
LABEL_PRIORITY = {
    "COMPLETE": 3,
    "INCOMPLETE": 2,
    "BACKCHANNEL": 1,
    "WAIT": 0,
}
LOG_LEVEL_RANK = {"quiet": 0, "basic": 1, "debug": 2}


def should_log(log_level: str, required_level: str) -> bool:
    return LOG_LEVEL_RANK[log_level] >= LOG_LEVEL_RANK[required_level]


@dataclass
class SampleRecord:
    key: str
    subset: str
    label: str
    transcript: str
    wav_path: Path
    raw_row: dict


class TurnWSClient:
    def __init__(self, server_url: str, timeout: float = 5.0, log_level: str = "basic"):
        self.server_url = server_url
        self.timeout = timeout
        self.log_level = log_level
        self._ws = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        if self._ws is not None:
            return
        from websockets.sync.client import connect

        if should_log(self.log_level, "debug"):
            print(
                f"[WS] connect url={self.server_url} open_timeout={self.timeout}s recv_timeout={self.timeout}s"
            )
        self._ws = connect(
            self.server_url,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
            max_size=None,
        )

    def close(self):
        if self._ws is None:
            return
        if should_log(self.log_level, "debug"):
            print("[WS] close connection")
        with suppress(Exception):
            self._ws.close()
        self._ws = None

    def _send_json(self, payload: dict, response_timeout: float | None = None) -> dict:
        if self._ws is None:
            self.connect()

        try:
            send_start = time.time()
            if should_log(self.log_level, "debug"):
                print(
                    f"[WS] send type={payload.get('type')} session_id={payload.get('session_id')}"
                )
            self._ws.send(json.dumps(payload, ensure_ascii=False))
            raw_response = self._ws.recv(timeout=response_timeout or self.timeout)
            response = json.loads(raw_response)
            if should_log(self.log_level, "debug"):
                state = response.get("state", {})
                debug = state.get("debug", {})
                print(
                    "[WS] recv "
                    f"type={response.get('type')} "
                    f"public={state.get('state')} "
                    f"internal={debug.get('internal_state')} "
                    f"hint={debug.get('eval_label_hint')} "
                    f"delta={debug.get('delta_text', '')[:60]} "
                    f"cascade={debug.get('cascade_text', '')[:60]} "
                    f"text={state.get('text', '')[:60]} "
                    f"elapsed={time.time() - send_start:.3f}s"
                )
            return response
        except TimeoutError as exc:
            if should_log(self.log_level, "basic"):
                print(
                    f"[WS] recv timeout type={payload.get('type')} session_id={payload.get('session_id')} timeout={response_timeout or self.timeout}s"
                )
            raise TimeoutError(
                f"Timed out waiting for WS response after {response_timeout or self.timeout}s "
                f"for type={payload.get('type')} session_id={payload.get('session_id')}"
            ) from exc
        except Exception as exc:
            if should_log(self.log_level, "basic"):
                print(
                    f"[WS] request failed type={payload.get('type')} session_id={payload.get('session_id')} error={exc!r}; reconnecting"
                )
            self.close()
            self.connect()
            self._ws.send(json.dumps(payload, ensure_ascii=False))
            raw_response = self._ws.recv(timeout=response_timeout or self.timeout)
            response = json.loads(raw_response)
            if should_log(self.log_level, "debug"):
                state = response.get("state", {})
                debug = state.get("debug", {})
                print(
                    "[WS] recv-after-reconnect "
                    f"type={response.get('type')} "
                    f"public={state.get('state')} "
                    f"internal={debug.get('internal_state')} "
                    f"hint={debug.get('eval_label_hint')} "
                    f"delta={debug.get('delta_text', '')[:60]} "
                    f"cascade={debug.get('cascade_text', '')[:60]} "
                    f"text={state.get('text', '')[:60]}"
                )
            return response

    def process(self, session_id: str, audio_chunk: np.ndarray) -> dict:
        payload = {
            "type": "audio",
            "session_id": session_id,
            "audio": base64.b64encode(
                np.asarray(audio_chunk, dtype=np.float32).tobytes()
            ).decode("utf-8"),
        }
        return self._send_json(payload)

    def reset(self, session_id: str) -> dict:
        return self._send_json({"type": "reset", "session_id": session_id})


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Easy-Turn over WS /turn")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("cv/reports"))
    parser.add_argument("--chunk-samples", type=int, default=2560)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--post-roll-ms", type=int, default=2000)
    parser.add_argument("--ws-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log-level",
        choices=["quiet", "basic", "debug"],
        default="basic",
        help="Evaluation logging verbosity. Default: basic.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Deprecated alias for --log-level quiet.",
    )
    return parser.parse_args()


def normalize_label(label: str | None) -> str:
    label = (label or "").strip().upper().replace("<", "").replace(">", "")
    if label not in LABEL_PRIORITY:
        raise ValueError(f"Unsupported label: {label}")
    return label


def extract_reference_text(row: dict) -> str:
    for key in ("transcript", "tagged_text", "txt"):
        value = row.get(key)
        if value:
            return strip_tags(value).strip()
    return ""


def resolve_wav_path(dataset_root: Path, wav_value: str) -> Path:
    wav_path = Path(wav_value)
    if wav_path.is_absolute():
        return wav_path
    return (dataset_root / wav_path).resolve()


def load_dataset_records(dataset_root: Path) -> list[SampleRecord]:
    manifest_path = dataset_root / "all_labels.tsv"
    rows: list[dict] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle, delimiter="\t"))
    else:
        for subset_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            labels_path = subset_dir / "labels.tsv"
            if not labels_path.exists():
                continue
            with labels_path.open("r", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle, delimiter="\t"))

    records = []
    for row in rows:
        subset = (row.get("subset") or Path(row["wav"]).parts[0]).strip("./")
        records.append(
            SampleRecord(
                key=row["key"],
                subset=subset,
                label=normalize_label(row.get("label")),
                transcript=extract_reference_text(row),
                wav_path=resolve_wav_path(dataset_root, row["wav"]),
                raw_row=row,
            )
        )
    return records


def load_audio_as_float32(path: Path, target_sample_rate: int) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'soundfile'. Install it before running evaluation."
        ) from exc

    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sample_rate != target_sample_rate:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'scipy'. Install it before running evaluation."
            ) from exc

        gcd = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(audio, target_sample_rate // gcd, sample_rate // gcd)
        audio = np.asarray(audio, dtype=np.float32)

    return audio


def iter_audio_chunks(audio: np.ndarray, chunk_samples: int) -> Iterable[np.ndarray]:
    total = len(audio)
    for start in range(0, total, chunk_samples):
        yield np.asarray(audio[start : start + chunk_samples], dtype=np.float32)


def collect_seen_internal_states(events: list[dict]) -> list[str]:
    seen = []
    for event in events:
        internal_state = event.get("state", {}).get("debug", {}).get("internal_state")
        if internal_state and internal_state not in seen:
            seen.append(internal_state)
    return seen


def reduce_predicted_label(events: list[dict]) -> str:
    best_label = "WAIT"
    best_priority = LABEL_PRIORITY[best_label]

    for event in events:
        label = event.get("state", {}).get("debug", {}).get("eval_label_hint")
        if label is None:
            continue
        priority = LABEL_PRIORITY[label]
        if priority > best_priority:
            best_label = label
            best_priority = priority

    return best_label


def _event_hint(event: dict) -> str | None:
    return event.get("state", {}).get("debug", {}).get("eval_label_hint")


def collect_segment_summaries(events: list[dict]) -> list[dict]:
    segments = []
    current_label = None

    for event in events:
        state = event.get("state", {})
        public_state = state.get("state")
        hint = _event_hint(event)

        if hint and (hint != "WAIT" or current_label is None):
            current_label = hint

        if public_state == "speak":
            segment_label = current_label or hint or "WAIT"
            segments.append(
                {
                    "label": segment_label,
                    "text": (state.get("text") or "").strip(),
                }
            )
            current_label = None

    return segments


def last_non_empty_residual(events: list[dict]) -> str:
    candidates = []
    for event in events:
        state = event.get("state", {})
        debug = state.get("debug", {})
        candidates.append(debug.get("cascade_text") or "")
        candidates.append(state.get("asr_buffer") or "")

    for candidate in reversed(candidates):
        if candidate:
            return candidate.strip()
    return ""


def extract_final_hypothesis(events: list[dict]) -> str:
    segments = collect_segment_summaries(events)
    pieces = []

    for segment in segments:
        text = segment["text"]
        if text and (not pieces or pieces[-1] != text):
            pieces.append(text)

    residual = last_non_empty_residual(events)
    if residual and (not pieces or residual not in pieces[-1]):
        pieces.append(residual)

    if pieces:
        return "".join(pieces).strip()
    return residual


def public_final_state(events: list[dict]) -> str:
    for event in reversed(events):
        state = event.get("state", {}).get("state")
        if state:
            return state
    return ""


def first_segment_label(events: list[dict]) -> str:
    segments = collect_segment_summaries(events)
    if segments:
        return segments[0]["label"]
    return "WAIT"


def last_segment_label(events: list[dict]) -> str:
    segments = collect_segment_summaries(events)
    if segments:
        return segments[-1]["label"]
    return "WAIT"


def fallback_stream_label(events: list[dict]) -> str:
    last_hint = None
    last_non_wait = None

    for event in events:
        hint = _event_hint(event)
        if hint is None:
            continue
        last_hint = hint
        if hint != "WAIT":
            last_non_wait = hint

    return last_non_wait or last_hint or "WAIT"


def evaluate_sample(
    client: TurnWSClient,
    record: SampleRecord,
    chunk_samples: int,
    sample_rate: int,
    post_roll_ms: int,
) -> dict:
    session_id = f"{record.key}-{uuid.uuid4().hex}"
    audio = load_audio_as_float32(record.wav_path, sample_rate)
    events = []
    total_chunks = math.ceil(len(audio) / chunk_samples) if len(audio) else 0
    log_level = getattr(client, "log_level", "basic")
    if should_log(log_level, "basic"):
        print(
            f"[Eval] start key={record.key} subset={record.subset} label={record.label} "
            f"samples={len(audio)} chunks={total_chunks} wav={record.wav_path}"
        )

    try:
        for chunk_index, chunk in enumerate(iter_audio_chunks(audio, chunk_samples), start=1):
            if should_log(log_level, "debug"):
                print(
                    f"[Eval] chunk key={record.key} index={chunk_index}/{max(total_chunks, 1)} samples={len(chunk)}"
                )
            events.append(client.process(session_id, chunk))

        post_roll_samples = int(sample_rate * post_roll_ms / 1000)
        if post_roll_samples > 0:
            silence = np.zeros(post_roll_samples, dtype=np.float32)
            total_post_chunks = math.ceil(post_roll_samples / chunk_samples)
            if should_log(log_level, "debug"):
                print(
                    f"[Eval] post-roll key={record.key} ms={post_roll_ms} samples={post_roll_samples} chunks={total_post_chunks}"
                )
            for chunk_index, chunk in enumerate(
                iter_audio_chunks(silence, chunk_samples), start=1
            ):
                if should_log(log_level, "debug"):
                    print(
                        f"[Eval] post-roll chunk key={record.key} index={chunk_index}/{max(total_post_chunks, 1)} samples={len(chunk)}"
                    )
                events.append(client.process(session_id, chunk))
    finally:
        if should_log(log_level, "debug"):
            print(f"[Eval] reset session key={record.key} session_id={session_id}")
        try:
            client.reset(session_id)
        except Exception as exc:
            if should_log(log_level, "basic"):
                print(
                    f"[Eval] reset failed key={record.key} session_id={session_id} error={exc!r}"
                )

    segments = collect_segment_summaries(events)
    hypothesis = extract_final_hypothesis(events)
    fallback_prediction = fallback_stream_label(events)
    prediction = segments[-1]["label"] if segments else fallback_prediction
    first_prediction = segments[0]["label"] if segments else fallback_prediction
    any_positive_prediction = reduce_predicted_label(events)
    ref_char_tokens = char_tokens(record.transcript)
    hyp_char_tokens = char_tokens(hypothesis)
    ref_word_tokens = word_tokens(record.transcript)
    hyp_word_tokens = word_tokens(hypothesis)

    return {
        "key": record.key,
        "subset": record.subset,
        "wav": str(record.wav_path),
        "ref_label": record.label,
        "pred_label": prediction,
        "label_correct": prediction == record.label,
        "first_segment_label": first_prediction,
        "last_segment_label": prediction,
        "any_positive_label": any_positive_prediction,
        "ref_text": record.transcript,
        "hyp_text": normalize_text(hypothesis),
        "cer": compute_cer(record.transcript, hypothesis),
        "wer": compute_wer(record.transcript, hypothesis),
        "public_final_state": public_final_state(events),
        "seen_internal_states": "|".join(collect_seen_internal_states(events)),
        "segment_labels": "|".join(segment["label"] for segment in segments),
        "char_edits": levenshtein_distance(ref_char_tokens, hyp_char_tokens),
        "char_ref_len": len(ref_char_tokens),
        "word_edits": levenshtein_distance(ref_word_tokens, hyp_word_tokens),
        "word_ref_len": len(ref_word_tokens),
    }


def build_confusion_matrix(rows: list[dict]) -> dict:
    matrix = {
        ref_label: {pred_label: 0 for pred_label in LABEL_ORDER}
        for ref_label in LABEL_ORDER
    }
    for row in rows:
        matrix[row["ref_label"]][row["pred_label"]] += 1
    return matrix


def summarize_overall(rows: list[dict]) -> dict:
    total = len(rows)
    label_correct = sum(1 for row in rows if row["label_correct"])
    char_edits = sum(row["char_edits"] for row in rows)
    char_ref_len = sum(row["char_ref_len"] for row in rows)
    word_edits = sum(row["word_edits"] for row in rows)
    word_ref_len = sum(row["word_ref_len"] for row in rows)

    return {
        "samples": total,
        "label_accuracy": (label_correct / total) if total else 0.0,
        "cer": (char_edits / max(char_ref_len, 1)),
        "wer": (word_edits / max(word_ref_len, 1)),
    }


def summarize_rows(rows: list[dict]) -> dict:
    total_summary = summarize_overall(rows)

    by_subset = defaultdict(list)
    for row in rows:
        by_subset[row["subset"]].append(row)

    subset_summary = {}
    for subset, subset_rows in sorted(by_subset.items()):
        subset_summary[subset] = summarize_overall(subset_rows)

    return {
        "overall": total_summary,
        "by_subset": subset_summary,
        "label_order": LABEL_ORDER,
        "confusion_matrix": build_confusion_matrix(rows),
    }


def write_reports(rows: list[dict], report_root: Path):
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "samples.csv"
    json_path = report_root / "summary.json"

    sample_rows = []
    for row in rows:
        sample_row = dict(row)
        sample_row.pop("char_edits", None)
        sample_row.pop("char_ref_len", None)
        sample_row.pop("word_edits", None)
        sample_row.pop("word_ref_len", None)
        sample_rows.append(sample_row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "subset",
                "wav",
                "ref_label",
                "pred_label",
                "label_correct",
                "first_segment_label",
                "last_segment_label",
                "any_positive_label",
                "ref_text",
                "hyp_text",
                "cer",
                "wer",
                "public_final_state",
                "seen_internal_states",
                "segment_labels",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summarize_rows(rows), handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def main():
    args = parse_args()
    if args.quiet:
        args.log_level = "quiet"
    dataset_root = args.dataset_root.resolve()
    report_dir = (args.report_dir / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()

    records = load_dataset_records(dataset_root)
    rows = []

    with TurnWSClient(
        args.ws_url,
        timeout=args.ws_timeout,
        log_level=args.log_level,
    ) as client:
        for index, record in enumerate(records, start=1):
            if should_log(args.log_level, "basic"):
                print(f"[{index}/{len(records)}] {record.key} -> {record.label}")
            rows.append(
                evaluate_sample(
                    client=client,
                    record=record,
                    chunk_samples=args.chunk_samples,
                    sample_rate=args.sample_rate,
                    post_roll_ms=args.post_roll_ms,
                )
            )

    csv_path, json_path = write_reports(rows, report_dir)
    print(f"Saved sample report to {csv_path}")
    print(f"Saved summary report to {json_path}")


if __name__ == "__main__":
    main()
