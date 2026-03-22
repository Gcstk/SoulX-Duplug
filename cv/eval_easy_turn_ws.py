from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
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


@dataclass
class SampleRecord:
    key: str
    subset: str
    label: str
    transcript: str
    wav_path: Path
    raw_row: dict


class TurnWSClient:
    def __init__(self, server_url: str, timeout: float = 5.0):
        self.server_url = server_url
        self.timeout = timeout
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

        self._ws = connect(
            self.server_url,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
            max_size=None,
        )

    def close(self):
        if self._ws is None:
            return
        with suppress(Exception):
            self._ws.close()
        self._ws = None

    def _send_json(self, payload: dict) -> dict:
        if self._ws is None:
            self.connect()

        try:
            self._ws.send(json.dumps(payload, ensure_ascii=False))
            return json.loads(self._ws.recv())
        except Exception:
            self.close()
            self.connect()
            self._ws.send(json.dumps(payload, ensure_ascii=False))
            return json.loads(self._ws.recv())

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


def extract_final_hypothesis(events: list[dict]) -> str:
    candidates = []
    for event in events:
        state = event.get("state", {})
        debug = state.get("debug", {})
        candidates.append(state.get("text") or "")
        candidates.append(debug.get("cascade_text") or "")
        candidates.append(state.get("asr_buffer") or "")

    for candidate in reversed(candidates):
        if candidate:
            return candidate.strip()
    return ""


def public_final_state(events: list[dict]) -> str:
    for event in reversed(events):
        state = event.get("state", {}).get("state")
        if state:
            return state
    return ""


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

    try:
        for chunk in iter_audio_chunks(audio, chunk_samples):
            events.append(client.process(session_id, chunk))

        post_roll_samples = int(sample_rate * post_roll_ms / 1000)
        if post_roll_samples > 0:
            silence = np.zeros(post_roll_samples, dtype=np.float32)
            for chunk in iter_audio_chunks(silence, chunk_samples):
                events.append(client.process(session_id, chunk))
    finally:
        client.reset(session_id)

    hypothesis = extract_final_hypothesis(events)
    prediction = reduce_predicted_label(events)
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
        "ref_text": record.transcript,
        "hyp_text": normalize_text(hypothesis),
        "cer": compute_cer(record.transcript, hypothesis),
        "wer": compute_wer(record.transcript, hypothesis),
        "public_final_state": public_final_state(events),
        "seen_internal_states": "|".join(collect_seen_internal_states(events)),
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
                "ref_text",
                "hyp_text",
                "cer",
                "wer",
                "public_final_state",
                "seen_internal_states",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summarize_rows(rows), handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_dir = (args.report_dir / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()

    records = load_dataset_records(dataset_root)
    rows = []

    with TurnWSClient(args.ws_url, timeout=args.ws_timeout) as client:
        for index, record in enumerate(records, start=1):
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
