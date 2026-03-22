from pathlib import Path

import numpy as np

from cv.eval_easy_turn_ws import (
    SampleRecord,
    build_confusion_matrix,
    evaluate_sample,
    extract_final_hypothesis,
    load_dataset_records,
    reduce_predicted_label,
    summarize_rows,
)


def test_load_dataset_records_from_all_labels(tmp_path: Path):
    (tmp_path / "complete" / "real").mkdir(parents=True)
    labels_path = tmp_path / "all_labels.tsv"
    labels_path.write_text(
        "key\tsubset\tlabel\ttranscript\ttagged_text\ttask\twav\tsource_list\n"
        "complete_real_001\tcomplete\tCOMPLETE\t你好吗\t你好吗<COMPLETE>\t<TRANSCRIBE>\t./complete/real/complete_real_001.wav\tlist\n",
        encoding="utf-8",
    )

    records = load_dataset_records(tmp_path)

    assert len(records) == 1
    assert records[0].label == "COMPLETE"
    assert records[0].transcript == "你好吗"
    assert records[0].wav_path == (tmp_path / "complete" / "real" / "complete_real_001.wav").resolve()


def test_reduce_predicted_label_and_final_hypothesis():
    events = [
        {
            "state": {
                "state": "nonidle",
                "asr_buffer": "你",
                "debug": {
                    "internal_state": "<|user_nonidle|>",
                    "eval_label_hint": None,
                    "cascade_text": "你",
                    "delta_text": "你",
                },
            }
        },
        {
            "state": {
                "state": "idle",
                "asr_buffer": "你好吗",
                "debug": {
                    "internal_state": "<|user_incomplete|>",
                    "eval_label_hint": "INCOMPLETE",
                    "cascade_text": "你好吗",
                    "delta_text": "",
                },
            }
        },
        {
            "state": {
                "state": "speak",
                "text": "你好吗",
                "debug": {
                    "internal_state": "<|user_complete|>",
                    "eval_label_hint": "COMPLETE",
                    "cascade_text": "你好吗",
                    "delta_text": "",
                },
            }
        },
    ]

    assert reduce_predicted_label(events) == "COMPLETE"
    assert extract_final_hypothesis(events) == "你好吗"


def test_evaluate_sample_resets_session(monkeypatch):
    events = [
        {
            "state": {
                "state": "nonidle",
                "asr_buffer": "你",
                "debug": {
                    "internal_state": "<|user_nonidle|>",
                    "eval_label_hint": None,
                    "cascade_text": "你",
                    "delta_text": "你",
                },
            }
        },
        {
            "state": {
                "state": "idle",
                "asr_buffer": "你好吗",
                "debug": {
                    "internal_state": "<|user_incomplete|>",
                    "eval_label_hint": "INCOMPLETE",
                    "cascade_text": "你好吗",
                    "delta_text": "",
                },
            }
        },
        {
            "state": {
                "state": "speak",
                "text": "你好吗",
                "debug": {
                    "internal_state": "<|user_complete|>",
                    "eval_label_hint": "COMPLETE",
                    "cascade_text": "你好吗",
                    "delta_text": "",
                },
            }
        },
    ]

    class FakeClient:
        def __init__(self):
            self.index = 0
            self.reset_session_id = None

        def process(self, session_id, audio_chunk):
            event = events[self.index]
            self.index += 1
            return event

        def reset(self, session_id):
            self.reset_session_id = session_id
            return {"type": "session_reset", "session_id": session_id, "ok": True}

    monkeypatch.setattr(
        "cv.eval_easy_turn_ws.load_audio_as_float32",
        lambda path, sample_rate: np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    )

    record = SampleRecord(
        key="sample-1",
        subset="complete",
        label="COMPLETE",
        transcript="你好吗",
        wav_path=Path("/tmp/sample.wav"),
        raw_row={},
    )
    client = FakeClient()

    row = evaluate_sample(
        client=client,
        record=record,
        chunk_samples=2,
        sample_rate=16000,
        post_roll_ms=0,
    )

    assert row["pred_label"] == "COMPLETE"
    assert row["label_correct"] is True
    assert row["public_final_state"] == "speak"
    assert row["hyp_text"] == "你好吗"
    assert client.reset_session_id is not None


def test_summary_and_confusion_matrix():
    rows = [
        {
            "key": "a",
            "subset": "complete",
            "wav": "a.wav",
            "ref_label": "COMPLETE",
            "pred_label": "COMPLETE",
            "label_correct": True,
            "ref_text": "你好",
            "hyp_text": "你好",
            "cer": 0.0,
            "wer": 0.0,
            "public_final_state": "speak",
            "seen_internal_states": "<|user_complete|>",
            "char_edits": 0,
            "char_ref_len": 2,
            "word_edits": 0,
            "word_ref_len": 2,
        },
        {
            "key": "b",
            "subset": "wait",
            "wav": "b.wav",
            "ref_label": "WAIT",
            "pred_label": "BACKCHANNEL",
            "label_correct": False,
            "ref_text": "",
            "hyp_text": "",
            "cer": 0.0,
            "wer": 0.0,
            "public_final_state": "idle",
            "seen_internal_states": "<|user_backchannel|>",
            "char_edits": 0,
            "char_ref_len": 0,
            "word_edits": 0,
            "word_ref_len": 0,
        },
    ]

    summary = summarize_rows(rows)
    matrix = build_confusion_matrix(rows)

    assert summary["overall"]["samples"] == 2
    assert summary["overall"]["label_accuracy"] == 0.5
    assert summary["by_subset"]["complete"]["samples"] == 1
    assert matrix["COMPLETE"]["COMPLETE"] == 1
    assert matrix["WAIT"]["BACKCHANNEL"] == 1
