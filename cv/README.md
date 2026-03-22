# Easy-Turn WS Evaluation

This directory contains a streaming evaluation tool for running the Easy-Turn test set against a running SoulX-Duplug `/turn` WebSocket service.

The evaluator:

- scans the Easy-Turn manifest
- streams each waveform chunk-by-chunk to `/turn`
- uses the final ASR hypothesis to compute CER and WER
- uses the exposed internal state to compute 4-class label accuracy
- writes per-sample and summary reports under `cv/reports/`

## Service Requirements

The target SoulX-Duplug service must expose the current extended `/turn` protocol.

Input messages:

```json
{"type":"audio","session_id":"...","audio":"<base64 float32 pcm 16k mono>"}
{"type":"reset","session_id":"..."}
```

Audio response envelope:

```json
{
  "type": "turn_state",
  "session_id": "...",
  "state": {
    "state": "blank | idle | nonidle | speak",
    "text": "...",
    "asr_segment": "...",
    "asr_buffer": "...",
    "debug": {
      "internal_state": "<|user_idle|> | <|user_nonidle|> | <|user_backchannel|> | <|user_complete|> | <|user_incomplete|>",
      "eval_label_hint": "WAIT | BACKCHANNEL | COMPLETE | INCOMPLETE | null",
      "cascade_text": "...",
      "delta_text": "..."
    }
  },
  "ts": 0.0
}
```

Reset response:

```json
{
  "type": "session_reset",
  "session_id": "...",
  "ok": true,
  "ts": 0.0
}
```

## Dataset Layout

The evaluator expects an Easy-Turn style directory, for example:

```text
testset/
├── all_labels.tsv
├── backchannel/
├── complete/
├── incomplete/
└── wait/
```

It prefers `all_labels.tsv`. If that file does not exist, it falls back to merging each subset's `labels.tsv`.

Reference text priority:

1. `transcript`
2. `tagged_text` with `<...>` tags removed
3. `txt` with `<...>` tags removed

## Runtime Dependencies

The repo already contains the Python code. For actually running the evaluator you still need these runtime packages available in the active environment:

```bash
python -m pip install soundfile scipy websockets
```

`numpy` is also required, but it is already a core dependency of the project.

## Run Evaluation

Start the SoulX-Duplug service first:

```bash
bash run.sh
```

Then run evaluation:

```bash
python cv/eval_easy_turn_ws.py \
  --dataset-root /data/liuke_data/datasets/asr_datasets/Easy_turn/Easy-Turn-Testset/testset \
  --ws-url ws://127.0.0.1:8000/turn \
  --report-dir cv/reports
```

Optional arguments:

- `--chunk-samples`: streaming chunk size, default `2560`
- `--sample-rate`: target sample rate, default `16000`
- `--post-roll-ms`: trailing silence appended after each sample, default `2000`
- `--ws-timeout`: WebSocket timeout in seconds, default `5`

Example against a remote service:

```bash
python cv/eval_easy_turn_ws.py \
  --dataset-root /data/liuke_data/datasets/asr_datasets/Easy_turn/Easy-Turn-Testset/testset \
  --ws-url ws://10.0.0.8:8000/turn \
  --report-dir cv/reports
```

## Label Evaluation Rule

The evaluator uses the service's exposed internal labels rather than reconstructing labels from public `idle / nonidle / speak`.

Per-event mapping:

- `<|user_complete|>` -> `COMPLETE`
- `<|user_incomplete|>` -> `INCOMPLETE`
- `<|user_backchannel|>` -> `BACKCHANNEL`
- `<|user_idle|>` -> `WAIT`
- `<|user_nonidle|>` -> ignored for final label reduction

Per-sample reduction priority:

1. `COMPLETE`
2. `INCOMPLETE`
3. `BACKCHANNEL`
4. `WAIT`

## ASR Evaluation Rule

The evaluator uses the final hypothesis for CER and WER, not chunk-level `asr_segment`.

Final hypothesis priority:

1. last non-empty `state.text`
2. last non-empty `state.debug.cascade_text`
3. last non-empty `state.asr_buffer`
4. empty string

Text normalization:

- remove `<...>` tags
- apply project text normalization when available
- remove punctuation
- collapse repeated whitespace

CER is computed on normalized character sequences.  
WER is computed on normalized token sequences using the repo's `split_cn_en` tokenizer.

## Outputs

Reports are written to:

```text
cv/reports/<timestamp>/
```

Generated files:

- `samples.csv`
- `summary.json`

`samples.csv` columns:

- `key`
- `subset`
- `wav`
- `ref_label`
- `pred_label`
- `label_correct`
- `ref_text`
- `hyp_text`
- `cer`
- `wer`
- `public_final_state`
- `seen_internal_states`

`summary.json` contains:

- overall sample count
- overall 4-class label accuracy
- overall CER / WER
- per-subset accuracy and CER / WER
- 4x4 confusion matrix

## Notes

- The evaluator sends one fresh `session_id` per sample.
- It also sends a `reset` message after each sample so the service releases session state immediately.
- If your service is running behind another host or container, only `--ws-url` needs to change.
