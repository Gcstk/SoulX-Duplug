from __future__ import annotations


INTERNAL_STATE_TO_EVAL_LABEL = {
    "<|user_complete|>": "COMPLETE",
    "<|user_incomplete|>": "INCOMPLETE",
    "<|user_backchannel|>": "BACKCHANNEL",
    "<|user_idle|>": "WAIT",
    "<|user_nonidle|>": None,
}


def map_internal_state_to_eval_label(internal_state: str | None) -> str | None:
    return INTERNAL_STATE_TO_EVAL_LABEL.get(internal_state)


def build_debug_payload(
    internal_state: str | None,
    cascade_text: str = "",
    delta_text: str = "",
) -> dict:
    return {
        "internal_state": internal_state,
        "eval_label_hint": map_internal_state_to_eval_label(internal_state),
        "cascade_text": cascade_text or "",
        "delta_text": delta_text or "",
    }
