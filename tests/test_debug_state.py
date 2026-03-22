from service.debug_state import build_debug_payload, map_internal_state_to_eval_label


def test_map_internal_state_to_eval_label():
    assert map_internal_state_to_eval_label("<|user_complete|>") == "COMPLETE"
    assert map_internal_state_to_eval_label("<|user_incomplete|>") == "INCOMPLETE"
    assert map_internal_state_to_eval_label("<|user_backchannel|>") == "BACKCHANNEL"
    assert map_internal_state_to_eval_label("<|user_idle|>") == "WAIT"
    assert map_internal_state_to_eval_label("<|user_nonidle|>") is None


def test_build_debug_payload():
    payload = build_debug_payload(
        "<|user_complete|>",
        cascade_text="你好世界",
        delta_text="世界",
    )

    assert payload == {
        "internal_state": "<|user_complete|>",
        "eval_label_hint": "COMPLETE",
        "cascade_text": "你好世界",
        "delta_text": "世界",
    }
