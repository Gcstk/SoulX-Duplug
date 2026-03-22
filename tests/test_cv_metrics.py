from cv.metrics import compute_cer, compute_wer, normalize_text, strip_tags


def test_strip_tags_and_normalize_text():
    assert strip_tags("你好<COMPLETE>") == "你好"
    assert normalize_text("Hello, 世界！<COMPLETE>") == "Hello 世界"


def test_compute_cer_and_wer():
    reference = "你们老师也是绝了"
    hypothesis = "你们老师也是真绝了"

    cer = compute_cer(reference, hypothesis)
    wer = compute_wer(reference, hypothesis)

    assert cer > 0
    assert wer > 0
    assert compute_cer(reference, reference) == 0
    assert compute_wer(reference, reference) == 0
