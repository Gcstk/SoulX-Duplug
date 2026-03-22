from __future__ import annotations

import re
import string
from typing import Sequence

from utils.text_utils import split_cn_en


TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
FALLBACK_PUNCTUATION = string.punctuation + "，。！？；：“”‘’（）【】、《》…￥"


def _get_text_normalizers():
    try:
        from utils.MyTn.textnorm import zh_norm as project_zh_norm
        from utils.MyTn.textnorm import zh_remove_punc as project_zh_remove_punc
    except ImportError:
        return _fallback_zh_norm, _fallback_zh_remove_punc
    return project_zh_norm, project_zh_remove_punc


def _fallback_zh_norm(text: str) -> str:
    return text


def _fallback_zh_remove_punc(text: str) -> str:
    for punctuation in FALLBACK_PUNCTUATION:
        text = text.replace(punctuation, "")
    return text


def strip_tags(text: str | None) -> str:
    return TAG_PATTERN.sub("", text or "")


def normalize_text(text: str | None) -> str:
    zh_norm, zh_remove_punc = _get_text_normalizers()
    normalized = strip_tags(text)
    normalized = zh_norm(normalized)
    normalized = zh_remove_punc(normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def char_tokens(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    return [char for char in normalized if not char.isspace()]


def word_tokens(text: str | None) -> list[str]:
    normalized = normalize_text(text)
    return split_cn_en(normalized)


def levenshtein_distance(source: Sequence[str], target: Sequence[str]) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (source_item != target_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    distance = levenshtein_distance(reference, hypothesis)
    denominator = max(len(reference), 1)
    return distance / denominator


def compute_cer(reference: str | None, hypothesis: str | None) -> float:
    return error_rate(char_tokens(reference), char_tokens(hypothesis))


def compute_wer(reference: str | None, hypothesis: str | None) -> float:
    return error_rate(word_tokens(reference), word_tokens(hypothesis))
