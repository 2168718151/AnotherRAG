from __future__ import annotations

import pytest

from story_rag.adapters.text_loader import (
    TextDecodeError,
    build_decoded_text,
    content_hash,
    decode_text_bytes,
    document_id,
)

CHINESE_STORY = "很久以前，山谷里住着一位会讲故事的老人。"
ENGLISH_STORY = "Long ago, an old storyteller lived in the valley."


def test_decode_utf8_without_bom() -> None:
    text, encoding = decode_text_bytes(CHINESE_STORY.encode("utf-8"))
    assert encoding == "utf-8"
    assert text == CHINESE_STORY


def test_decode_utf8_with_bom() -> None:
    text, encoding = decode_text_bytes(b"\xef\xbb\xbf" + ENGLISH_STORY.encode("utf-8"))
    assert encoding == "utf-8-sig"
    assert text == ENGLISH_STORY


def test_decode_gb18030_fallback() -> None:
    text, encoding = decode_text_bytes(CHINESE_STORY.encode("gb18030"))
    assert encoding == "gb18030"
    assert text == CHINESE_STORY


def test_decode_normalizes_line_endings() -> None:
    text, _ = decode_text_bytes(b"line one\r\nline two\rline three")
    assert text == "line one\nline two\nline three"


def test_decode_rejects_empty_and_invalid_bytes() -> None:
    with pytest.raises(TextDecodeError):
        decode_text_bytes(b"")
    with pytest.raises(TextDecodeError):
        decode_text_bytes(b"\xff\xfe\x00\x00invalid")


def test_document_id_is_stable_and_path_case_insensitive() -> None:
    assert document_id(r"Fiction\dragon.txt") == document_id("fiction/dragon.txt")
    assert document_id("a.txt") != document_id("b.txt")


def test_build_decoded_text_uses_file_stem_as_title() -> None:
    decoded = build_decoded_text(
        CHINESE_STORY.encode("utf-8"),
        source_name="stories/dragon.txt",
    )
    assert decoded.title == "dragon"
    assert decoded.content_hash == content_hash(CHINESE_STORY)
    assert decoded.text == CHINESE_STORY
