from __future__ import annotations

from story_rag.adapters.qa_parser import extract_qa_pairs


def test_extract_qa_pairs_skips_article_preamble() -> None:
    text = (
        "标题：面试记录\n正文：原始采访内容\n\n技术问题 -> 答案\n\n"
        "问题：什么是向量检索？\n答案：通过向量相似度检索。\n\n"
        "问题：什么是 BM25？\n答案：一种词法检索算法。"
    )

    pairs = extract_qa_pairs(text)

    assert [pair.question for pair in pairs] == ["什么是向量检索？", "什么是 BM25？"]
    assert "原始采访内容" not in pairs[0].text
    assert "答案：一种词法检索算法。" in pairs[1].text


def test_extract_qa_pairs_requires_answer_and_accepts_english() -> None:
    pairs = extract_qa_pairs("Q: What is RRF?\nA: Rank fusion.\nQ: Incomplete question")

    assert len(pairs) == 1
    assert pairs[0].question == "What is RRF?"
