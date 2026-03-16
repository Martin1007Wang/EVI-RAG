from __future__ import annotations

from src.llm.backends import (
    _build_openai_headers,
    _format_subgraphrag_user_content,
    _join_url,
    _parse_subgraphrag_user_content,
    _trim_messages_to_vllm_budget,
)


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del add_generation_prompt
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if tokenize:
            return list(range(len(rendered)))
        return rendered

    def encode(self, rendered: str):
        return list(range(len(rendered)))


def test_join_url_normalizes_slashes() -> None:
    assert _join_url("https://api.openai.com/v1/", "chat/completions") == (
        "https://api.openai.com/v1/chat/completions"
    )


def test_build_openai_headers_merges_authorization_and_extra_headers() -> None:
    headers = _build_openai_headers(
        "secret-token",
        extra_headers={"X-Test": 123, "X-Skip": None},
    )

    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
        "X-Test": "123",
    }


def test_parse_subgraphrag_user_content_preserves_group_separators() -> None:
    content = _format_subgraphrag_user_content(
        ["(A,r,B)", "", "(B,r,C)"],
        "Where was X born?",
    )

    triplets, question = _parse_subgraphrag_user_content(content) or (None, None)

    assert triplets == ["(A,r,B)", "", "(B,r,C)"]
    assert question == "Where was X born?"


def test_trim_messages_to_vllm_budget_drops_late_triplet_groups() -> None:
    tokenizer = _DummyTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": _format_subgraphrag_user_content(
                ["(A,r,B)", "", "(B,r,C)"],
                "Where was X born?",
            ),
        },
    ]
    budget_messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": _format_subgraphrag_user_content(
                ["(A,r,B)"],
                "Where was X born?",
            ),
        },
    ]
    budget = len(tokenizer.apply_chat_template(budget_messages, tokenize=False))

    trimmed = _trim_messages_to_vllm_budget(
        messages,
        tokenizer=tokenizer,
        budget=budget,
    )

    user_content = trimmed[1]["content"]
    assert "(A,r,B)" in user_content
    assert "(B,r,C)" not in user_content


def test_trim_messages_to_vllm_budget_falls_back_to_suffix_for_plain_user_text() -> (
    None
):
    tokenizer = _DummyTokenizer()
    messages = [{"role": "user", "content": "abcdefghijklmnopqrstuvwxyz"}]
    budget_messages = [{"role": "user", "content": "uvwxyz"}]
    budget = len(tokenizer.apply_chat_template(budget_messages, tokenize=False))

    trimmed = _trim_messages_to_vllm_budget(
        messages,
        tokenizer=tokenizer,
        budget=budget,
    )

    assert trimmed[0]["content"] == "uvwxyz"
