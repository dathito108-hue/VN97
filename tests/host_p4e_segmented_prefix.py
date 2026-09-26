from __future__ import annotations

from vn97.tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from vn97.training import (
    VN97ChatMessage,
    encode_chat_completion_messages,
    encode_chat_completion_prompt,
)


def test_segmented_inference_prefix_matches_training_prefix() -> None:
    package = VN97TokenizerPackage(
        learned_tokens=(
            b"hello\n\n<|assistant|>",
            b"<|user|>",
            b"<|assistant|>",
        )
    )
    tokenizer = VN97Tokenizer(package)
    user = VN97ChatMessage(
        role="user",
        content="hello",
    )
    assistant = VN97ChatMessage(
        role="assistant",
        content="world",
    )
    example = encode_chat_completion_messages(
        tokenizer,
        (user, assistant),
    )
    first_target = next(
        index
        for index, enabled
        in enumerate(example.target_mask)
        if enabled
    )
    training_prefix = (
        example.token_ids[:first_target]
    )
    canonical_prefix = (
        encode_chat_completion_prompt(
            tokenizer,
            (user,),
        )
    )
    assert canonical_prefix == training_prefix


def test_whole_string_tokenization_can_diverge_at_chat_boundary() -> None:
    package = VN97TokenizerPackage(
        learned_tokens=(
            b"hello\n\n<|assistant|>",
            b"<|user|>",
            b"<|assistant|>",
        )
    )
    tokenizer = VN97Tokenizer(package)
    user = VN97ChatMessage(
        role="user",
        content="hello",
    )
    canonical = encode_chat_completion_prompt(
        tokenizer,
        (user,),
    )
    rendered = (
        "\n<|user|>\nhello\n"
        "\n<|assistant|>\n"
    )
    whole = tuple(
        tokenizer.encode(
            rendered,
            add_bos=True,
            add_text_tag=True,
        )
    )
    assert whole != canonical
