from __future__ import annotations


_CHAT_ROLE_MARKERS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
)
_CHAT_ASSISTANT_MARKER = "<|assistant|>"


def recover_chat_response_text(
    raw_text: str,
) -> str:
    """Recover the assistant payload from legacy textual role-marker leakage."""
    if not isinstance(raw_text, str):
        raise TypeError(
            "raw_text must be a string"
        )

    text = raw_text
    boundary = text.rfind(
        _CHAT_ASSISTANT_MARKER
    )
    if boundary >= 0:
        text = text[
            boundary
            + len(
                _CHAT_ASSISTANT_MARKER
            ):
        ]

    cut_positions = [
        position
        for marker in _CHAT_ROLE_MARKERS
        if (
            position := text.find(
                marker
            )
        )
        >= 0
    ]
    if cut_positions:
        text = text[
            :min(cut_positions)
        ]

    return text.strip()
