from __future__ import annotations

from vn97.cognition_adapter import (
    recover_chat_response_text,
)


def test_boundary_recovery_keeps_plain_response() -> None:
    assert (
        recover_chat_response_text(
            "  answer  "
        )
        == "answer"
    )


def test_boundary_recovery_uses_last_assistant_marker() -> None:
    raw = (
        "legacy noise\n"
        "<|assistant|>\n"
        "wrong\n"
        "<|assistant|>\n"
        '{"capability":"web.fetch","requires_approval":false}\n'
    )
    assert (
        recover_chat_response_text(raw)
        == '{"capability":"web.fetch","requires_approval":false}'
    )


def test_boundary_recovery_truncates_following_role() -> None:
    raw = (
        "noise\n"
        "<|assistant|>\n"
        "Lan\n"
        "<|user|>\n"
        "next"
    )
    assert (
        recover_chat_response_text(raw)
        == "Lan"
    )


def test_boundary_recovery_does_not_invent_payload() -> None:
    assert (
        recover_chat_response_text(
            "<|assistant|>"
        )
        == ""
    )
