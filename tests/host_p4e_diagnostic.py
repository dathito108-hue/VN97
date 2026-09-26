from __future__ import annotations

import json

from vn97.p4_artifact import (
    _P4D_HASHED_FILES,
    _parse_p4d_sums,
)
from vn97.p4_generation_diagnostic_cli import (
    _classify_failure,
    _select_diagnostic_records,
)
from vn97.p4_generalization_curriculum import (
    P4D_CATEGORIES,
)


def test_p4e_selection_is_balanced_and_deterministic() -> None:
    first = _select_diagnostic_records(
        per_category=3,
    )
    second = _select_diagnostic_records(
        per_category=3,
    )
    assert first == second
    assert len(first) == (
        len(P4D_CATEGORIES)
        * 3
    )

    counts = {
        category: 0
        for category in P4D_CATEGORIES
    }
    for item in first:
        counts[item.category] += 1
    assert counts == {
        category: 3
        for category in P4D_CATEGORIES
    }


def test_p4e_failure_classification_surfaces_common_modes() -> None:
    assert (
        _classify_failure(
            "answer",
            "answer",
        )
        == "pass"
    )
    assert (
        _classify_failure(
            "answer",
            "",
        )
        == "empty_output"
    )
    assert (
        _classify_failure(
            "answer",
            "<|assistant|> answer",
        )
        == "role_marker_leakage"
    )
    assert (
        _classify_failure(
            "answer",
            "prefix answer suffix",
        )
        == "extra_text"
    )
    assert (
        _classify_failure(
            '{"allowed":false}',
            '{"allowed":true}',
        )
        == "structured_json_value_mismatch"
    )
    assert (
        _classify_failure(
            '{"allowed":false}',
            'result={"allowed":false}',
        )
        == "structured_extra_text"
    )


def test_p4d_sha256sum_parser_accepts_exact_file_set() -> None:
    rows = []
    for index, name in enumerate(
        sorted(_P4D_HASHED_FILES),
        start=1,
    ):
        digest = (
            f"{index:064x}"
        )
        rows.append(
            f"{digest}  {name}\n"
        )
    parsed = _parse_p4d_sums(
        "".join(rows).encode("ascii")
    )
    assert set(parsed) == _P4D_HASHED_FILES
    assert all(
        len(value) == 64
        for value in parsed.values()
    )


def test_p4e_json_expected_is_canonicalizable() -> None:
    expected = {
        "tool": "clipboard.write",
        "allowed": False,
    }
    canonical = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert canonical == (
        '{"allowed":false,'
        '"tool":"clipboard.write"}'
    )
