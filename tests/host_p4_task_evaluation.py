from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from vn97.config import VN97Config
from vn97.deployment_checkpoint import (
    build_deployment_checkpoint,
)
from vn97.model import VN97LanguageCore
from vn97.model_image import build_model_image
from vn97.p3_language_campaign import (
    CANDIDATES,
    P2_BASELINE_RUN_IDENTITY,
    VN97P3CampaignReceipt,
    candidate_ids,
    profile_sha256,
)
from vn97.p4_task_evaluation import (
    P4_CATEGORIES,
    VN97P4EvaluationError,
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
    verify_p3_final_artifact,
)
from vn97.tokenizer import VN97TokenizerPackage


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_p3_final(
    root: Path,
) -> None:
    root.mkdir()
    tokenizer = VN97TokenizerPackage()
    tokenizer_bytes = tokenizer.to_bytes()

    candidate = CANDIDATES[0]
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=int(
                candidate["d_model"]
            ),
            n_layers=int(
                candidate["n_layers"]
            ),
            d_state=int(
                candidate["d_state"]
            ),
            embedding_rank=int(
                candidate[
                    "embedding_rank"
                ]
            ),
        )
    )
    checkpoint_bytes = (
        build_deployment_checkpoint(
            model
        )
    )
    checkpoint_sha = hashlib.sha256(
        checkpoint_bytes
    ).hexdigest()
    tokenizer_sha = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()

    campaign = {
        "schema": "VN97CAMP2",
        "selected_candidate_id":
            candidate_ids()[0],
        "selected_checkpoint_sha256":
            checkpoint_sha,
        "tokenizer_sha256":
            tokenizer_sha,
    }
    campaign_bytes = _canonical(
        campaign
    )

    image = build_model_image(
        model,
        tokenizer=tokenizer,
        tile_rows=16,
        tile_cols=16,
    ).data
    image_sha = hashlib.sha256(
        image
    ).hexdigest()

    receipt = VN97P3CampaignReceipt(
        corpus_manifest_id="1" * 64,
        corpus_manifest_sha256="2" * 64,
        campaign_report_sha256=
            hashlib.sha256(
                campaign_bytes
            ).hexdigest(),
        selected_candidate_id=
            candidate_ids()[0],
        checkpoint_sha256=
            checkpoint_sha,
        tokenizer_sha256=
            tokenizer_sha,
        model_image_sha256=
            image_sha,
        model_image_bytes=len(image),
        validation_mean_loss=5.0,
        validation_top1_accuracy=0.2,
        release_mean_loss=5.1,
        release_top1_accuracy=0.19,
        device="cuda:0",
        profile_sha256=
            profile_sha256(),
    )
    assert (
        receipt.p2_baseline_run_identity
        if hasattr(
            receipt,
            "p2_baseline_run_identity",
        )
        else P2_BASELINE_RUN_IDENTITY
    )

    files = {
        "campaign-report.json":
            campaign_bytes,
        "model.vn97ck1":
            checkpoint_bytes,
        "model.vn97mi1":
            image,
        "p3-run.vn97p3run1.json":
            receipt.to_bytes(),
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    for name, data in files.items():
        (root / name).write_bytes(
            data
        )

    lines = []
    for name in sorted(files):
        digest = hashlib.sha256(
            files[name]
        ).hexdigest()
        lines.append(
            f"{digest}  {name}\n"
        )
    (
        root / "SHA256SUMS"
    ).write_text(
        "".join(lines),
        encoding="ascii",
    )


def _task_object(
    *,
    task_id: str,
    category: str,
    scoring: dict,
) -> dict:
    return {
        "category": category,
        "max_new_tokens": 32,
        "prompt": (
            f"Return the answer for "
            f"{task_id}."
        ),
        "schema": "VN97P4TASK1",
        "scoring": scoring,
        "task_id": task_id,
    }


def _write_suite(
    path: Path,
    *,
    categories=P4_CATEGORIES,
) -> None:
    rows = []
    for index, category in enumerate(
        categories
    ):
        if index % 3 == 0:
            scoring = {
                "expected": "OK",
                "kind": "exact_text",
            }
        elif index % 3 == 1:
            scoring = {
                "case_sensitive": False,
                "expected": [
                    "alpha",
                    "beta",
                ],
                "kind": "contains_all",
            }
        else:
            scoring = {
                "expected": {
                    "answer": index,
                },
                "kind": "json_exact",
            }
        rows.append(
            _task_object(
                task_id=f"task-{index}",
                category=category,
                scoring=scoring,
            )
        )
    path.write_bytes(
        b"".join(
            _canonical(row) + b"\n"
            for row in rows
        )
    )


def test_verify_p3_final_artifact(
    tmp_path: Path,
) -> None:
    root = tmp_path / "p3-final"
    _write_p3_final(root)

    artifact = (
        verify_p3_final_artifact(
            root
        )
    )

    assert (
        artifact.selected_candidate_id
        == candidate_ids()[0]
    )
    assert (
        artifact.checkpoint.config.d_model
        == CANDIDATES[0]["d_model"]
    )
    assert (
        artifact.tokenizer_package.vocab_size
        == artifact.checkpoint.config.vocab_size
    )


def test_verify_p3_final_rejects_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "p3-final"
    _write_p3_final(root)
    (
        root / "model.vn97mi1"
    ).write_bytes(b"tampered")

    with pytest.raises(
        VN97P4EvaluationError,
        match="SHA256SUMS mismatch",
    ):
        verify_p3_final_artifact(
            root
        )


def test_load_p4_task_suite_and_score(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_suite(path)
    suite = load_p4_task_suite(
        path
    )
    assert len(suite.tasks) == 6
    assert len(
        suite.suite_sha256
    ) == 64

    exact = suite.tasks[0]
    assert score_p4_output(
        exact,
        " OK \n",
    )
    assert not score_p4_output(
        exact,
        "NO",
    )

    contains = suite.tasks[1]
    assert score_p4_output(
        contains,
        "BETA then Alpha",
    )

    structured = suite.tasks[2]
    assert score_p4_output(
        structured,
        '{"answer":2}',
    )
    assert not score_p4_output(
        structured,
        '{"answer":3}',
    )


def test_suite_requires_all_categories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_suite(
        path,
        categories=
            P4_CATEGORIES[:-1],
    )
    with pytest.raises(
        VN97P4EvaluationError,
        match="cover every canonical category",
    ):
        load_p4_task_suite(path)


def test_suite_rejects_noncanonical_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = []
    for index, category in enumerate(
        P4_CATEGORIES
    ):
        row = _task_object(
            task_id=f"x-{index}",
            category=category,
            scoring={
                "expected": "OK",
                "kind": "exact_text",
            },
        )
        rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
    path.write_text(
        "".join(rows),
        encoding="utf-8",
    )

    with pytest.raises(
        VN97P4EvaluationError,
        match="canonical JSON",
    ):
        load_p4_task_suite(path)


def test_p4_prompt_matches_p3_chat_training_shape() -> None:
    rendered = render_p4_chat_prompt(
        "Reply with exactly OK"
    )
    assert rendered == (
        "\n<|user|>\n"
        "Reply with exactly OK\n"
        "\n<|assistant|>\n"
    )
