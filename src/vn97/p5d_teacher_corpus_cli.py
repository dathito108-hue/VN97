from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import torch

from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    default_training as p4_training,
)
from .p5_scaled_foundation_cli import _held_out_dev
from .p5d_distillation import TEACHER_REPO
from .p5d_teacher_corpus import (
    DEFAULT_SYSTEM_PROMPT,
    P3_RECORDS,
    P4_PER_CATEGORY,
    P5D1Prompt,
    P5D1TeacherRecord,
    generation_limit_for_category,
    prompt_digest,
    report_object,
    stable_record_id,
    teacher_corpus_digest,
    teacher_matches_reference,
)
from .training import VN97ChatMessage
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
)


class VN97P5D1Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a sealed teacher-output corpus from the frozen "
            "Falcon3-Mamba-7B-Instruct teacher for later VN97 distillation."
        )
    )
    parser.add_argument(
        "--p3-corpus-dir",
        required=True,
    )
    parser.add_argument(
        "--dev-suite",
        default=None,
    )
    parser.add_argument(
        "--work-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--teacher-revision",
        default="main",
    )
    parser.add_argument(
        "--p3-records",
        type=int,
        default=P3_RECORDS,
    )
    parser.add_argument(
        "--p4-per-category",
        type=int,
        default=P4_PER_CATEGORY,
    )
    parser.add_argument(
        "--accept-teacher-license",
        required=True,
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
    )
    return parser


def _hash_messages(
    messages: tuple[
        VN97ChatMessage,
        ...
    ],
) -> bytes:
    payload = _canonical_json(
        {
            "messages": [
                {
                    "role": item.role,
                    "content": item.content,
                }
                for item in messages
            ]
        }
    )
    return hashlib.sha256(
        payload
    ).digest()


def _with_system(
    messages: tuple[
        VN97ChatMessage,
        ...
    ],
) -> tuple[
    VN97ChatMessage,
    ...
]:
    if (
        messages
        and messages[0].role
        == "system"
    ):
        return messages
    return (
        VN97ChatMessage(
            role="system",
            content=
                DEFAULT_SYSTEM_PROMPT,
        ),
        *messages,
    )


def _p4_prompts(
    *,
    per_category: int,
    held_out_prompts: set[str],
) -> list[P5D1Prompt]:
    if per_category <= 0:
        raise VN97P5D1Error(
            "p4-per-category must be positive"
        )

    output: list[
        P5D1Prompt
    ] = []
    records = p4_training()

    for category in P4D_CATEGORIES:
        rows = [
            row
            for row in records
            if (
                row.category
                == category
                and row.prompt
                not in held_out_prompts
            )
        ]
        rows.sort(
            key=lambda row:
                hashlib.sha256(
                    (
                        row.category
                        + "\0"
                        + row.prompt
                    ).encode(
                        "utf-8"
                    )
                ).digest()
        )
        if len(rows) < per_category:
            raise VN97P5D1Error(
                f"not enough P4 prompts for {category}"
            )

        for row in rows[
            :per_category
        ]:
            messages = _with_system(
                (
                    VN97ChatMessage(
                        role="user",
                        content=row.prompt,
                    ),
                )
            )
            output.append(
                P5D1Prompt(
                    record_id=
                        stable_record_id(
                            category=
                                category,
                            source=
                                "p4_training",
                            messages=
                                messages,
                        ),
                    category=
                        category,
                    source=
                        "p4_training",
                    messages=
                        messages,
                    reference_response=
                        row.answer,
                    max_new_tokens=
                        generation_limit_for_category(
                            category
                        ),
                )
            )

    return output


def _p3_prompts(
    *,
    corpus_dir: Path,
    count: int,
    held_out_prompts: set[str],
) -> tuple[
    list[P5D1Prompt],
    str,
]:
    if count <= 0:
        raise VN97P5D1Error(
            "p3-records must be positive"
        )

    records, source_sha256 = (
        _load_records(
            [
                corpus_dir
                / "training.jsonl"
            ],
            mode="chat",
            max_input_bytes=
                64 * 1024 * 1024,
            max_examples=100_000,
        )
    )

    eligible: list[
        tuple[
            VN97ChatMessage,
            ...,
        ]
    ] = []
    for raw in records:
        messages = tuple(raw)
        if (
            len(messages) < 2
            or messages[-1].role
            != "assistant"
        ):
            continue

        prefix = messages[:-1]
        if (
            not prefix
            or not any(
                message.role
                == "user"
                for message in prefix
            )
        ):
            continue

        if any(
            message.content
            in held_out_prompts
            for message in prefix
            if message.role
            == "user"
        ):
            continue

        eligible.append(
            messages
        )

    eligible.sort(
        key=_hash_messages
    )
    if len(eligible) < count:
        raise VN97P5D1Error(
            "not enough eligible P3 teacher prompts"
        )

    output: list[
        P5D1Prompt
    ] = []
    for messages in eligible[
        :count
    ]:
        reference = (
            messages[-1].content
        )
        prefix = _with_system(
            messages[:-1]
        )
        output.append(
            P5D1Prompt(
                record_id=
                    stable_record_id(
                        category=
                            "general_language",
                        source=
                            "p3_training",
                        messages=
                            prefix,
                    ),
                category=
                    "general_language",
                source=
                    "p3_training",
                messages=
                    prefix,
                reference_response=
                    reference,
                max_new_tokens=
                    generation_limit_for_category(
                        "general_language"
                    ),
            )
        )

    return (
        output,
        source_sha256,
    )


def _build_manifest(
    *,
    p3_root: Path,
    p3_count: int,
    p4_per_category: int,
    held_out_prompts: set[str],
) -> tuple[
    list[P5D1Prompt],
    str,
]:
    p4 = _p4_prompts(
        per_category=
            p4_per_category,
        held_out_prompts=
            held_out_prompts,
    )
    p3, p3_source_sha = (
        _p3_prompts(
            corpus_dir=p3_root,
            count=p3_count,
            held_out_prompts=
                held_out_prompts,
        )
    )
    prompts = [
        *p4,
        *p3,
    ]
    prompts.sort(
        key=lambda item:
            item.record_id
    )

    ids = [
        prompt.record_id
        for prompt in prompts
    ]
    if len(ids) != len(set(ids)):
        raise VN97P5D1Error(
            "duplicate P5D1 record id"
        )

    return (
        prompts,
        p3_source_sha,
    )


def _manifest_bytes(
    prompts: Iterable[
        P5D1Prompt
    ],
    *,
    p3_source_sha256: str,
) -> bytes:
    rows = [
        prompt.canonical_object()
        for prompt in prompts
    ]
    payload = {
        "p3_source_sha256":
            p3_source_sha256,
        "prompts":
            rows,
        "schema":
            "VN97P5D1MANIFEST1",
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _load_dependencies():
    try:
        from huggingface_hub import (
            model_info,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except Exception as exc:
        raise VN97P5D1Error(
            "P5D1 requires transformers, accelerate and huggingface_hub"
        ) from exc

    return (
        model_info,
        AutoModelForCausalLM,
        AutoTokenizer,
    )


def _load_teacher(
    *,
    revision: str,
):
    if torch.cuda.device_count() < 2:
        raise VN97P5D1Error(
            "P5D1 teacher generation requires two CUDA GPUs"
        )

    (
        model_info,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _load_dependencies()

    info = model_info(
        TEACHER_REPO,
        revision=revision,
    )
    resolved_revision = str(
        info.sha
    )

    print(
        "VN97 P5D1 TEACHER RESOLVED "
        f"repo={TEACHER_REPO} "
        f"revision={resolved_revision}",
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TEACHER_REPO,
            revision=
                resolved_revision,
        )
    )
    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    max_memory = {
        0: "14GiB",
        1: "14GiB",
        "cpu": "20GiB",
    }
    model = (
        AutoModelForCausalLM
        .from_pretrained(
            TEACHER_REPO,
            revision=
                resolved_revision,
            torch_dtype=
                torch.float16,
            device_map=
                "balanced",
            max_memory=
                max_memory,
            low_cpu_mem_usage=True,
        )
    )
    model.eval()

    print(
        "VN97 P5D1 TEACHER READY "
        f"device_map={getattr(model, 'hf_device_map', None)}",
        flush=True,
    )

    return (
        model,
        tokenizer,
        resolved_revision,
    )


def _teacher_generate(
    *,
    model,
    tokenizer,
    prompt: P5D1Prompt,
) -> str:
    rendered = (
        tokenizer.apply_chat_template(
            [
                {
                    "role":
                        message.role,
                    "content":
                        message.content,
                }
                for message
                in prompt.messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    first_device = (
        next(
            model.parameters()
        ).device
    )
    inputs = {
        key:
            value.to(
                first_device
            )
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=
                prompt.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=
                tokenizer.pad_token_id,
            eos_token_id=
                tokenizer.eos_token_id,
        )

    input_length = int(
        inputs[
            "input_ids"
        ].shape[1]
    )
    generated = output[
        0,
        input_length:
    ]
    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()
    if not text:
        raise VN97P5D1Error(
            f"teacher produced empty output for {prompt.record_id}"
        )
    return text


def _record_path(
    root: Path,
    record_id: str,
) -> Path:
    return (
        root
        / "records"
        / f"{record_id}.json"
    )


def _write_record(
    path: Path,
    record: P5D1TeacherRecord,
) -> None:
    payload = (
        json.dumps(
            record.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(
        path,
        payload,
    )


def _load_record(
    path: Path,
) -> P5D1TeacherRecord:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    messages = tuple(
        VN97ChatMessage(
            role=item["role"],
            content=item["content"],
        )
        for item in value[
            "messages"
        ]
    )
    prompt = P5D1Prompt(
        record_id=
            value["record_id"],
        category=
            value["category"],
        source=
            value["source"],
        messages=
            messages,
        reference_response=
            value.get(
                "reference_response"
            ),
        max_new_tokens=int(
            value["max_new_tokens"]
        ),
    )
    return P5D1TeacherRecord(
        prompt=prompt,
        teacher_response=
            value[
                "teacher_response"
            ],
        teacher_revision=
            value[
                "teacher_revision"
            ],
    )


def _finalize(
    *,
    prompts: list[P5D1Prompt],
    work: Path,
    output: Path,
    teacher_revision: str,
    manifest_sha256: str,
) -> None:
    records: list[
        P5D1TeacherRecord
    ] = []
    category_counts: dict[
        str,
        int,
    ] = {}
    exact_counts: dict[
        str,
        int,
    ] = {}

    for prompt in prompts:
        path = _record_path(
            work,
            prompt.record_id,
        )
        if not path.is_file():
            raise VN97P5D1Error(
                "cannot finalize incomplete P5D1 corpus"
            )
        record = _load_record(
            path
        )
        if (
            record.teacher_revision
            != teacher_revision
        ):
            raise VN97P5D1Error(
                "teacher revision drift in P5D1 records"
            )
        records.append(
            record
        )
        category_counts[
            prompt.category
        ] = (
            category_counts.get(
                prompt.category,
                0,
            )
            + 1
        )
        matched = (
            teacher_matches_reference(
                category=
                    prompt.category,
                teacher_response=
                    record.teacher_response,
                reference_response=
                    prompt.reference_response,
            )
        )
        if matched is True:
            exact_counts[
                prompt.category
            ] = (
                exact_counts.get(
                    prompt.category,
                    0,
                )
                + 1
            )

    corpus_sha = (
        teacher_corpus_digest(
            records
        )
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_bytes = b"".join(
        (
            json.dumps(
                record.canonical_object(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )
    _atomic_write(
        output
        / "teacher-corpus.jsonl",
        corpus_bytes,
    )

    report = report_object(
        teacher_revision=
            teacher_revision,
        manifest_sha256=
            manifest_sha256,
        corpus_sha256=
            corpus_sha,
        total_records=
            len(prompts),
        completed_records=
            len(records),
        category_counts=
            category_counts,
        exact_reference_counts=
            exact_counts,
    )
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(
        output
        / "p5d1-report.json",
        report_bytes,
    )

    files = {
        "p5d1-report.json":
            report_bytes,
        "teacher-corpus.jsonl":
            corpus_bytes,
    }
    sums = b"".join(
        (
            hashlib.sha256(
                files[name]
            ).hexdigest()
            + "  "
            + name
            + "\n"
        ).encode("ascii")
        for name in sorted(
            files
        )
    )
    _atomic_write(
        output
        / "SHA256SUMS",
        sums,
    )

    print(
        "VN97P5D1 "
        "status=TEACHER_CORPUS_READY "
        f"records={len(records)} "
        f"teacher_revision={teacher_revision} "
        f"corpus_sha256={corpus_sha}",
        flush=True,
    )


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    if (
        args.accept_teacher_license
        != "TII-FALCON-LLM-2.0"
    ):
        raise VN97P5D1Error(
            "teacher license acknowledgement must be exactly "
            "TII-FALCON-LLM-2.0"
        )
    if args.progress_interval <= 0:
        raise VN97P5D1Error(
            "progress interval must be positive"
        )

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(
        strict=True
    )
    suite = (
        Path(
            args.dev_suite
        )
        if args.dev_suite
        is not None
        else None
    )
    held_out_prompts, _ = (
        _held_out_dev(
            suite
        )
    )

    prompts, p3_source_sha = (
        _build_manifest(
            p3_root=p3_root,
            p3_count=
                args.p3_records,
            p4_per_category=
                args.p4_per_category,
            held_out_prompts=
                held_out_prompts,
        )
    )
    manifest_sha = prompt_digest(
        prompts
    )
    manifest_data = _manifest_bytes(
        prompts,
        p3_source_sha256=
            p3_source_sha,
    )

    work = Path(
        args.work_dir
    )
    output = Path(
        args.output_dir
    )
    if work.is_symlink():
        raise VN97P5D1Error(
            "work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        work
        / "records"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        work
        / "manifest.json"
    )
    if manifest_path.exists():
        if (
            manifest_path.read_bytes()
            != manifest_data
        ):
            raise VN97P5D1Error(
                "P5D1 resume manifest mismatch"
            )
    else:
        _atomic_write(
            manifest_path,
            manifest_data,
        )

    existing = sum(
        1
        for prompt in prompts
        if _record_path(
            work,
            prompt.record_id,
        ).is_file()
    )
    print(
        "VN97 P5D1 MANIFEST "
        f"records={len(prompts)} "
        f"resume_records={existing} "
        f"manifest_sha256={manifest_sha}",
        flush=True,
    )

    model = None
    tokenizer = None
    started = time.monotonic()
    (
        model,
        tokenizer,
        resolved_revision,
    ) = _load_teacher(
        revision=
            args.teacher_revision,
    )

    completed = 0
    for index, prompt in enumerate(
        prompts,
        start=1,
    ):
        path = _record_path(
            work,
            prompt.record_id,
        )
        if path.is_file():
            record = _load_record(
                path
            )
            if (
                record.teacher_revision
                != resolved_revision
            ):
                raise VN97P5D1Error(
                    "resume teacher revision mismatch"
                )
            completed += 1
            continue

        response = _teacher_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
        )
        record = P5D1TeacherRecord(
            prompt=prompt,
            teacher_response=
                response,
            teacher_revision=
                resolved_revision,
        )
        _write_record(
            path,
            record,
        )
        completed += 1

        if (
            completed
            % args.progress_interval
            == 0
            or completed
            == len(prompts)
        ):
            elapsed = (
                time.monotonic()
                - started
            )
            rate = (
                completed
                / max(
                    elapsed,
                    1e-9,
                )
            )
            remaining = (
                len(prompts)
                - completed
            )
            eta = (
                remaining
                / max(
                    rate,
                    1e-9,
                )
            )
            print(
                "VN97 P5D1 PROGRESS "
                f"records={completed}/{len(prompts)} "
                f"percent={100.0 * completed / len(prompts):.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"category={prompt.category}",
                flush=True,
            )

    del model
    del tokenizer
    torch.cuda.empty_cache()

    _finalize(
        prompts=prompts,
        work=work,
        output=output,
        teacher_revision=
            resolved_revision,
        manifest_sha256=
            manifest_sha,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p5d1-teacher-corpus: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
