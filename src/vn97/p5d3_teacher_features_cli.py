from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from .p5d_behavioral_distillation import (
    TEACHER_HOLDOUT_GENERAL_LANGUAGE,
    TEACHER_HOLDOUT_PER_P4_CATEGORY,
)
from .p5d_distillation import TEACHER_REPO
from .p5d3_relational_features import (
    P5D3A_PROFILE_ID,
    P5D3ATarget,
    RELATIONAL_SEGMENTS,
    TEACHER_HIDDEN_STATE_INDICES,
    canonical_excerpt,
    excerpt_sha256,
    profile_object,
    profile_sha256,
    target_digest,
)
from .training_cli import _atomic_write


class VN97P5D3AError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract compact relational hidden-state targets from the exact "
            "Falcon3-Mamba revision used for the sealed P5D1 corpus."
        )
    )
    parser.add_argument(
        "--p5d1-dir",
        required=True,
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
        "--progress-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--accept-teacher-license",
        required=True,
    )
    return parser


def _sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )
            if not chunk:
                break
            digest.update(
                chunk
            )
    return digest.hexdigest()


def _verify_p5d1(
    root: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
]:
    resolved = root.resolve(
        strict=True
    )
    required = {
        "SHA256SUMS",
        "p5d1-report.json",
        "teacher-corpus.jsonl",
    }
    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != required:
        raise VN97P5D3AError(
            "P5D1 artifact file set mismatch"
        )

    sums: dict[str, str] = {}
    for line in (
        resolved
        / "SHA256SUMS"
    ).read_text(
        encoding="ascii"
    ).splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P5D3AError(
                "malformed P5D1 SHA256SUMS"
            )
        sums[
            line[66:]
        ] = line[:64]

    for name, expected in sums.items():
        if _sha256(
            resolved / name
        ) != expected:
            raise VN97P5D3AError(
                f"P5D1 SHA256 mismatch: {name}"
            )

    report = json.loads(
        (
            resolved
            / "p5d1-report.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(
            report,
            dict,
        )
        or report.get(
            "schema"
        ) != "VN97P5D1"
        or report.get(
            "completed_records"
        ) != report.get(
            "total_records"
        )
    ):
        raise VN97P5D3AError(
            "P5D1 report is incomplete"
        )

    rows: list[
        dict[str, object]
    ] = []
    for line in (
        resolved
        / "teacher-corpus.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        value = json.loads(
            line
        )
        if not isinstance(
            value,
            dict,
        ):
            raise VN97P5D3AError(
                "P5D1 corpus row must be an object"
            )
        rows.append(
            value
        )

    if len(rows) != int(
        report[
            "total_records"
        ]
    ):
        raise VN97P5D3AError(
            "P5D1 record count mismatch"
        )

    teacher = report.get(
        "teacher"
    )
    if (
        not isinstance(
            teacher,
            dict,
        )
        or str(
            teacher.get(
                "repo",
                "",
            )
        )
        != TEACHER_REPO
    ):
        raise VN97P5D3AError(
            "P5D1 teacher repo mismatch"
        )

    revision = str(
        teacher.get(
            "revision",
            "",
        )
    )
    if not revision:
        raise VN97P5D3AError(
            "P5D1 teacher revision missing"
        )

    for row in rows:
        if str(
            row.get(
                "teacher_revision",
                "",
            )
        ) != revision:
            raise VN97P5D3AError(
                "teacher revision drift inside P5D1 corpus"
            )

    return rows, report


def _split_ids(
    rows: list[
        dict[str, object]
    ],
) -> dict[str, str]:
    by_category: dict[
        str,
        list[dict[str, object]],
    ] = {}
    for row in rows:
        category = str(
            row[
                "category"
            ]
        )
        by_category.setdefault(
            category,
            [],
        ).append(
            row
        )

    split: dict[
        str,
        str,
    ] = {}
    for (
        category,
        category_rows,
    ) in sorted(
        by_category.items()
    ):
        category_rows.sort(
            key=lambda row:
                str(
                    row[
                        "record_id"
                    ]
                )
        )
        holdout_count = (
            TEACHER_HOLDOUT_GENERAL_LANGUAGE
            if category
            == "general_language"
            else TEACHER_HOLDOUT_PER_P4_CATEGORY
        )
        if len(
            category_rows
        ) <= holdout_count:
            raise VN97P5D3AError(
                f"not enough records for {category}"
            )
        holdout_ids = {
            str(
                row[
                    "record_id"
                ]
            )
            for row in category_rows[
                :holdout_count
            ]
        }
        for row in category_rows:
            record_id = str(
                row[
                    "record_id"
                ]
            )
            split[
                record_id
            ] = (
                "holdout"
                if record_id
                in holdout_ids
                else "train"
            )
    return split


def _load_dependencies():
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except Exception as exc:
        raise VN97P5D3AError(
            "P5D3A requires transformers and accelerate"
        ) from exc
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
    )


def _load_teacher(
    *,
    revision: str,
):
    if torch.cuda.device_count() < 2:
        raise VN97P5D3AError(
            "P5D3A requires two CUDA GPUs"
        )

    (
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = _load_dependencies()

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TEACHER_REPO,
            revision=
                revision,
        )
    )
    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            TEACHER_REPO,
            revision=
                revision,
            torch_dtype=
                torch.float16,
            device_map=
                "balanced",
            max_memory={
                0: "14GiB",
                1: "14GiB",
                "cpu": "20GiB",
            },
            low_cpu_mem_usage=True,
        )
    )
    model.eval()

    print(
        "VN97 P5D3A TEACHER READY "
        f"repo={TEACHER_REPO} "
        f"revision={revision} "
        f"device_map={getattr(model, 'hf_device_map', None)}",
        flush=True,
    )
    return (
        model,
        tokenizer,
    )


def _layer_targets(
    hidden: torch.Tensor,
) -> tuple[
    list[list[float]],
    list[float],
]:
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
    ):
        raise VN97P5D3AError(
            "unexpected teacher hidden-state shape"
        )

    sequence = hidden[
        0
    ].float()
    if (
        int(
            sequence.shape[0]
        )
        < RELATIONAL_SEGMENTS
    ):
        raise VN97P5D3AError(
            "teacher sequence too short for relational segments"
        )

    chunks = torch.tensor_split(
        sequence,
        RELATIONAL_SEGMENTS,
        dim=0,
    )
    if any(
        int(
            chunk.shape[0]
        )
        == 0
        for chunk in chunks
    ):
        raise VN97P5D3AError(
            "empty teacher relational segment"
        )

    pooled = torch.stack(
        [
            chunk.mean(
                dim=0
            )
            for chunk in chunks
        ],
        dim=0,
    )
    norms = torch.linalg.vector_norm(
        pooled,
        dim=-1,
    ).clamp_min(
        1e-8
    )
    normalized = (
        pooled
        / norms.unsqueeze(
            -1
        )
    )
    gram = (
        normalized
        @ normalized.transpose(
            0,
            1,
        )
    )
    relative_norms = (
        norms
        / norms.mean()
    )

    return (
        [
            [
                round(
                    float(value),
                    7,
                )
                for value
                in row
            ]
            for row
            in gram.cpu()
        ],
        [
            round(
                float(value),
                7,
            )
            for value
            in relative_norms.cpu()
        ],
    )


@torch.inference_mode()
def _extract_target(
    *,
    model,
    tokenizer,
    row: dict[str, object],
    split: str,
    revision: str,
) -> P5D3ATarget:
    excerpt = canonical_excerpt(
        row
    )
    inputs = tokenizer(
        excerpt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    first_device = next(
        model.parameters()
    ).device
    inputs = {
        key:
            value.to(
                first_device
            )
        for key, value
        in inputs.items()
    }

    output = model(
        **inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = (
        output.hidden_states
    )
    if hidden_states is None:
        raise VN97P5D3AError(
            "teacher did not return hidden states"
        )
    if len(
        hidden_states
    ) <= max(
        TEACHER_HIDDEN_STATE_INDICES
    ):
        raise VN97P5D3AError(
            "teacher hidden-state tuple is shorter than expected"
        )

    grams: dict[
        str,
        list[list[float]],
    ] = {}
    norms: dict[
        str,
        list[float],
    ] = {}
    for index in (
        TEACHER_HIDDEN_STATE_INDICES
    ):
        gram, relative_norms = (
            _layer_targets(
                hidden_states[
                    index
                ]
            )
        )
        key = str(
            index
        )
        grams[
            key
        ] = gram
        norms[
            key
        ] = relative_norms

    token_count = int(
        inputs[
            "input_ids"
        ].shape[1]
    )

    del output
    del hidden_states
    del inputs

    return P5D3ATarget(
        record_id=str(
            row[
                "record_id"
            ]
        ),
        category=str(
            row[
                "category"
            ]
        ),
        split=split,
        excerpt_sha256=
            excerpt_sha256(
                row
            ),
        teacher_revision=
            revision,
        teacher_token_count=
            token_count,
        layer_grams=
            grams,
        layer_relative_norms=
            norms,
    )


def _record_path(
    root: Path,
    record_id: str,
) -> Path:
    return (
        root
        / "records"
        / f"{record_id}.json"
    )


def _write_target(
    path: Path,
    target: P5D3ATarget,
) -> None:
    payload = (
        json.dumps(
            target.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(
            "utf-8"
        )
        + b"\n"
    )
    _atomic_write(
        path,
        payload,
    )


def _load_target(
    path: Path,
) -> P5D3ATarget:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    return P5D3ATarget(
        record_id=str(
            value[
                "record_id"
            ]
        ),
        category=str(
            value[
                "category"
            ]
        ),
        split=str(
            value[
                "split"
            ]
        ),
        excerpt_sha256=str(
            value[
                "excerpt_sha256"
            ]
        ),
        teacher_revision=str(
            value[
                "teacher_revision"
            ]
        ),
        teacher_token_count=int(
            value[
                "teacher_token_count"
            ]
        ),
        layer_grams={
            str(
                key
            ):
                [
                    [
                        float(
                            item
                        )
                        for item
                        in row
                    ]
                    for row
                    in matrix
                ]
            for key, matrix
            in value[
                "layer_grams"
            ].items()
        },
        layer_relative_norms={
            str(
                key
            ):
                [
                    float(
                        item
                    )
                    for item
                    in values
                ]
            for key, values
            in value[
                "layer_relative_norms"
            ].items()
        },
    )


def _finalize(
    *,
    rows: list[
        dict[str, object]
    ],
    split_by_id: dict[
        str,
        str,
    ],
    work: Path,
    output: Path,
    teacher_revision: str,
    p5d1_report: dict[str, object],
) -> None:
    targets: list[
        P5D3ATarget
    ] = []
    counts = {
        "train": 0,
        "holdout": 0,
    }

    for row in rows:
        record_id = str(
            row[
                "record_id"
            ]
        )
        path = _record_path(
            work,
            record_id,
        )
        if not path.is_file():
            raise VN97P5D3AError(
                "cannot finalize incomplete P5D3A features"
            )
        target = _load_target(
            path
        )
        if (
            target.teacher_revision
            != teacher_revision
        ):
            raise VN97P5D3AError(
                "P5D3A teacher revision drift"
            )
        if (
            target.excerpt_sha256
            != excerpt_sha256(
                row
            )
        ):
            raise VN97P5D3AError(
                "P5D3A excerpt identity drift"
            )
        if (
            target.split
            != split_by_id[
                record_id
            ]
        ):
            raise VN97P5D3AError(
                "P5D3A split drift"
            )
        targets.append(
            target
        )
        counts[
            target.split
        ] += 1

    targets.sort(
        key=lambda item:
            item.record_id
    )
    digest = target_digest(
        targets
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    target_bytes = b"".join(
        (
            json.dumps(
                target.canonical_object(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode(
            "utf-8"
        )
        for target in targets
    )
    _atomic_write(
        output
        / "relational-targets.jsonl",
        target_bytes,
    )

    report = {
        "p5d1_corpus_sha256":
            p5d1_report[
                "corpus_sha256"
            ],
        "profile":
            profile_object(),
        "profile_sha256":
            profile_sha256(),
        "records":
            len(
                targets
            ),
        "schema":
            "VN97P5D3A",
        "split_counts":
            counts,
        "target_sha256":
            digest,
        "teacher": {
            "repo":
                TEACHER_REPO,
            "revision":
                teacher_revision,
        },
    }
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode(
            "utf-8"
        )
        + b"\n"
    )
    _atomic_write(
        output
        / "p5d3a-report.json",
        report_bytes,
    )

    files = {
        "p5d3a-report.json":
            output
            / "p5d3a-report.json",
        "relational-targets.jsonl":
            output
            / "relational-targets.jsonl",
    }
    sums = "".join(
        f"{_sha256(path)}  {name}\n"
        for name, path
        in sorted(
            files.items()
        )
    ).encode(
        "ascii"
    )
    _atomic_write(
        output
        / "SHA256SUMS",
        sums,
    )

    print(
        "VN97P5D3A "
        "status=RELATIONAL_TARGETS_READY "
        f"records={len(targets)} "
        f"train={counts['train']} "
        f"holdout={counts['holdout']} "
        f"target_sha256={digest}",
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
        raise VN97P5D3AError(
            "teacher license acknowledgement must be exactly "
            "TII-FALCON-LLM-2.0"
        )
    if args.progress_interval <= 0:
        raise VN97P5D3AError(
            "progress interval must be positive"
        )

    rows, report = _verify_p5d1(
        Path(
            args.p5d1_dir
        )
    )
    teacher_revision = str(
        report[
            "teacher"
        ][
            "revision"
        ]
    )
    split_by_id = _split_ids(
        rows
    )

    work = Path(
        args.work_dir
    )
    output = Path(
        args.output_dir
    )
    if (
        work.is_symlink()
        or output.is_symlink()
    ):
        raise VN97P5D3AError(
            "work/output directories must not be symlinks"
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

    manifest = {
        "p5d1_corpus_sha256":
            report[
                "corpus_sha256"
            ],
        "profile_sha256":
            profile_sha256(),
        "record_ids":
            sorted(
                str(
                    row[
                        "record_id"
                    ]
                )
                for row in rows
            ),
        "schema":
            "VN97P5D3AMANIFEST1",
        "teacher_revision":
            teacher_revision,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(
            "utf-8"
        )
        + b"\n"
    )
    manifest_path = (
        work
        / "manifest.json"
    )
    if manifest_path.exists():
        if (
            manifest_path.read_bytes()
            != manifest_bytes
        ):
            raise VN97P5D3AError(
                "P5D3A resume manifest mismatch"
            )
    else:
        _atomic_write(
            manifest_path,
            manifest_bytes,
        )

    completed = sum(
        1
        for row in rows
        if _record_path(
            work,
            str(
                row[
                    "record_id"
                ]
            ),
        ).is_file()
    )
    print(
        "VN97 P5D3A MANIFEST "
        f"records={len(rows)} "
        f"resume_records={completed} "
        f"profile={P5D3A_PROFILE_ID}",
        flush=True,
    )

    model, tokenizer = _load_teacher(
        revision=
            teacher_revision
    )

    started = time.monotonic()
    for row in rows:
        record_id = str(
            row[
                "record_id"
            ]
        )
        path = _record_path(
            work,
            record_id,
        )
        if path.is_file():
            existing = _load_target(
                path
            )
            if (
                existing.teacher_revision
                != teacher_revision
                or existing.excerpt_sha256
                != excerpt_sha256(
                    row
                )
            ):
                raise VN97P5D3AError(
                    "P5D3A existing record identity mismatch"
                )
            continue

        target = _extract_target(
            model=model,
            tokenizer=tokenizer,
            row=row,
            split=
                split_by_id[
                    record_id
                ],
            revision=
                teacher_revision,
        )
        _write_target(
            path,
            target,
        )
        completed += 1

        if (
            completed
            % args.progress_interval
            == 0
            or completed
            == len(
                rows
            )
        ):
            elapsed = (
                time.monotonic()
                - started
            )
            rate = completed / max(
                elapsed,
                1e-9,
            )
            eta = (
                len(
                    rows
                )
                - completed
            ) / max(
                rate,
                1e-9,
            )
            print(
                "VN97 P5D3A PROGRESS "
                f"records={completed}/{len(rows)} "
                f"percent={100.0 * completed / len(rows):.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"category={target.category} "
                f"split={target.split}",
                flush=True,
            )

    del model
    del tokenizer
    torch.cuda.empty_cache()

    _finalize(
        rows=rows,
        split_by_id=
            split_by_id,
        work=work,
        output=output,
        teacher_revision=
            teacher_revision,
        p5d1_report=report,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p5d3a-teacher-features: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
