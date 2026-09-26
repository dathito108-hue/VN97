from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

from .config import VN97Config
from .model import VN97LanguageCore
from .p3_language_campaign_cli import _validate_corpus
from .p4_compositional_repair_cli import _canonical_measure_records
from .p4_generalization_cli import _probe_records
from .p4_generalization_curriculum import default_validation as p4_validation
from .p5_scaled_foundation_cli import _select_windows, _tensor_batch
from .p5d_behavioral_distillation_cli import (
    _evaluate_fp32,
    _split_teacher_records,
    _teacher_messages,
    _verify_teacher_corpus,
    _windows_from_messages,
)
from .p5d3_relational_distillation import (
    BEHAVIOR_SEQUENCE_LENGTH,
    BEHAVIOR_WEIGHT,
    CANDIDATES,
    CHECKPOINT_INTERVAL_STEPS,
    DYNAMICS_WEIGHT,
    GRAM_WEIGHT,
    MAX_GRAD_NORM,
    NORM_WEIGHT,
    P3_VALIDATION_MAX_WINDOWS,
    P5D3B_FLOAT_ARTIFACT_SCHEMA,
    P5D3B_PROFILE_ID,
    PROGRESS_INTERVAL_STEPS,
    STUDENT_HIDDEN_DEPTHS,
    TEACHER_HIDDEN_DEPTHS,
    TRAIN_STEPS,
    WEIGHT_DECAY,
    profile_sha256,
)
from .p5d3_relational_features import (
    RELATIONAL_SEGMENTS,
    canonical_excerpt,
    excerpt_sha256,
)
from .quantization import set_float_shadow_mode
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import IGNORE_INDEX
from .training_cli import _atomic_write, _load_records


class VN97P5D3BError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distill Falcon3-Mamba relational representation/dynamics targets "
            "into a selected P5D2 309M VN97 float-shadow student."
        )
    )
    parser.add_argument("--p5d1-dir", required=True)
    parser.add_argument("--p5d2-student-dir", required=True)
    parser.add_argument("--p5d3a-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sums(root: Path, expected_names: set[str]) -> None:
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise VN97P5D3BError("missing SHA256SUMS")
    sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise VN97P5D3BError("malformed SHA256SUMS")
        sums[line[66:]] = line[:64]
    if set(sums) != expected_names:
        raise VN97P5D3BError("SHA256SUMS file set mismatch")
    for name, expected in sums.items():
        if _sha256(root / name) != expected:
            raise VN97P5D3BError(f"SHA256 mismatch: {name}")


def _load_selected_student(
    root: Path,
) -> tuple[
    VN97LanguageCore,
    VN97Tokenizer,
    bytes,
    dict[str, object],
    str,
]:
    resolved = root.resolve(strict=True)
    names = {item.name for item in resolved.iterdir()}
    required = {
        "SHA256SUMS",
        "p5d2-report.json",
        "student-float.pt",
        "tokenizer.vn97tk1",
    }
    if names != required:
        raise VN97P5D3BError("P5D2 student artifact file set mismatch")
    _verify_sums(
        resolved,
        {
            "p5d2-report.json",
            "student-float.pt",
            "tokenizer.vn97tk1",
        },
    )

    report = json.loads(
        (resolved / "p5d2-report.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != "VN97P5D2CAND1"
        or report.get("status")
        not in {"BEHAVIORAL_SIGNAL", "WEAK_SIGNAL"}
    ):
        raise VN97P5D3BError("P5D2 student is not an accepted behavioral artifact")

    tokenizer_bytes = (resolved / "tokenizer.vn97tk1").read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    tokenizer = VN97Tokenizer(package)

    payload = torch.load(
        resolved / "student-float.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "VN97P5D2FLOAT1"
        or payload.get("float_shadow_required") is not True
    ):
        raise VN97P5D3BError("invalid P5D2 float student schema")
    if payload.get("tokenizer_sha256") != tokenizer_sha:
        raise VN97P5D3BError("P5D2 tokenizer hash mismatch")
    if int(payload.get("candidate_index", -1)) != int(report["candidate_index"]):
        raise VN97P5D3BError("P5D2 report/artifact candidate mismatch")

    config_raw = payload.get("config")
    if not isinstance(config_raw, dict):
        raise VN97P5D3BError("P5D2 float student config missing")
    config = VN97Config(**config_raw)
    if config.vocab_size != package.vocab_size:
        raise VN97P5D3BError("P5D2 model/tokenizer vocab mismatch")

    model = VN97LanguageCore(config)
    set_float_shadow_mode(model, True)
    model.load_state_dict(payload["state_dict"])

    return (
        model,
        tokenizer,
        tokenizer_bytes,
        report,
        _sha256(resolved / "student-float.pt"),
    )


def _load_relational_targets(
    root: Path,
    *,
    p5d1_report: dict[str, object],
    p5d1_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    resolved = root.resolve(strict=True)
    names = {item.name for item in resolved.iterdir()}
    required = {
        "SHA256SUMS",
        "p5d3a-report.json",
        "relational-targets.jsonl",
    }
    if names != required:
        raise VN97P5D3BError("P5D3A artifact file set mismatch")
    _verify_sums(
        resolved,
        {
            "p5d3a-report.json",
            "relational-targets.jsonl",
        },
    )

    report = json.loads(
        (resolved / "p5d3a-report.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != "VN97P5D3A"
        or int(report.get("records", 0)) != len(p5d1_rows)
    ):
        raise VN97P5D3BError("invalid P5D3A report")
    if report.get("p5d1_corpus_sha256") != p5d1_report.get("corpus_sha256"):
        raise VN97P5D3BError("P5D3A/P5D1 corpus identity mismatch")

    rows_by_id = {
        str(row["record_id"]): row
        for row in p5d1_rows
    }
    targets: list[dict[str, object]] = []
    for line in (resolved / "relational-targets.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise VN97P5D3BError("P5D3A target row must be an object")
        record_id = str(value.get("record_id", ""))
        source = rows_by_id.get(record_id)
        if source is None:
            raise VN97P5D3BError("P5D3A target record not found in P5D1")
        if str(value.get("excerpt_sha256", "")) != excerpt_sha256(source):
            raise VN97P5D3BError("P5D3A excerpt identity mismatch")
        targets.append(value)

    if len(targets) != len(p5d1_rows):
        raise VN97P5D3BError("P5D3A target count mismatch")

    train = sorted(
        [row for row in targets if row.get("split") == "train"],
        key=lambda row: str(row["record_id"]),
    )
    holdout = sorted(
        [row for row in targets if row.get("split") == "holdout"],
        key=lambda row: str(row["record_id"]),
    )
    if len(train) != 500 or len(holdout) != 100:
        raise VN97P5D3BError("unexpected P5D3A train/holdout split")

    return train, holdout, report


def _segment_relations(
    sequence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence.ndim != 2:
        raise VN97P5D3BError("student hidden sequence must be rank-2")
    if int(sequence.shape[0]) < RELATIONAL_SEGMENTS:
        raise VN97P5D3BError("student sequence is too short")

    chunks = torch.tensor_split(
        sequence,
        RELATIONAL_SEGMENTS,
        dim=0,
    )
    if any(int(chunk.shape[0]) == 0 for chunk in chunks):
        raise VN97P5D3BError("empty student relational segment")

    pooled = torch.stack(
        [chunk.mean(dim=0) for chunk in chunks],
        dim=0,
    )
    norms = torch.linalg.vector_norm(
        pooled,
        dim=-1,
    ).clamp_min(1e-8)
    normalized = pooled / norms.unsqueeze(-1)
    gram = normalized @ normalized.transpose(0, 1)
    relative_norms = norms / norms.mean().clamp_min(1e-8)
    return gram, relative_norms


def _student_relations(
    model: VN97LanguageCore,
    token_ids: list[int],
    *,
    device: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if len(token_ids) < RELATIONAL_SEGMENTS:
        raise VN97P5D3BError("student relational input too short")

    inputs = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )
    x = model.embedding(inputs)
    output: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    wanted = set(STUDENT_HIDDEN_DEPTHS)
    for depth, layer in enumerate(model.layers, start=1):
        normalized = layer.norm(x)
        y, _ = layer.core.forward_sequential_reference(
            normalized,
            None,
        )
        x = x + y
        if depth in wanted:
            output[depth] = _segment_relations(x[0])

    if set(output) != wanted:
        raise VN97P5D3BError("student hidden-depth extraction incomplete")
    return output


def _teacher_target_tensors(
    target: dict[str, object],
    *,
    device: str,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    grams_raw = target.get("layer_grams")
    norms_raw = target.get("layer_relative_norms")
    if not isinstance(grams_raw, dict) or not isinstance(norms_raw, dict):
        raise VN97P5D3BError("malformed P5D3A relational target")

    result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for teacher_depth in TEACHER_HIDDEN_DEPTHS:
        key = str(teacher_depth)
        if key not in grams_raw or key not in norms_raw:
            raise VN97P5D3BError("P5D3A target depth missing")
        gram = torch.tensor(
            grams_raw[key],
            dtype=torch.float32,
            device=device,
        )
        norms = torch.tensor(
            norms_raw[key],
            dtype=torch.float32,
            device=device,
        )
        if tuple(gram.shape) != (RELATIONAL_SEGMENTS, RELATIONAL_SEGMENTS):
            raise VN97P5D3BError("bad teacher Gram shape")
        if tuple(norms.shape) != (RELATIONAL_SEGMENTS,):
            raise VN97P5D3BError("bad teacher norm shape")
        result[teacher_depth] = (gram, norms)
    return result


def _relational_loss(
    student: dict[int, tuple[torch.Tensor, torch.Tensor]],
    teacher: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gram_losses: list[torch.Tensor] = []
    norm_losses: list[torch.Tensor] = []
    dynamics_losses: list[torch.Tensor] = []

    paired = list(zip(STUDENT_HIDDEN_DEPTHS, TEACHER_HIDDEN_DEPTHS))
    previous_student: torch.Tensor | None = None
    previous_teacher: torch.Tensor | None = None

    for student_depth, teacher_depth in paired:
        student_gram, student_norms = student[student_depth]
        teacher_gram, teacher_norms = teacher[teacher_depth]
        gram_losses.append(F.mse_loss(student_gram.float(), teacher_gram))
        norm_losses.append(F.mse_loss(student_norms.float(), teacher_norms))

        if previous_student is not None and previous_teacher is not None:
            dynamics_losses.append(
                F.mse_loss(
                    (student_gram.float() - previous_student.float()),
                    (teacher_gram - previous_teacher),
                )
            )
        previous_student = student_gram
        previous_teacher = teacher_gram

    gram_loss = torch.stack(gram_losses).mean()
    norm_loss = torch.stack(norm_losses).mean()
    dynamics_loss = torch.stack(dynamics_losses).mean()
    total = (
        GRAM_WEIGHT * gram_loss
        + DYNAMICS_WEIGHT * dynamics_loss
        + NORM_WEIGHT * norm_loss
    )
    return total, gram_loss, dynamics_loss, norm_loss


def _student_tokens(
    tokenizer: VN97Tokenizer,
    row: dict[str, object],
) -> list[int]:
    excerpt = canonical_excerpt(row)
    tokens = tokenizer.encode(
        excerpt,
        add_bos=True,
        add_eos=True,
        add_text_tag=True,
    )
    if len(tokens) > 512:
        raise VN97P5D3BError(
            f"relational excerpt unexpectedly exceeds 512 VN97 tokens: {len(tokens)}"
        )
    return tokens


@torch.inference_mode()
def _evaluate_relational(
    model: VN97LanguageCore,
    tokenizer: VN97Tokenizer,
    targets: list[dict[str, object]],
    rows_by_id: dict[str, dict[str, object]],
    *,
    device: str,
) -> dict[str, float | int]:
    if not targets:
        raise VN97P5D3BError("relational evaluation set is empty")

    model.eval()
    total_sum = 0.0
    gram_sum = 0.0
    dynamics_sum = 0.0
    norm_sum = 0.0

    for target in targets:
        record_id = str(target["record_id"])
        source = rows_by_id[record_id]
        student = _student_relations(
            model,
            _student_tokens(tokenizer, source),
            device=device,
        )
        teacher = _teacher_target_tensors(target, device=device)
        total, gram, dynamics, norm = _relational_loss(student, teacher)
        total_sum += float(total.item())
        gram_sum += float(gram.item())
        dynamics_sum += float(dynamics.item())
        norm_sum += float(norm.item())

    count = len(targets)
    return {
        "records": count,
        "mean_total_loss": total_sum / count,
        "mean_gram_loss": gram_sum / count,
        "mean_dynamics_loss": dynamics_sum / count,
        "mean_norm_loss": norm_sum / count,
    }


def _safe_torch_save(payload: object, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    with temp.open("wb") as handle:
        torch.save(
            payload,
            handle,
            _use_new_zipfile_serialization=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def _light_resume_state(
    model: VN97LanguageCore,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype=torch.float16)
        state[name] = value
    return state


def _save_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
    next_step: int,
    cumulative: dict[str, float],
) -> int:
    _safe_torch_save(
        {
            "format": "VN97P5D3BLIGHT1",
            "identity": identity,
            "model": _light_resume_state(model),
            "next_step": next_step,
            "optimizer_state": "reset_on_resume",
            "cumulative": cumulative,
        },
        path,
    )
    return path.stat().st_size


def _load_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
) -> tuple[int, dict[str, float]]:
    zero = {
        "behavior": 0.0,
        "dynamics": 0.0,
        "gram": 0.0,
        "norm": 0.0,
        "relational": 0.0,
        "steps": 0.0,
    }
    if not path.exists():
        return 0, zero

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("identity") != identity:
        raise VN97P5D3BError("P5D3B resume identity mismatch")
    if "model" not in payload:
        raise VN97P5D3BError("P5D3B resume model missing")
    model.load_state_dict(payload["model"])
    cumulative = dict(zero)
    raw = payload.get("cumulative", {})
    if isinstance(raw, dict):
        for key in cumulative:
            cumulative[key] = float(raw.get(key, cumulative[key]))
    next_step = int(payload["next_step"])
    print(
        "VN97 P5D3B RESUME "
        f"step={next_step} optimizer_state=reset "
        f"format={payload.get('format', 'unknown')}",
        flush=True,
    )
    return next_step, cumulative


def _status(
    *,
    relation_before: dict[str, float | int],
    relation_after: dict[str, float | int],
    teacher_before: dict[str, float | int],
    teacher_after: dict[str, float | int],
    p3_before: dict[str, float | int],
    p3_after: dict[str, float | int],
) -> str:
    rel_before = float(relation_before["mean_total_loss"])
    rel_after = float(relation_after["mean_total_loss"])
    teacher_ratio = (
        float(teacher_after["mean_loss"])
        / max(float(teacher_before["mean_loss"]), 1e-9)
    )
    p3_ratio = (
        float(p3_after["mean_loss"])
        / max(float(p3_before["mean_loss"]), 1e-9)
    )
    if not all(math.isfinite(value) for value in (rel_after, teacher_ratio, p3_ratio)):
        return "REJECTED_NONFINITE"

    relational_gain = (rel_before - rel_after) / max(rel_before, 1e-9)
    if relational_gain >= 0.08 and teacher_ratio <= 1.05 and p3_ratio <= 1.10:
        return "RELATIONAL_SIGNAL"
    if relational_gain > 0.0 and teacher_ratio <= 1.10 and p3_ratio <= 1.15:
        return "WEAK_RELATIONAL_SIGNAL"
    return "REJECTED_REGRESSION"


def _write_float_artifact(
    *,
    output: Path,
    model: VN97LanguageCore,
    tokenizer_bytes: bytes,
    candidate_index: int,
    candidate,
    report_sha256: str,
    parent_student_sha256: str,
    p5d3a_target_sha256: str,
) -> str:
    path = output / "student-float.pt"
    _safe_torch_save(
        {
            "candidate": candidate.canonical_object(),
            "candidate_index": candidate_index,
            "config": asdict(model.config),
            "float_shadow_required": True,
            "p5d3a_target_sha256": p5d3a_target_sha256,
            "p5d3b_profile_id": P5D3B_PROFILE_ID,
            "parent_student_sha256": parent_student_sha256,
            "report_sha256": report_sha256,
            "schema": P5D3B_FLOAT_ARTIFACT_SCHEMA,
            "state_dict": model.state_dict(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        },
        path,
    )
    return _sha256(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not (0 <= args.candidate_index < len(CANDIDATES)):
        raise VN97P5D3BError("candidate-index is out of range")
    if not torch.cuda.is_available():
        raise VN97P5D3BError("P5D3B requires CUDA")

    candidate = CANDIDATES[args.candidate_index]

    p5d1_rows, p5d1_report = _verify_teacher_corpus(Path(args.p5d1_dir))
    teacher_train_rows, teacher_holdout_rows = _split_teacher_records(p5d1_rows)
    rows_by_id = {
        str(row["record_id"]): row
        for row in p5d1_rows
    }

    (
        model,
        tokenizer,
        tokenizer_bytes,
        parent_report,
        parent_student_sha,
    ) = _load_selected_student(Path(args.p5d2_student_dir))

    relation_train, relation_holdout, p5d3a_report = _load_relational_targets(
        Path(args.p5d3a_dir),
        p5d1_report=p5d1_report,
        p5d1_rows=p5d1_rows,
    )

    relation_train_ids = {str(row["record_id"]) for row in relation_train}
    teacher_train_ids = {str(row["record_id"]) for row in teacher_train_rows}
    if relation_train_ids != teacher_train_ids:
        raise VN97P5D3BError("P5D3A/P5D2 train split mismatch")

    teacher_train_windows = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[_teacher_messages(row) for row in teacher_train_rows],
    )
    teacher_holdout_windows = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[_teacher_messages(row) for row in teacher_holdout_rows],
    )
    if not teacher_train_windows or not teacher_holdout_windows:
        raise VN97P5D3BError("teacher behavioral windows are empty")

    p3_root = Path(args.p3_corpus_dir).resolve(strict=True)
    _validate_corpus(p3_root)
    p3_validation_raw, p3_validation_sha = _load_records(
        [p3_root / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )
    p3_validation_all = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[tuple(messages) for messages in p3_validation_raw],
    )
    p3_validation = _select_windows(
        p3_validation_all,
        min(P3_VALIDATION_MAX_WINDOWS, len(p3_validation_all)),
    )

    set_float_shadow_mode(model, True)
    model.to(args.device)

    relation_before = _evaluate_relational(
        model,
        tokenizer,
        relation_holdout,
        rows_by_id,
        device=args.device,
    )
    teacher_before = _evaluate_fp32(
        model,
        teacher_holdout_windows,
        device=args.device,
    )
    p3_before = _evaluate_fp32(
        model,
        p3_validation,
        device=args.device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    work = Path(args.work_dir)
    output = Path(args.output_dir)
    if work.is_symlink() or output.is_symlink():
        raise VN97P5D3BError("work/output directories must not be symlinks")
    work.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise VN97P5D3BError("output-dir must be new or empty")

    identity_payload = json.dumps(
        {
            "candidate": candidate.canonical_object(),
            "candidate_index": args.candidate_index,
            "p3_validation_sha256": p3_validation_sha,
            "p5d1_corpus_sha256": p5d1_report["corpus_sha256"],
            "p5d3a_target_sha256": p5d3a_report["target_sha256"],
            "parent_student_sha256": parent_student_sha,
            "profile_sha256": profile_sha256(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    identity = hashlib.sha256(b"VN97P5D3BRUN\0" + identity_payload).hexdigest()

    resume_path = work / "state.p5d3b.pt"
    start_step, cumulative = _load_resume(
        resume_path,
        identity=identity,
        model=model,
    )

    relation_order = list(range(len(relation_train)))
    behavior_order = list(range(len(teacher_train_windows)))
    random.Random(candidate.seed).shuffle(relation_order)
    random.Random(candidate.seed ^ 0x3B97).shuffle(behavior_order)

    print(
        "VN97 P5D3B START "
        f"candidate={args.candidate_index} "
        f"candidate_id={candidate.candidate_id} "
        f"parent_p5d2_candidate={parent_report['candidate_index']} "
        f"parameters={sum(int(p.numel()) for p in model.parameters())} "
        f"relation_train={len(relation_train)} "
        f"relation_holdout={len(relation_holdout)} "
        f"steps={TRAIN_STEPS} "
        f"resume_step={start_step} "
        f"lr={candidate.learning_rate} "
        f"device={args.device}",
        flush=True,
    )
    print(
        "VN97 P5D3B BASELINE "
        f"candidate={args.candidate_index} "
        f"relational_loss={float(relation_before['mean_total_loss']):.6f} "
        f"teacher_loss={float(teacher_before['mean_loss']):.6f} "
        f"teacher_top1={float(teacher_before['top1_accuracy']):.6f} "
        f"p3_loss={float(p3_before['mean_loss']):.6f} "
        f"p3_top1={float(p3_before['top1_accuracy']):.6f}",
        flush=True,
    )

    model.train()
    started = time.monotonic()

    for step in range(start_step, TRAIN_STEPS):
        relation_target = relation_train[
            relation_order[step % len(relation_order)]
        ]
        source = rows_by_id[str(relation_target["record_id"])]

        optimizer.zero_grad(set_to_none=True)

        student_relations = _student_relations(
            model,
            _student_tokens(tokenizer, source),
            device=args.device,
        )
        teacher_relations = _teacher_target_tensors(
            relation_target,
            device=args.device,
        )
        (
            relational_total,
            gram_loss,
            dynamics_loss,
            norm_loss,
        ) = _relational_loss(student_relations, teacher_relations)

        if not bool(torch.isfinite(relational_total)):
            raise VN97P5D3BError("relational loss became non-finite")
        relational_total.backward()

        behavior_window = teacher_train_windows[
            behavior_order[step % len(behavior_order)]
        ]
        inputs, labels = _tensor_batch(
            behavior_window,
            device=args.device,
        )
        logits, _ = model.forward_sequential_reference(inputs)
        behavior_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )
        if not bool(torch.isfinite(behavior_loss)):
            raise VN97P5D3BError("behavior loss became non-finite")
        (BEHAVIOR_WEIGHT * behavior_loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5D3BError("gradient norm became non-finite")
        optimizer.step()

        completed = step + 1
        cumulative["relational"] += float(relational_total.detach().item())
        cumulative["gram"] += float(gram_loss.detach().item())
        cumulative["dynamics"] += float(dynamics_loss.detach().item())
        cumulative["norm"] += float(norm_loss.detach().item())
        cumulative["behavior"] += float(behavior_loss.detach().item())
        cumulative["steps"] += 1.0

        del (
            student_relations,
            teacher_relations,
            relational_total,
            gram_loss,
            dynamics_loss,
            norm_loss,
            inputs,
            labels,
            logits,
            behavior_loss,
        )

        if (
            completed % PROGRESS_INTERVAL_STEPS == 0
            or completed == TRAIN_STEPS
        ):
            elapsed = time.monotonic() - started
            local_steps = max(1, completed - start_step)
            eta = (TRAIN_STEPS - completed) * (elapsed / local_steps)
            denom = max(cumulative["steps"], 1.0)
            print(
                "VN97 P5D3B PROGRESS "
                f"candidate={args.candidate_index} "
                f"step={completed}/{TRAIN_STEPS} "
                f"percent={100.0 * completed / TRAIN_STEPS:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"mean_relational={cumulative['relational'] / denom:.6f} "
                f"mean_gram={cumulative['gram'] / denom:.6f} "
                f"mean_dynamics={cumulative['dynamics'] / denom:.6f} "
                f"mean_behavior={cumulative['behavior'] / denom:.6f}",
                flush=True,
            )

        checkpoint_offset = (
            0
            if args.candidate_index == 0
            else CHECKPOINT_INTERVAL_STEPS // 2
        )
        checkpoint_due = (
            completed >= checkpoint_offset
            and (completed - checkpoint_offset) % CHECKPOINT_INTERVAL_STEPS == 0
        )
        if checkpoint_due and completed < TRAIN_STEPS:
            checkpoint_bytes = _save_resume(
                resume_path,
                identity=identity,
                model=model,
                next_step=completed,
                cumulative=cumulative,
            )
            print(
                "VN97 P5D3B CHECKPOINT "
                f"candidate={args.candidate_index} "
                f"step={completed}/{TRAIN_STEPS} "
                f"bytes={checkpoint_bytes} "
                "optimizer_state=reset_on_resume "
                f"path={resume_path}",
                flush=True,
            )

    del optimizer
    torch.cuda.empty_cache()
    model.eval()

    relation_after = _evaluate_relational(
        model,
        tokenizer,
        relation_holdout,
        rows_by_id,
        device=args.device,
    )
    teacher_after = _evaluate_fp32(
        model,
        teacher_holdout_windows,
        device=args.device,
    )
    p3_after = _evaluate_fp32(
        model,
        p3_validation,
        device=args.device,
    )
    probe_after = _canonical_measure_records(
        model=model,
        tokenizer=tokenizer,
        records=_probe_records(p4_validation(), per_category=2),
        device=args.device,
    )

    status = _status(
        relation_before=relation_before,
        relation_after=relation_after,
        teacher_before=teacher_before,
        teacher_after=teacher_after,
        p3_before=p3_before,
        p3_after=p3_after,
    )

    report = {
        "candidate": candidate.canonical_object(),
        "candidate_id": candidate.candidate_id,
        "candidate_index": args.candidate_index,
        "p3": {
            "after": p3_after,
            "before": p3_before,
        },
        "p4_probe_after": probe_after,
        "p5d1": {
            "corpus_sha256": p5d1_report["corpus_sha256"],
            "teacher": p5d1_report["teacher"],
        },
        "p5d2_parent": {
            "candidate_id": parent_report["candidate_id"],
            "candidate_index": parent_report["candidate_index"],
            "student_sha256": parent_student_sha,
        },
        "p5d3a": {
            "target_sha256": p5d3a_report["target_sha256"],
            "profile_sha256": p5d3a_report["profile_sha256"],
        },
        "profile_id": P5D3B_PROFILE_ID,
        "profile_sha256": profile_sha256(),
        "relational_holdout": {
            "after": relation_after,
            "before": relation_before,
        },
        "schema": "VN97P5D3BCAND1",
        "status": status,
        "teacher_holdout": {
            "after": teacher_after,
            "before": teacher_before,
            "records": len(teacher_holdout_rows),
        },
        "training": {
            "behavior_weight": BEHAVIOR_WEIGHT,
            "dynamics_weight": DYNAMICS_WEIGHT,
            "final_cumulative": cumulative,
            "gram_weight": GRAM_WEIGHT,
            "norm_weight": NORM_WEIGHT,
            "precision": "fp32-float-shadow",
            "steps": TRAIN_STEPS,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
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
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    _atomic_write(output / "p5d3b-report.json", report_bytes)

    model.cpu()
    torch.cuda.empty_cache()
    student_sha = _write_float_artifact(
        output=output,
        model=model,
        tokenizer_bytes=tokenizer_bytes,
        candidate_index=args.candidate_index,
        candidate=candidate,
        report_sha256=report_sha,
        parent_student_sha256=parent_student_sha,
        p5d3a_target_sha256=str(p5d3a_report["target_sha256"]),
    )
    _atomic_write(output / "tokenizer.vn97tk1", tokenizer_bytes)

    files = {
        "p5d3b-report.json": output / "p5d3b-report.json",
        "student-float.pt": output / "student-float.pt",
        "tokenizer.vn97tk1": output / "tokenizer.vn97tk1",
    }
    sums = "".join(
        f"{_sha256(path)}  {name}\n"
        for name, path in sorted(files.items())
    ).encode("ascii")
    _atomic_write(output / "SHA256SUMS", sums)
    resume_path.unlink(missing_ok=True)

    rel_before = float(relation_before["mean_total_loss"])
    rel_after = float(relation_after["mean_total_loss"])
    rel_gain = (rel_before - rel_after) / max(rel_before, 1e-9)

    print(
        "VN97P5D3B "
        f"status={status} "
        f"candidate={args.candidate_index} "
        f"candidate_id={candidate.candidate_id} "
        f"rel_before={rel_before:.6f} "
        f"rel_after={rel_after:.6f} "
        f"rel_gain={rel_gain:.6f} "
        f"teacher_before={float(teacher_before['mean_loss']):.6f} "
        f"teacher_after={float(teacher_after['mean_loss']):.6f} "
        f"p3_before={float(p3_before['mean_loss']):.6f} "
        f"p3_after={float(p3_after['mean_loss']):.6f} "
        f"probe_after={int(probe_after['passed'])}/{int(probe_after['task_count'])} "
        f"student_sha256={student_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5d3b-distill: {exc}", file=sys.stderr)
        raise
