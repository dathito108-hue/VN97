from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .p4_arithmetic_mechanism_audit_cli import (
    _verify_artifact,
)
from .p4_generalization_cli import (
    _probe_records,
    _record_score,
)
from .p4_generalization_curriculum import (
    default_validation,
)
from .tokenizer import VN97Tokenizer
from .training import (
    VN97ChatMessage,
    encode_chat_completion_prompt,
)


P4E_M_REPORT_SCHEMA = "VN97P4EM1"
_DIGITS_RE = re.compile(rb"^[0-9]+(?:\n)?$")


@dataclass(frozen=True)
class _Beam:
    tokens: tuple[int, ...]
    payload: bytes
    logprob: float
    ended: bool


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _numeric_token_payloads(
    tokenizer: VN97Tokenizer,
) -> dict[int, bytes]:
    result: dict[int, bytes] = {}
    for token_id in range(
        tokenizer.vocab_size
    ):
        if token_id < 8:
            continue
        payload = tokenizer.decode_bytes(
            [token_id],
            skip_control=True,
        )
        if (
            payload
            and all(
                (
                    48 <= value <= 57
                    or value == 10
                )
                for value in payload
            )
        ):
            result[token_id] = payload
    if not result:
        raise RuntimeError(
            "tokenizer exposes no numeric transport tokens"
        )
    return result


def _valid_numeric_payload(
    payload: bytes,
) -> bool:
    if not payload:
        return False
    if _DIGITS_RE.fullmatch(
        payload
    ) is None:
        return False
    if b"\n" in payload[
        :-1
    ]:
        return False
    return True


def _beam_score(
    beam: _Beam,
) -> float:
    length = max(
        1,
        len(
            beam.tokens
        ),
    )
    return (
        beam.logprob
        / (length ** 0.7)
    )


@torch.inference_mode()
def _numeric_beam_decode(
    *,
    model,
    tokenizer: VN97Tokenizer,
    prompt_tokens: tuple[int, ...],
    beam_width: int,
    max_steps: int,
    token_payloads: dict[int, bytes],
) -> str:
    if beam_width <= 0:
        raise ValueError(
            "beam_width must be positive"
        )
    if max_steps <= 0:
        raise ValueError(
            "max_steps must be positive"
        )

    device = next(
        model.parameters()
    ).device
    model.eval()

    beams = (
        _Beam(
            tokens=(),
            payload=b"",
            logprob=0.0,
            ended=False,
        ),
    )

    allowed_ids = tuple(
        sorted(
            token_payloads
        )
    )

    for _step in range(
        max_steps
    ):
        expanded: list[
            _Beam
        ] = []

        for beam in beams:
            if beam.ended:
                expanded.append(
                    beam
                )
                continue

            input_ids = torch.tensor(
                [
                    [
                        *prompt_tokens,
                        *beam.tokens,
                    ]
                ],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model(
                input_ids
            )
            log_probs = F.log_softmax(
                logits[
                    0,
                    -1,
                ],
                dim=-1,
            )

            if beam.payload:
                expanded.append(
                    _Beam(
                        tokens=beam.tokens,
                        payload=beam.payload,
                        logprob=(
                            beam.logprob
                            + float(
                                log_probs[
                                    tokenizer.eos_id
                                ].item()
                            )
                        ),
                        ended=True,
                    )
                )

            if beam.payload.endswith(
                b"\n"
            ):
                continue

            candidates: list[
                tuple[float, int, bytes]
            ] = []
            for token_id in allowed_ids:
                payload = (
                    beam.payload
                    + token_payloads[
                        token_id
                    ]
                )
                if not _valid_numeric_payload(
                    payload
                ):
                    continue
                candidates.append(
                    (
                        float(
                            log_probs[
                                token_id
                            ].item()
                        ),
                        token_id,
                        payload,
                    )
                )

            candidates.sort(
                reverse=True,
                key=lambda item:
                    item[0],
            )
            for (
                token_logprob,
                token_id,
                payload,
            ) in candidates[
                :beam_width
            ]:
                ended = (
                    payload.endswith(
                        b"\n"
                    )
                )
                expanded.append(
                    _Beam(
                        tokens=(
                            *beam.tokens,
                            token_id,
                        ),
                        payload=payload,
                        logprob=(
                            beam.logprob
                            + token_logprob
                        ),
                        ended=ended,
                    )
                )

        if not expanded:
            raise RuntimeError(
                "numeric beam search produced no candidates"
            )

        expanded.sort(
            key=_beam_score,
            reverse=True,
        )
        beams = tuple(
            expanded[
                :beam_width
            ]
        )

        if all(
            beam.ended
            for beam in beams
        ):
            break

    completed = [
        beam
        for beam in beams
        if beam.payload
    ]
    if not completed:
        raise RuntimeError(
            "numeric beam search produced no output"
        )

    completed.sort(
        key=lambda beam: (
            beam.ended,
            _beam_score(
                beam
            ),
        ),
        reverse=True,
    )
    return completed[
        0
    ].payload.decode(
        "ascii"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit contract-constrained numeric beam decoding on P4E-K "
            "without changing model weights."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--output-json",
        default=None,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )
    if (
        args.max_steps <= 0
        or not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith(
            "cuda"
        )
    ):
        raise RuntimeError(
            "P4E-M requires CUDA and positive max-steps"
        )

    (
        _root,
        checkpoint,
        package,
        parent_report,
    ) = _verify_artifact(
        Path(
            args.artifact_dir
        )
    )

    tokenizer = VN97Tokenizer(
        package
    )
    model = checkpoint.model.to(
        args.device
    )
    model.eval()

    current_engine = (
        TorchVN97InferenceEngine(
            model,
            tokenizer,
            limits=VN97InferenceLimits(
                max_prompt_tokens=4096,
                repetition_penalty=1.12,
                no_repeat_ngram_size=4,
            ),
        )
    )
    raw_engine = (
        TorchVN97InferenceEngine(
            model,
            tokenizer,
            limits=VN97InferenceLimits(
                max_prompt_tokens=4096,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            ),
        )
    )
    token_payloads = (
        _numeric_token_payloads(
            tokenizer
        )
    )

    records = tuple(
        record
        for record in _probe_records(
            default_validation(),
            per_category=30,
        )
        if record.category
        == "reasoning_planning"
    )

    counts = {
        "current_greedy": 0,
        "raw_greedy": 0,
        "beam4": 0,
        "beam8": 0,
        "beam16": 0,
    }
    rows: list[
        dict[str, object]
    ] = []

    for index, record in enumerate(
        records,
        start=1,
    ):
        messages = (
            VN97ChatMessage(
                role="user",
                content=record.prompt,
            ),
        )
        prompt_tokens = (
            encode_chat_completion_prompt(
                tokenizer,
                messages,
            )
        )

        current = (
            current_engine.generate_chat_completion(
                messages,
                max_new_tokens=32,
            )
        )
        raw = (
            raw_engine.generate_chat_completion(
                messages,
                max_new_tokens=32,
            )
        )
        beam_outputs: dict[
            int,
            str,
        ] = {}
        for width in (
            4,
            8,
            16,
        ):
            beam_outputs[
                width
            ] = _numeric_beam_decode(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=
                    prompt_tokens,
                beam_width=width,
                max_steps=
                    args.max_steps,
                token_payloads=
                    token_payloads,
            )

        results = {
            "current_greedy":
                _record_score(
                    record.answer,
                    current,
                ),
            "raw_greedy":
                _record_score(
                    record.answer,
                    raw,
                ),
            "beam4":
                _record_score(
                    record.answer,
                    beam_outputs[4],
                ),
            "beam8":
                _record_score(
                    record.answer,
                    beam_outputs[8],
                ),
            "beam16":
                _record_score(
                    record.answer,
                    beam_outputs[16],
                ),
        }

        for key, passed in (
            results.items()
        ):
            counts[key] += int(
                passed
            )

        rows.append(
            {
                "beam16":
                    beam_outputs[16],
                "beam4":
                    beam_outputs[4],
                "beam8":
                    beam_outputs[8],
                "current_greedy":
                    current,
                "expected":
                    record.answer,
                "index":
                    index,
                "passed":
                    results,
                "prompt":
                    record.prompt,
                "raw_greedy":
                    raw,
            }
        )

    print(
        "VN97 P4E-M SUMMARY "
        f"current={counts['current_greedy']}/30 "
        f"raw={counts['raw_greedy']}/30 "
        f"beam4={counts['beam4']}/30 "
        f"beam8={counts['beam8']}/30 "
        f"beam16={counts['beam16']}/30",
        flush=True,
    )

    rescued = [
        row
        for row in rows
        if (
            not row[
                "passed"
            ][
                "current_greedy"
            ]
            and (
                row[
                    "passed"
                ][
                    "beam4"
                ]
                or row[
                    "passed"
                ][
                    "beam8"
                ]
                or row[
                    "passed"
                ][
                    "beam16"
                ]
            )
        )
    ]
    print(
        "VN97 P4E-M RESCUED "
        f"{len(rescued)}/30",
        flush=True,
    )

    for row in rescued[
        :10
    ]:
        print(
            "\n"
            + "=" * 78,
            flush=True,
        )
        print(
            "P4E-M BEAM-RESCUE",
            flush=True,
        )
        for key in (
            "prompt",
            "expected",
            "current_greedy",
            "raw_greedy",
            "beam4",
            "beam8",
            "beam16",
        ):
            print(
                key.upper()
                + "_JSON="
                + json.dumps(
                    row[key],
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report = {
        "checkpoint_sha256":
            checkpoint.checkpoint_sha256,
        "counts":
            counts,
        "parent_selected_candidate":
            parent_report.get(
                "selected_candidate"
            ),
        "parent_status":
            parent_report.get(
                "status"
            ),
        "rescued":
            len(
                rescued
            ),
        "rows":
            rows,
        "schema":
            P4E_M_REPORT_SCHEMA,
    }

    if args.output_json is not None:
        path = Path(
            args.output_json
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(
            _canonical_json(
                report
            )
            + b"\n"
        )
        print(
            f"VN97 P4E-M REPORT {path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p4e-numeric-beam-audit: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
