from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

import torch

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
)
from .p4_generalization_cli import (
    _probe_records,
    _record_score,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    default_validation,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from .training import (
    VN97ChatMessage,
    encode_chat_completion_messages,
    encode_chat_completion_prompt,
)


def _verify_sums(root: Path) -> None:
    sums_path = root / "SHA256SUMS"
    text = sums_path.read_text(
        encoding="ascii"
    )
    if not text.endswith("\n"):
        raise RuntimeError(
            "SHA256SUMS must end with newline"
        )
    for line in text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise RuntimeError(
                "malformed SHA256SUMS"
            )
        digest = line[:64]
        name = line[66:]
        path = root / name
        if not path.is_file():
            raise RuntimeError(
                f"artifact file missing: {name}"
            )
        actual = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        if actual != digest:
            raise RuntimeError(
                f"SHA256 mismatch: {name}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure legacy whole-string versus canonical segmented chat "
            "prefix inference on an existing VN97 checkpoint."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
    )
    parser.add_argument(
        "--dev-suite",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--probe-per-category",
        type=int,
        default=30,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if (
        args.probe_per_category <= 0
        or not str(
            args.device
        ).startswith("cuda")
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "P4E-F requires positive probe size and CUDA"
        )

    root = Path(
        args.artifact_dir
    ).resolve(strict=True)
    _verify_sums(root)

    checkpoint = (
        load_deployment_checkpoint_file(
            root / "model.vn97ck1"
        )
    )
    package = (
        VN97TokenizerPackage.from_bytes(
            (
                root
                / "tokenizer.vn97tk1"
            ).read_bytes()
        )
    )
    tokenizer = VN97Tokenizer(
        package
    )
    model = checkpoint.model.to(
        args.device
    )
    model.eval()

    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )

    records = _probe_records(
        default_validation(),
        per_category=
            args.probe_per_category,
    )

    prefix_matches = 0
    legacy_raw = 0
    legacy_recovered = 0
    canonical_raw = 0
    canonical_recovered = 0
    categories = {
        category: Counter()
        for category in P4D_CATEGORIES
    }

    for record in records:
        messages = (
            VN97ChatMessage(
                role="user",
                content=record.prompt,
            ),
        )
        training_example = (
            encode_chat_completion_messages(
                tokenizer,
                (
                    *messages,
                    VN97ChatMessage(
                        role="assistant",
                        content=record.answer,
                    ),
                ),
            )
        )
        first_target = next(
            index
            for index, enabled
            in enumerate(
                training_example.target_mask
            )
            if enabled
        )
        training_prefix = (
            training_example.token_ids[
                :first_target
            ]
        )
        canonical_prefix = (
            encode_chat_completion_prompt(
                tokenizer,
                messages,
            )
        )
        if (
            training_prefix
            == canonical_prefix
        ):
            prefix_matches += 1

        legacy = engine.generate_text(
            render_p4_chat_prompt(
                record.prompt
            ),
            max_new_tokens=96,
        )
        canonical = (
            engine.generate_chat_completion(
                messages,
                max_new_tokens=96,
            )
        )
        legacy_fixed = (
            recover_chat_response_text(
                legacy
            )
        )
        canonical_fixed = (
            recover_chat_response_text(
                canonical
            )
        )

        legacy_ok = _record_score(
            record.answer,
            legacy,
        )
        legacy_fixed_ok = (
            _record_score(
                record.answer,
                legacy_fixed,
            )
        )
        canonical_ok = (
            _record_score(
                record.answer,
                canonical,
            )
        )
        canonical_fixed_ok = (
            _record_score(
                record.answer,
                canonical_fixed,
            )
        )

        legacy_raw += int(
            legacy_ok
        )
        legacy_recovered += int(
            legacy_fixed_ok
        )
        canonical_raw += int(
            canonical_ok
        )
        canonical_recovered += int(
            canonical_fixed_ok
        )

        row = categories[
            record.category
        ]
        row[
            "legacy_raw"
        ] += int(legacy_ok)
        row[
            "legacy_recovered"
        ] += int(
            legacy_fixed_ok
        )
        row[
            "canonical_raw"
        ] += int(
            canonical_ok
        )
        row[
            "canonical_recovered"
        ] += int(
            canonical_fixed_ok
        )
        row["tasks"] += 1

    tasks = len(records)
    print(
        "VN97 P4E-F PREFIX "
        f"matches={prefix_matches}/{tasks}",
        flush=True,
    )
    print(
        "VN97 P4E-F GENERALIZATION "
        f"legacy_raw={legacy_raw}/{tasks} "
        f"legacy_recovered={legacy_recovered}/{tasks} "
        f"canonical_raw={canonical_raw}/{tasks} "
        f"canonical_recovered={canonical_recovered}/{tasks}",
        flush=True,
    )
    for category in P4D_CATEGORIES:
        row = categories[
            category
        ]
        print(
            "VN97 P4E-F CATEGORY "
            f"{category} "
            f"legacy_raw={row['legacy_raw']}/{row['tasks']} "
            f"legacy_recovered={row['legacy_recovered']}/{row['tasks']} "
            f"canonical_raw={row['canonical_raw']}/{row['tasks']} "
            f"canonical_recovered={row['canonical_recovered']}/{row['tasks']}",
            flush=True,
        )

    if args.dev_suite is not None:
        suite = load_p4_task_suite(
            Path(args.dev_suite)
        )
        legacy_dev = 0
        canonical_dev = 0
        canonical_dev_recovered = 0
        for task in suite.tasks:
            legacy = engine.generate_text(
                render_p4_chat_prompt(
                    task.prompt
                ),
                max_new_tokens=
                    task.max_new_tokens,
            )
            canonical = (
                engine.generate_chat_completion(
                    (
                        VN97ChatMessage(
                            role="user",
                            content=task.prompt,
                        ),
                    ),
                    max_new_tokens=
                        task.max_new_tokens,
                )
            )
            legacy_dev += int(
                score_p4_output(
                    task,
                    legacy,
                )
            )
            canonical_dev += int(
                score_p4_output(
                    task,
                    canonical,
                )
            )
            canonical_dev_recovered += int(
                score_p4_output(
                    task,
                    recover_chat_response_text(
                        canonical
                    ),
                )
            )
        print(
            "VN97 P4E-F DEV "
            f"legacy_raw={legacy_dev}/{len(suite.tasks)} "
            f"canonical_raw={canonical_dev}/{len(suite.tasks)} "
            f"canonical_recovered={canonical_dev_recovered}/{len(suite.tasks)}",
            flush=True,
        )

    if prefix_matches != tasks:
        raise RuntimeError(
            "canonical segmented prefix still diverges from training"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p4e-prefix-eval: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
