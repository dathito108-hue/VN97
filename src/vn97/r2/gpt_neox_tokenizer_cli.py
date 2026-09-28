from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gpt_neox_tokenizer import (
    GptNeoXBpeTokenizer,
    build_g08_from_g03_capsule,
    compile_g08_tokenizer_descriptor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.8 GPT-NeoX tokenizer identity and parity tooling."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser("assess")
    assess.add_argument("--tokenizer-root", required=True, type=Path)
    assess.add_argument("--capsule-id", required=True)
    assess.add_argument("--expected-token-id-space", type=int)

    build = sub.add_parser("build-from-g03")
    build.add_argument("--capsule-root", required=True, type=Path)
    build.add_argument("--output", type=Path)

    probe = sub.add_parser("probe")
    probe.add_argument("--tokenizer-root", required=True, type=Path)
    probe.add_argument("--text", required=True)

    args = parser.parse_args()

    if args.command == "assess":
        result = compile_g08_tokenizer_descriptor(
            args.tokenizer_root,
            capsule_id=args.capsule_id,
            expected_token_id_space=args.expected_token_id_space,
        )
        print(json.dumps(result, sort_keys=True))
        return

    if args.command == "build-from-g03":
        result = build_g08_from_g03_capsule(
            args.capsule_root,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_MAMBA2_G08_TOKENIZER_BUILT",
                    "tokenizer_id": result["tokenizer_id"],
                    "token_id_space": result["token_id_space"],
                    "eos_token_id": result["eos_token_id"],
                    "runtime_logits_size": result["runtime_logits_size"],
                    "sampling_must_mask_padded_ids":
                        result["sampling_must_mask_padded_ids"],
                    "production_activation_authorized":
                        result["production_activation_authorized"],
                },
                sort_keys=True,
            )
        )
        return

    root = args.tokenizer_root
    config = json.loads(
        (root / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    tokenizer = GptNeoXBpeTokenizer.from_files(
        root / "vocab.json",
        root / "merges.txt",
        eos_token=str(config["eos_token"]),
    )
    ids = tokenizer.encode(args.text)
    print(
        json.dumps(
            {
                "text": args.text,
                "token_ids": ids,
                "decoded": tokenizer.decode(ids),
                "token_id_space": tokenizer.token_id_space,
                "eos_token_id": tokenizer.eos_token_id,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
