from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_source_integrity import PINNED_ORACLE_MAMBA_COMMIT\n\nfrom .mamba2_parity import (
    Mamba2ParityReceipt,
    compare_tensor,
    hash_generated_tokens,
    hash_token_probe,
    load_trace,
    write_parity_receipt,
)


def _load_tokens(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("token JSON must be a non-empty list")
    return [int(value) for value in payload]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare source Mamba-2 and VN97 G0 traces before production use."
        )
    )
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument("--vn97-trace", required=True, type=Path)
    parser.add_argument("--token-probe", required=True, type=Path)
    parser.add_argument("--source-generated", required=True, type=Path)
    parser.add_argument("--vn97-generated", required=True, type=Path)
    parser.add_argument("--source-weight-sha256", required=True)
    parser.add_argument("--vn97-checkpoint-sha256", required=True)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument(
        "--implementation-source",
        default=(\n            "state-spaces/mamba@"\n            + PINNED_ORACLE_MAMBA_COMMIT\n            + "_vs_vn97_native"\n        ),
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = load_trace(args.source_trace)
    target = load_trace(args.vn97_trace)
    required = (
        "layer0.hidden",
        "layer0.conv_state",
        "layer0.ssm_state",
        "final.logits",
    )
    comparisons = []
    for name in required:
        if name not in source or name not in target:
            raise ValueError(f"missing required trace tensor: {name}")
        comparisons.append(
            compare_tensor(
                name,
                source[name],
                target[name],
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    extra_names = sorted(
        (set(source) & set(target)) - set(required)
    )
    for name in extra_names:
        comparisons.append(
            compare_tensor(
                name,
                source[name],
                target[name],
                atol=args.atol,
                rtol=args.rtol,
            )
        )

    token_probe = _load_tokens(args.token_probe)
    source_generated = _load_tokens(args.source_generated)
    vn97_generated = _load_tokens(args.vn97_generated)
    receipt = Mamba2ParityReceipt(
        source_weight_sha256=args.source_weight_sha256,
        vn97_checkpoint_sha256=args.vn97_checkpoint_sha256,
        token_probe_sha256=hash_token_probe(token_probe),
        precision=args.precision,
        implementation_source=args.implementation_source,
        comparisons=tuple(comparisons),
        source_generated_tokens_sha256=hash_generated_tokens(
            source_generated
        ),
        vn97_generated_tokens_sha256=hash_generated_tokens(
            vn97_generated
        ),
    )
    write_parity_receipt(args.output, receipt)
    print(
        json.dumps(
            {
                "status": (
                    "EXACT_PARITY_PASS"
                    if receipt.passed
                    else "EXACT_PARITY_FAIL"
                ),
                "receipt_id": receipt.receipt_id(),
                "comparisons": len(receipt.comparisons),
                "output": str(args.output),
                "production_parity": receipt.passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
