#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_RUNTIME_SCHEMA = "VN97R2F1RUNTIME1"
OUTPUT_SCHEMA = "VN97R2F2BRIDGE1"

def require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SystemExit(f"{label} must be lowercase SHA-256")
    return value

def build_binding(runtime: dict, tokenizer_model_id: str) -> dict:
    if runtime.get("schema") != EXPECTED_RUNTIME_SCHEMA:
        raise SystemExit("runtime descriptor schema mismatch")
    runtime_id = require_sha(runtime.get("runtime_id"), "runtime_id")
    bundle_id = require_sha(runtime.get("bundle_id"), "bundle_id")
    tokenizer_model_id = require_sha(tokenizer_model_id, "tokenizer_model_id")
    vocab_size = runtime.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 1:
        raise SystemExit("vocab_size must be an integer greater than 1")
    if runtime.get("same_weights_semantics") is not True:
        raise SystemExit("runtime must declare same_weights_semantics=true")
    if runtime.get("quantization_used") is not False:
        raise SystemExit("F2 production binding requires dense non-quantized runtime")
    return {
        "bundle_id": bundle_id,
        "runtime_id": runtime_id,
        "same_weights_semantics": True,
        "schema": OUTPUT_SCHEMA,
        "tokenizer_model_id": tokenizer_model_id,
        "vocab_size": vocab_size,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fail-closed VN97 R2 F2 tokenizer/runtime binding."
    )
    parser.add_argument("--runtime-descriptor", required=True, type=Path)
    parser.add_argument("--tokenizer-model-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = args.runtime_descriptor.read_text(encoding="utf-8")
    if "\x00" in raw:
        raise SystemExit("runtime descriptor contains NUL")
    runtime = json.loads(raw)
    if not isinstance(runtime, dict):
        raise SystemExit("runtime descriptor must be a JSON object")
    binding = build_binding(runtime, args.tokenizer_model_id)
    encoded = json.dumps(
        binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="ascii")

if __name__ == "__main__":
    main()
