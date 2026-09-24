from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.p2_corpus_bundle",
    SRC / "p2_corpus_bundle.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p2_corpus_bundle.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p2_corpus_bundle"] = module
spec.loader.exec_module(module)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def with_identity(
    body: dict[str, object],
    *,
    key: str,
    prefix: bytes,
) -> bytes:
    value = dict(body)
    value[key] = hashlib.sha256(
        prefix + canonical(body)
    ).hexdigest()
    return canonical(value) + b"\n"


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        intake = root / "intake"
        corpus = root / "corpus"
        intake.mkdir()
        corpus.mkdir()

        raw_source_ids = (
            "vi-dialogue-hoanghai2110",
            "databricks-dolly-15k",
            "openai-gsm8k-train",
        )
        raw_hashes = {
            source_id: hashlib.sha256(
                source_id.encode("utf-8")
            ).hexdigest()
            for source_id in raw_source_ids
        }

        fetch_body = {
            "definition_sha256": "a" * 64,
            "schema": "VN97P2FETCH1",
            "sources": [
                {
                    "bytes": 100 + index,
                    "filename": f"raw-{index}.jsonl",
                    "final_url": "https://huggingface.co/example",
                    "records": 10 + index,
                    "revision": str(index + 1) * 40,
                    "sha256": raw_hashes[source_id],
                    "source_id": source_id,
                }
                for index, source_id in enumerate(
                    raw_source_ids
                )
            ],
        }
        fetch_path = root / "source-fetch.vn97p2fetch1.json"
        fetch_path.write_bytes(
            with_identity(
                fetch_body,
                key="receipt_id",
                prefix=b"VN97P2FETCH1\0",
            )
        )

        split_rows = {
            "training": (
                "vi-dialogue-hoanghai2110-training",
                "vi",
                "Apache-2.0",
                {
                    "messages": [
                        {"role": "user", "content": "Xin chào"},
                        {"role": "assistant", "content": "Chào bạn"},
                    ]
                },
            ),
            "validation": (
                "databricks-dolly-15k-validation",
                "dolly",
                "CC-BY-SA-3.0",
                {
                    "messages": [
                        {"role": "user", "content": "Summarize A"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
            ),
            "release": (
                "openai-gsm8k-train-release",
                "gsm8k",
                "MIT",
                {
                    "messages": [
                        {"role": "user", "content": "2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                },
            ),
        }

        definition_sources = []
        manifest_sources = []
        split_specs = {}
        for split, (
            source_id,
            stem,
            license_name,
            row,
        ) in split_rows.items():
            filename = f"{stem}.{split}.jsonl"
            data = canonical(row) + b"\n"
            (intake / filename).write_bytes(data)
            origin = f"https://example.invalid/{stem}"
            definition_sources.append(
                {
                    "license": license_name,
                    "license_approved": True,
                    "mode": "chat",
                    "origin": origin,
                    "path": filename,
                    "source_id": source_id,
                    "split": split,
                }
            )
            manifest_sources.append(
                {
                    "content_bytes": len(data),
                    "content_sha256": hashlib.sha256(data).hexdigest(),
                    "duplicate_records": 0,
                    "input_records": 1,
                    "kept_records": 1,
                    "license": license_name,
                    "license_approved": True,
                    "mode": "chat",
                    "origin": origin,
                    "source_id": source_id,
                    "split": split,
                }
            )
            (corpus / f"{split}.jsonl").write_bytes(data)
            split_specs[split] = {
                "bytes": len(data),
                "mode": "chat",
                "records": 1,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        definition = {
            "profile_id": "vn97-production-intelligence-v1",
            "schema": "VN97CORPUSDEF1",
            "sources": sorted(
                definition_sources,
                key=lambda item: (
                    item["split"],
                    item["source_id"],
                ),
            ),
        }
        definition_bytes = canonical(definition) + b"\n"
        definition_path = intake / "corpus-definition.vn97corpusdef1.json"
        definition_path.write_bytes(definition_bytes)
        definition_sha = hashlib.sha256(
            definition_bytes
        ).hexdigest()

        summary_body = {
            "definition_sha256": definition_sha,
            "raw_source_sha256": {
                key: raw_hashes[key]
                for key in sorted(raw_hashes)
            },
            "schema": "VN97P2SRC1",
            "sources": [],
            "split_records": {
                "release": 1,
                "training": 1,
                "validation": 1,
            },
        }
        summary_path = root / "p2-source-summary.vn97p2src1.json"
        summary_path.write_bytes(
            with_identity(
                summary_body,
                key="summary_id",
                prefix=b"VN97P2SRC1\0",
            )
        )

        manifest_identity = {
            "profile_id": "vn97-production-intelligence-v1",
            "schema": "VN97CORPUS1",
            "sources": sorted(
                manifest_sources,
                key=lambda item: (
                    item["split"],
                    item["source_id"],
                ),
            ),
            "splits": {
                split: split_specs[split]
                for split in (
                    "release",
                    "training",
                    "validation",
                )
            },
        }
        manifest = dict(manifest_identity)
        manifest["manifest_id"] = hashlib.sha256(
            b"VN97CORPUS1\0"
            + canonical(manifest_identity)
        ).hexdigest()
        manifest_path = corpus / "corpus.vn97corpus1.json"
        manifest_path.write_bytes(
            canonical(manifest) + b"\n"
        )

        receipt = module.verify_p2_corpus_chain(
            fetch_receipt_path=fetch_path,
            source_summary_path=summary_path,
            definition_path=definition_path,
            corpus_dir=corpus,
            repository_commit="1" * 40,
        )
        payload = receipt.canonical_object()
        assert payload["schema"] == "VN97P2BUNDLE1"
        assert len(payload["bundle_id"]) == 64
        assert payload["corpus_manifest_id"] == manifest["manifest_id"]
        assert payload["definition_sha256"] == definition_sha
        assert payload["splits"]["training"]["records"] == 1
        assert receipt.to_bytes().endswith(b"\n")

        original = (corpus / "release.jsonl").read_bytes()
        (corpus / "release.jsonl").write_bytes(
            original + b"{}\n"
        )
        expect_failure(
            "tampered sealed release split",
            lambda: module.verify_p2_corpus_chain(
                fetch_receipt_path=fetch_path,
                source_summary_path=summary_path,
                definition_path=definition_path,
                corpus_dir=corpus,
                repository_commit="1" * 40,
            ),
        )
        (corpus / "release.jsonl").write_bytes(original)

        expect_failure(
            "invalid repository commit",
            lambda: module.verify_p2_corpus_chain(
                fetch_receipt_path=fetch_path,
                source_summary_path=summary_path,
                definition_path=definition_path,
                corpus_dir=corpus,
                repository_commit="bad",
            ),
        )

        print(
            "VN97 P2 corpus bundle host contract PASS "
            f"bundle={payload['bundle_id']}"
        )


if __name__ == "__main__":
    main()
