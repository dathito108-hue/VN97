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
    "vn97.p2_pilot_handoff",
    SRC / "p2_pilot_handoff.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p2_pilot_handoff.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p2_pilot_handoff"] = module
spec.loader.exec_module(module)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def write_sha256sums(root: Path) -> None:
    rows = []
    for rel in sorted(
        [
            *(f"corpus/{name}" for name in (
                "corpus.vn97corpus1.json",
                "training.jsonl",
                "validation.jsonl",
                "release.jsonl",
            )),
            *(f"evidence/{name}" for name in (
                "source-fetch.vn97p2fetch1.json",
                "p2-source-summary.vn97p2src1.json",
                "corpus-definition.vn97corpusdef1.json",
                "p2-corpus-bundle.vn97p2bundle1.json",
            )),
        ]
    ):
        data = (root / rel).read_bytes()
        rows.append(
            f"{hashlib.sha256(data).hexdigest()}  {rel}"
        )
    (root / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        artifact = root / "artifact"
        corpus_dir = artifact / "corpus"
        evidence_dir = artifact / "evidence"
        corpus_dir.mkdir(parents=True)
        evidence_dir.mkdir(parents=True)

        split_specs = {}
        for split, row in (
            (
                "training",
                {
                    "messages": [
                        {"role": "user", "content": "Xin chào"},
                        {"role": "assistant", "content": "Chào bạn"},
                    ]
                },
            ),
            (
                "validation",
                {
                    "messages": [
                        {"role": "user", "content": "2+2?"},
                        {"role": "assistant", "content": "4"},
                    ]
                },
            ),
            (
                "release",
                {
                    "messages": [
                        {"role": "user", "content": "Summarize A"},
                        {"role": "assistant", "content": "A"},
                    ]
                },
            ),
        ):
            data = canonical(row) + b"\n"
            (corpus_dir / f"{split}.jsonl").write_bytes(data)
            split_specs[split] = {
                "bytes": len(data),
                "mode": "chat",
                "records": 1,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        manifest_body = {
            "profile_id": "vn97-production-intelligence-v1",
            "schema": "VN97CORPUS1",
            "sources": [],
            "splits": {
                split: split_specs[split]
                for split in (
                    "release",
                    "training",
                    "validation",
                )
            },
        }
        manifest = dict(manifest_body)
        manifest["manifest_id"] = hashlib.sha256(
            b"VN97CORPUS1\0" + canonical(manifest_body)
        ).hexdigest()
        manifest_bytes = canonical(manifest) + b"\n"
        (corpus_dir / "corpus.vn97corpus1.json").write_bytes(
            manifest_bytes
        )

        for name in (
            "source-fetch.vn97p2fetch1.json",
            "p2-source-summary.vn97p2src1.json",
            "corpus-definition.vn97corpusdef1.json",
        ):
            (evidence_dir / name).write_bytes(
                canonical({"placeholder": name}) + b"\n"
            )

        bundle_body = {
            "corpus_manifest_id": manifest["manifest_id"],
            "corpus_manifest_sha256":
                hashlib.sha256(manifest_bytes).hexdigest(),
            "definition_sha256": "1" * 64,
            "fetch_receipt_sha256": "2" * 64,
            "repository_commit": "a" * 40,
            "schema": "VN97P2BUNDLE1",
            "source_summary_sha256": "3" * 64,
            "splits": {
                split: {
                    "bytes": split_specs[split]["bytes"],
                    "records": split_specs[split]["records"],
                    "sha256": split_specs[split]["sha256"],
                }
                for split in (
                    "release",
                    "training",
                    "validation",
                )
            },
        }
        bundle = dict(bundle_body)
        bundle["bundle_id"] = hashlib.sha256(
            b"VN97P2BUNDLE1\0" + canonical(bundle_body)
        ).hexdigest()
        bundle_bytes = canonical(bundle) + b"\n"
        bundle_path = (
            evidence_dir / "p2-corpus-bundle.vn97p2bundle1.json"
        )
        bundle_path.write_bytes(bundle_bytes)

        write_sha256sums(artifact)

        verified = module.verify_corpus_artifact(
            artifact,
            expected_corpus_commit="a" * 40,
        )
        assert verified.bundle_id == bundle["bundle_id"]
        assert verified.corpus_manifest_id == manifest["manifest_id"]
        assert verified.corpus_manifest_sha256 == hashlib.sha256(
            manifest_bytes
        ).hexdigest()

        pilot = root / "pilot"
        pilot.mkdir()
        files = {
            "tokenizer.vn97tk1": b"tokenizer-bytes",
            "model.vn97ck1": b"checkpoint-bytes",
            "campaign-report.json": canonical(
                {"schema": "VN97CAMP2", "selected_candidate_id": "0123456789abcdef"}
            ),
            "model.vn97mi1": b"model-image-bytes",
        }
        for name, data in files.items():
            (pilot / name).write_bytes(data)

        pilot_receipt = {
            "campaign_report_sha256":
                hashlib.sha256(files["campaign-report.json"]).hexdigest(),
            "candidate_id": "0123456789abcdef",
            "checkpoint_sha256":
                hashlib.sha256(files["model.vn97ck1"]).hexdigest(),
            "corpus_manifest_id": manifest["manifest_id"],
            "corpus_manifest_sha256":
                hashlib.sha256(manifest_bytes).hexdigest(),
            "device": "cpu",
            "model_image_bytes": len(files["model.vn97mi1"]),
            "model_image_sha256":
                hashlib.sha256(files["model.vn97mi1"]).hexdigest(),
            "profile_sha256": "4" * 64,
            "schema": "VN97PILOT1",
            "tokenizer_sha256":
                hashlib.sha256(files["tokenizer.vn97tk1"]).hexdigest(),
        }
        pilot_bytes = canonical(pilot_receipt) + b"\n"
        (pilot / "pilot.vn97pilot1.json").write_bytes(
            pilot_bytes
        )

        run = module.build_pilot_run_receipt(
            corpus=verified,
            pilot_output=pilot,
            training_commit="b" * 40,
            corpus_run_id="12345",
            training_run_id="67890",
            python_version="3.12.7",
            torch_version="2.8.0+cpu",
        )
        payload = run.canonical_object()
        assert payload["schema"] == "VN97P2RUN1"
        assert payload["candidate_id"] == "0123456789abcdef"
        assert payload["corpus_bundle_id"] == bundle["bundle_id"]
        assert payload["corpus_run_id"] == "12345"
        assert payload["training_run_id"] == "67890"
        assert payload["python_version"] == "3.12.7"
        assert payload["torch_version"] == "2.8.0+cpu"
        assert len(payload["run_identity"]) == 64
        assert run.to_bytes().endswith(b"\n")

        original = (corpus_dir / "release.jsonl").read_bytes()
        (corpus_dir / "release.jsonl").write_bytes(
            original + b"{}\n"
        )
        expect_failure(
            "tampered downloaded corpus",
            lambda: module.verify_corpus_artifact(
                artifact,
                expected_corpus_commit="a" * 40,
            ),
        )
        (corpus_dir / "release.jsonl").write_bytes(original)

        original_sums = (artifact / "SHA256SUMS").read_text(
            encoding="utf-8"
        )
        write_sha256sums(artifact)
        bad_bundle = dict(bundle)
        bad_bundle["repository_commit"] = "c" * 40
        bad_body = dict(bad_bundle)
        bad_body.pop("bundle_id", None)
        bad_bundle["bundle_id"] = hashlib.sha256(
            b"VN97P2BUNDLE1\0" + canonical(bad_body)
        ).hexdigest()
        bundle_path.write_bytes(canonical(bad_bundle) + b"\n")
        write_sha256sums(artifact)
        expect_failure(
            "wrong corpus commit",
            lambda: module.verify_corpus_artifact(
                artifact,
                expected_corpus_commit="a" * 40,
            ),
        )
        bundle_path.write_bytes(bundle_bytes)
        (artifact / "SHA256SUMS").write_text(
            original_sums,
            encoding="utf-8",
        )

        bad_pilot = dict(pilot_receipt)
        bad_pilot["corpus_manifest_sha256"] = "f" * 64
        (pilot / "pilot.vn97pilot1.json").write_bytes(
            canonical(bad_pilot) + b"\n"
        )
        expect_failure(
            "pilot bound to wrong corpus bytes",
            lambda: module.build_pilot_run_receipt(
                corpus=verified,
                pilot_output=pilot,
                training_commit="b" * 40,
                corpus_run_id="12345",
                training_run_id="67890",
                python_version="3.12.7",
                torch_version="2.8.0+cpu",
            ),
        )

        print(
            "VN97 P2 pilot handoff host contract PASS "
            f"bundle={verified.bundle_id}"
        )


if __name__ == "__main__":
    main()
