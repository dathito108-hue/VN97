from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "vn97" / "release_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "vn97_release_candidate_m19c",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load release_candidate.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Device = MODULE.VN97ReleaseCandidateDeviceEvidence
Manifest = MODULE.VN97ReleaseCandidateManifest
load_candidate = MODULE.load_release_candidate_directory


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def build_candidate(root: Path) -> tuple[object, str]:
    root.mkdir()
    checkpoint = b"VN97CK1\0" + (b"c" * 256)
    tokenizer = b"VN97TK1\0" + (b"t" * 64)
    production = (
        b'{"schema":"VN97PRODCAMP1"}'
    )
    evidence = (
        b'{"schema":"VN97MOBEVID1"}'
    )

    checkpoint_sha = sha(checkpoint)
    tokenizer_sha = sha(tokenizer)
    production_sha = sha(production)
    evidence_sha = sha(evidence)

    evidence_entry = Device(
        evidence_sha256=evidence_sha,
        manufacturer="fixture",
        model="phone",
        sdk_int=37,
        abi="arm64-v8a",
        runs=5,
        text_prefill_p95_ms=10.0,
        text_decode_p95_ms_per_token=6.0,
        speech_prefill_p95_ms=None,
        peak_pss_kib=120000,
        thermal_status_max=2,
        battery_energy_counter_delta_nwh=100,
    )
    manifest = Manifest(
        selected_candidate_id="0123456789abcdef",
        checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=len(checkpoint),
        tokenizer_sha256=tokenizer_sha,
        tokenizer_bytes=len(tokenizer),
        production_campaign_report_sha256=production_sha,
        model_image_sha256="55" * 32,
        tile_rows=16,
        tile_cols=16,
        speech_enabled=False,
        vision_enabled=False,
        speech_training_report_sha256=None,
        vision_training_report_sha256=None,
        device_evidence=(evidence_entry,),
    )

    (root / "model.vn97ck1").write_bytes(
        checkpoint
    )
    (root / "tokenizer.vn97tk1").write_bytes(
        tokenizer
    )
    (
        root /
        "production-campaign-report.json"
    ).write_bytes(production)

    evidence_root = root / "device-evidence"
    evidence_root.mkdir()
    (
        evidence_root /
        f"001-{evidence_sha}.json"
    ).write_bytes(evidence)

    manifest_bytes = manifest.to_bytes()
    (
        root /
        "release-candidate.vn97rc1"
    ).write_bytes(manifest_bytes)
    return manifest, sha(manifest_bytes)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        good = base / "good"
        manifest, manifest_sha = build_candidate(
            good
        )
        loaded = load_candidate(good)
        assert loaded.manifest == manifest
        assert loaded.manifest_sha256 == manifest_sha
        assert loaded.checkpoint_path.name == "model.vn97ck1"
        assert (
            len(loaded.device_evidence_paths)
            == 1
        )

        tampered = base / "tampered"
        build_candidate(tampered)
        with (
            tampered /
            "model.vn97ck1"
        ).open("ab") as output:
            output.write(b"x")
        expect_failure(
            "checkpoint tamper",
            lambda: load_candidate(tampered),
        )

        unexpected = base / "unexpected"
        build_candidate(unexpected)
        (
            unexpected /
            "private.key"
        ).write_bytes(b"secret")
        expect_failure(
            "unexpected candidate file",
            lambda: load_candidate(unexpected),
        )

        wrong_name = base / "wrong-evidence-name"
        build_candidate(wrong_name)
        evidence_root = (
            wrong_name /
            "device-evidence"
        )
        evidence_file = next(
            evidence_root.iterdir()
        )
        evidence_file.rename(
            evidence_root /
            "001-wrong.json"
        )
        expect_failure(
            "evidence filename mismatch",
            lambda: load_candidate(wrong_name),
        )

        symlink_root = base / "candidate-link"
        try:
            symlink_root.symlink_to(
                good,
                target_is_directory=True,
            )
        except OSError:
            symlink_root = None
        if symlink_root is not None:
            expect_failure(
                "candidate root symlink",
                lambda: load_candidate(
                    symlink_root
                ),
            )

    print(
        "M19C signed candidate directory contracts: PASS"
    )


if __name__ == "__main__":
    main()
