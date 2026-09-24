from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        SRC / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load {filename}"
        )
    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RC = load_module(
    "vn97.release_candidate",
    "release_candidate.py",
)
READY = load_module(
    "vn97.production_readiness",
    "production_readiness.py",
)

Device = RC.VN97ReleaseCandidateDeviceEvidence
Manifest = RC.VN97ReleaseCandidateManifest
evaluate = READY.evaluate_production_readiness
parse_report = (
    READY.parse_production_readiness_report
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_candidate(root: Path) -> None:
    root.mkdir()
    checkpoint = b"VN97CK1\0" + (b"c" * 256)
    tokenizer = b"VN97TK1\0" + (b"t" * 64)
    production = (
        b'{"schema":"VN97PRODCAMP1"}'
    )
    evidence = (
        b'{"schema":"VN97MOBEVID1"}'
    )
    evidence_sha = sha(evidence)

    manifest = Manifest(
        selected_candidate_id=
            "0123456789abcdef",
        checkpoint_sha256=
            sha(checkpoint),
        checkpoint_bytes=
            len(checkpoint),
        tokenizer_sha256=
            sha(tokenizer),
        tokenizer_bytes=
            len(tokenizer),
        production_campaign_report_sha256=
            sha(production),
        model_image_sha256=
            "55" * 32,
        tile_rows=16,
        tile_cols=16,
        speech_enabled=False,
        vision_enabled=False,
        speech_training_report_sha256=None,
        vision_training_report_sha256=None,
        device_evidence=(
            Device(
                evidence_sha256=
                    evidence_sha,
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
            ),
        ),
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

    (
        root /
        "release-candidate.vn97rc1"
    ).write_bytes(manifest.to_bytes())


def make_executable(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\nexit 0\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def build_repo(path: Path) -> None:
    (
        path /
        "android/app/src/main/assets/vn97-bootstrap"
    ).mkdir(parents=True)
    (
        path /
        "android/app/src/main/assets/vn97-bootstrap/README.txt"
    ).write_text(
        "bootstrap slot\n",
        encoding="utf-8",
    )
    (
        path /
        "android/app/build.gradle.kts"
    ).write_text(
        'val vn97VersionCode = 190100\n'
        'val vn97VersionName = "1.0.0-rc1"\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "-q"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "fixture@example.invalid",
        ],
        cwd=path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "config",
            "user.name",
            "VN97 Fixture",
        ],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "add", "."],
        cwd=path,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=path,
        check=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        repo = base / "repo"
        repo.mkdir()
        build_repo(repo)

        candidate = base / "candidate"
        build_candidate(candidate)

        publisher_key = (
            base /
            "publisher.private"
        )
        publisher_key.write_bytes(b"k" * 32)

        validation = base / "heldout.jsonl"
        validation.write_text(
            '{"text":"heldout"}\n',
            encoding="utf-8",
        )

        keystore = base / "release.jks"
        keystore.write_bytes(b"keystore")

        tools = base / "tools"
        tools.mkdir()
        gradle = tools / "gradle"
        apksigner = tools / "apksigner"
        aapt = tools / "aapt"
        for path in (
            gradle,
            apksigner,
            aapt,
        ):
            make_executable(path)

        output = base / "release-output"
        environment = {
            "VN97_RELEASE_KEYSTORE":
                str(keystore),
            "VN97_RELEASE_STORE_PASSWORD":
                "store-pass",
            "VN97_RELEASE_KEY_ALIAS":
                "release",
            "VN97_RELEASE_KEY_PASSWORD":
                "key-pass",
        }

        ready = evaluate(
            repository_root=repo,
            release_candidate_dir=candidate,
            publisher_private_key=
                publisher_key,
            validation_inputs=(
                validation,
            ),
            speech_validation_input=None,
            vision_validation_input=None,
            output_dir=output,
            key_id="publisher.main",
            capability_version=1,
            source_origin=
                "vn97-release-candidate",
            source_license="proprietary",
            max_validation_loss=3.0,
            max_speech_validation_loss=None,
            max_vision_validation_loss=None,
            gradle=str(gradle),
            apksigner=str(apksigner),
            aapt=str(aapt),
            environment=environment,
            module_available=
                lambda _: True,
        )
        assert ready.ready
        assert ready.status == "READY"
        assert not ready.blockers
        assert ready.candidate is not None
        assert (
            ready.candidate
            .selected_candidate_id
            == "0123456789abcdef"
        )
        encoded = ready.to_bytes()
        assert parse_report(encoded) == ready

        blocked = evaluate(
            repository_root=repo,
            release_candidate_dir=None,
            publisher_private_key=None,
            validation_inputs=tuple(),
            speech_validation_input=None,
            vision_validation_input=None,
            output_dir=None,
            key_id=None,
            capability_version=None,
            source_origin=None,
            source_license=None,
            max_validation_loss=None,
            max_speech_validation_loss=None,
            max_vision_validation_loss=None,
            gradle="missing-gradle",
            apksigner=None,
            aapt=None,
            environment={},
            which=lambda _: None,
            module_available=
                lambda _: False,
        )
        assert not blocked.ready
        codes = {
            item.code
            for item in blocked.blockers
        }
        required = {
            "candidate.missing",
            "publisher_private_key.missing",
            "language_validation.missing",
            "release_metadata.invalid",
            "quality_thresholds.invalid",
            "python_dependencies.missing",
            "android_signing_environment.missing",
            "gradle.missing",
            "apksigner.missing",
            "aapt.missing",
            "output_directory.missing",
        }
        assert required.issubset(codes)
        assert (
            parse_report(
                blocked.to_bytes()
            )
            == blocked
        )

        dirty_file = (
            repo /
            "android/app/build.gradle.kts"
        )
        dirty_file.write_text(
            dirty_file.read_text(
                encoding="utf-8"
            )
            + "// dirty\n",
            encoding="utf-8",
        )
        dirty = evaluate(
            repository_root=repo,
            release_candidate_dir=candidate,
            publisher_private_key=
                publisher_key,
            validation_inputs=(
                validation,
            ),
            speech_validation_input=None,
            vision_validation_input=None,
            output_dir=output,
            key_id="publisher.main",
            capability_version=1,
            source_origin=
                "vn97-release-candidate",
            source_license="proprietary",
            max_validation_loss=3.0,
            max_speech_validation_loss=None,
            max_vision_validation_loss=None,
            gradle=str(gradle),
            apksigner=str(apksigner),
            aapt=str(aapt),
            environment=environment,
            module_available=
                lambda _: True,
        )
        assert not dirty.ready
        assert (
            "repository.git_not_clean"
            in {
                item.code
                for item in dirty.blockers
            }
        )

    print(
        "M19E VN97READY1 production readiness contracts: PASS"
    )


if __name__ == "__main__":
    main()
