from __future__ import annotations

import hashlib
import importlib.util
import json
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


RUN = load_module(
    "vn97.production_run_manifest",
    "production_run_manifest.py",
)
SEAL = load_module(
    "vn97.production_run_sealer",
    "production_run_sealer.py",
)
CLI = load_module(
    "vn97.production_run_seal_cli",
    "production_run_seal_cli.py",
)

Environment = SEAL.VN97ProductionEnvironment
parse_manifest = RUN.parse_production_run_manifest
verify_inputs = RUN.verify_production_run_inputs


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def write(
    root: Path,
    relative: str,
    data: bytes,
) -> None:
    target = root / relative
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_bytes(data)


def populate_workspace(root: Path) -> None:
    write(
        root,
        "config/campaign.json",
        canonical(
            {
                "candidates": [
                    {
                        "d_model": 8,
                        "d_state": 2,
                        "embedding_rank": 4,
                        "learning_rate": 0.001,
                        "n_layers": 1,
                        "seed": 0,
                    }
                ],
                "schema": "VN97CAMPDEF1",
            }
        ),
    )
    write(
        root,
        "config/language-options.json",
        canonical(
            {
                "device": "cpu",
                "max_parameters": 1_000_000,
                "max_validation_loss": 10.0,
            }
        ),
    )
    write(
        root,
        "config/speech-options.json",
        canonical(
            {
                "device": "cpu",
                "max_speech_validation_loss": 10.0,
            }
        ),
    )
    write(
        root,
        "config/intake-options.json",
        canonical(
            {
                "min_device_runs": 5,
                "min_distinct_device_profiles": 1,
            }
        ),
    )

    write(
        root,
        "data/train/a.jsonl",
        b'{"text":"train-a"}\n',
    )
    write(
        root,
        "data/train/b.txt",
        b"train-b\n",
    )
    write(
        root,
        "data/validation/validation.jsonl",
        b'{"text":"validation"}\n',
    )
    write(
        root,
        "data/release/release.jsonl",
        b'{"text":"release"}\n',
    )

    for split, text in (
        ("train", "speech train"),
        ("validation", "speech validation"),
        ("release", "speech release"),
    ):
        write(
            root,
            f"speech/{split}/001.wav",
            (
                b"RIFF-fixture-"
                + split.encode("ascii")
            ),
        )
        write(
            root,
            f"speech/{split}/manifest.jsonl",
            (
                json.dumps(
                    {
                        "audio": "001.wav",
                        "text": text,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower()
    assert len(commit) == 40

    env = Environment(
        python_version="3.12.fixture",
        torch_version="2.5.fixture",
        platform_system="Linux",
        platform_machine="x86_64",
    )

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        workspace = base / "workspace"

        assert CLI.main(
            [
                "bootstrap",
                "--workspace-root",
                str(workspace),
            ]
        ) == 0

        expected_dirs = {
            "config",
            "data",
            "data/train",
            "data/validation",
            "data/release",
            "speech",
            "speech/train",
            "speech/validation",
            "speech/release",
            "device-evidence",
            "out",
        }
        actual_dirs = {
            path.relative_to(
                workspace
            ).as_posix()
            for path in workspace.rglob("*")
            if path.is_dir()
        }
        assert expected_dirs.issubset(
            actual_dirs
        )
        assert not (
            workspace /
            "production-run.vn97run1"
        ).exists()
        assert (
            workspace /
            "config/campaign.json.example"
        ).is_file()

        expect_failure(
            "unpopulated bootstrap seal",
            lambda:
                SEAL.build_production_run_manifest(
                    workspace_root=workspace,
                    repository_root=ROOT,
                    environment=env,
                ),
        )

        populate_workspace(
            workspace
        )
        manifest = (
            SEAL.build_production_run_manifest(
                workspace_root=workspace,
                repository_root=ROOT,
                environment=env,
            )
        )
        assert (
            manifest.repository_commit
            == commit
        )
        assert (
            manifest.python_version
            == env.python_version
        )
        assert tuple(
            item.path
            for item in
            manifest.language_training
        ) == (
            "data/train/a.jsonl",
            "data/train/b.txt",
        )
        assert (
            manifest
            .speech_training
            .audio[0]
            .path
            == "speech/train/001.wav"
        )

        encoded = manifest.to_bytes()
        assert (
            parse_manifest(encoded)
            == manifest
        )
        assert (
            manifest.manifest_sha256
            == hashlib.sha256(
                encoded
            ).hexdigest()
        )
        verify_inputs(
            manifest,
            workspace_root=workspace,
        )

        output = (
            workspace /
            "production-run.vn97run1"
        )
        SEAL.write_manifest_atomic(
            output,
            manifest,
        )
        assert (
            output.read_bytes()
            == encoded
        )
        expect_failure(
            "manifest overwrite",
            lambda:
                SEAL.write_manifest_atomic(
                    output,
                    manifest,
                ),
        )

        train_audio = (
            workspace /
            "speech/train/001.wav"
        )
        original = train_audio.read_bytes()
        train_audio.write_bytes(
            original + b"x"
        )
        expect_failure(
            "audio tamper after seal",
            lambda:
                verify_inputs(
                    manifest,
                    workspace_root=
                        workspace,
                ),
        )
        train_audio.write_bytes(
            original
        )

        write(
            workspace,
            "data/train/junk.bin",
            b"junk",
        )
        expect_failure(
            "unknown language file type",
            lambda:
                SEAL.build_production_run_manifest(
                    workspace_root=workspace,
                    repository_root=ROOT,
                    environment=env,
                ),
        )
        (
            workspace /
            "data/train/junk.bin"
        ).unlink()

    with tempfile.TemporaryDirectory() as tmp:
        nonempty = Path(tmp)
        write(
            nonempty,
            "already.txt",
            b"x",
        )
        expect_failure(
            "bootstrap nonempty directory",
            lambda:
                SEAL.bootstrap_workspace(
                    nonempty
                ),
        )

    print(
        "M19H VN97RUN1 sealer/bootstrap contracts: PASS"
    )


if __name__ == "__main__":
    main()
