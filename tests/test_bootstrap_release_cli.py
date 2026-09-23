import json

import pytest
import torch

from vn97 import (
    VN97Config,
    VN97ReleaseQualityError,
    VN97LanguageCore,
    VN97TokenizerPackage,
    save_deployment_checkpoint,
)
from vn97.bootstrap_release_cli import main


cryptography = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ed25519"
)
serialization = pytest.importorskip(
    "cryptography.hazmat.primitives.serialization"
)


def _checkpoint_and_tokenizer(tmp_path):
    tokenizer = VN97TokenizerPackage((b" the", b"ing", b"VN97"))
    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            n_layers=2,
            d_state=4,
        )
    ).eval()

    checkpoint = tmp_path / "model.vn97ck1"
    save_deployment_checkpoint(model, checkpoint)
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer.to_bytes())
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        '{"text":"VN97 validation sample for release quality."}\n',
        encoding="utf-8",
    )
    return checkpoint, tokenizer_path, validation


def test_release_cli_writes_exact_m10j_assets(tmp_path, capsys):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    private = cryptography.Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path = tmp_path / "publisher.private"
    private_path.write_bytes(raw_private)

    assets = tmp_path / "assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "1",
            "--source-origin", "vn97-training",
            "--source-license", "proprietary",
            "--assets-dir", str(assets),
            "--validation-input", str(validation),
            "--validation-format", "text",
            "--validation-sequence-length", "32",
            "--validation-batch-size", "1",
            "--min-validation-target-tokens", "1",
            "--max-validation-loss", "100",
            "--tile-rows", "4",
            "--tile-cols", "4",
        ]
    ) == 0

    assert {path.name for path in assets.iterdir()} == {
        "model.vn97cap1",
        "model.vn97sig1",
        "publisher.ed25519",
    }
    assert (assets / "publisher.ed25519").read_bytes() == (
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    assert all(
        path.read_bytes() != raw_private
        for path in assets.iterdir()
    )

    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "VN97BOOTREL2"
    assert report["publisher_key_id"] == "publisher.main"
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["package_sha256"]) == 64
    assert report["validation"]["target_tokens"] > 0
    assert report["validation"]["mean_loss"] <= 100
    assert len(report["validation"]["dataset_sha256"]) == 64


def test_release_cli_accepts_lowercase_hex_private_key(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    raw_private = bytes(range(32))
    private_path = tmp_path / "publisher.hex"
    private_path.write_text(raw_private.hex(), encoding="ascii")

    assets = tmp_path / "assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "2",
            "--source-origin", "vn97-training",
            "--source-license", "proprietary",
            "--assets-dir", str(assets),
            "--validation-input", str(validation),
            "--validation-format", "text",
            "--validation-sequence-length", "32",
            "--validation-batch-size", "1",
            "--min-validation-target-tokens", "1",
            "--max-validation-loss", "100",
        ]
    ) == 0
    assert (assets / "model.vn97cap1").is_file()


def test_release_cli_rejects_private_key_symlink(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    key = tmp_path / "real-key"
    key.write_bytes(bytes(range(32)))
    link = tmp_path / "key-link"
    link.symlink_to(key)

    with pytest.raises(ValueError, match="opened safely"):
        main(
            [
                "--checkpoint", str(checkpoint),
                "--tokenizer", str(tokenizer),
                "--private-key", str(link),
                "--key-id", "publisher.main",
                "--capability-version", "1",
                "--source-origin", "vn97-training",
                "--source-license", "proprietary",
                "--assets-dir", str(tmp_path / "assets"),
                "--validation-input", str(validation),
                "--validation-format", "text",
                "--validation-sequence-length", "32",
                "--validation-batch-size", "1",
                "--min-validation-target-tokens", "1",
                "--max-validation-loss", "100",
            ]
        )


def test_release_quality_gate_runs_before_private_key_read(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    assets = tmp_path / "assets"

    with pytest.raises(VN97ReleaseQualityError, match="loss"):
        main(
            [
                "--checkpoint", str(checkpoint),
                "--tokenizer", str(tokenizer),
                "--private-key", str(tmp_path / "missing-private-key"),
                "--key-id", "publisher.main",
                "--capability-version", "1",
                "--source-origin", "vn97-training",
                "--source-license", "proprietary",
                "--assets-dir", str(assets),
                "--validation-input", str(validation),
                "--validation-format", "text",
                "--validation-sequence-length", "32",
                "--validation-batch-size", "1",
                "--min-validation-target-tokens", "1",
                "--max-validation-loss", "0.000001",
            ]
        )
    assert not assets.exists()
