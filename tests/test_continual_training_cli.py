import json

import pytest
import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97TokenizerPackage,
    save_deployment_checkpoint,
)
from vn97.continual_training_cli import main


def _fixture(tmp_path):
    tokenizer = VN97TokenizerPackage()
    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    ).eval()
    checkpoint = tmp_path / "parent.vn97ck1"
    save_deployment_checkpoint(model, checkpoint)
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer.to_bytes())

    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"text":"aaaa aaaa aaaa"}\n{"text":"bbbb bbbb bbbb"}\n',
        encoding="utf-8",
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        '{"text":"held out validation sample"}\n',
        encoding="utf-8",
    )
    return checkpoint, tokenizer_path, train, validation


def _args(checkpoint, tokenizer, train, validation, output):
    return [
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        "--input", str(train),
        "--validation-input", str(validation),
        "--format", "text",
        "--output-dir", str(output),
        "--sequence-length", "16",
        "--batch-size", "1",
        "--validation-batch-size", "1",
        "--epochs", "1",
        "--learning-rate", "0.001",
        "--min-validation-target-tokens", "1",
        "--max-validation-loss", "100",
        "--max-validation-loss-increase", "100",
        "--device", "cpu",
    ]


def test_finetune_cli_writes_only_after_quality_pass(tmp_path):
    checkpoint, tokenizer, train, validation = _fixture(tmp_path)
    output = tmp_path / "out"
    assert main(
        _args(checkpoint, tokenizer, train, validation, output)
    ) == 0
    assert (output / "model.vn97ck1").is_file()
    assert (output / "tokenizer.vn97tk1").is_file()
    report = json.loads(
        (output / "finetune-report.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "VN97FT1"
    assert len(report["parent_checkpoint_sha256"]) == 64
    assert len(report["output_checkpoint_sha256"]) == 64
    assert report["validation"]["dataset_sha256"] != report["training"]["dataset_sha256"]


def test_finetune_cli_rejects_train_validation_same_file(tmp_path):
    checkpoint, tokenizer, train, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="physically distinct"):
        main(
            _args(
                checkpoint,
                tokenizer,
                train,
                train,
                tmp_path / "out",
            )
        )


def test_finetune_cli_quality_failure_creates_no_output(tmp_path):
    checkpoint, tokenizer, train, validation = _fixture(tmp_path)
    output = tmp_path / "out"
    args = _args(checkpoint, tokenizer, train, validation, output)
    index = args.index("--max-validation-loss") + 1
    args[index] = "0.000001"
    with pytest.raises(Exception, match="loss"):
        main(args)
    assert not output.exists()
