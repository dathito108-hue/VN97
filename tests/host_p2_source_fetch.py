from __future__ import annotations

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
    "vn97.p2_source_fetch",
    SRC / "p2_source_fetch.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p2_source_fetch.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p2_source_fetch"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def main() -> None:
    assert module.is_allowed_download_url(
        "https://huggingface.co/datasets/x/y/resolve/"
        + "a" * 40
        + "/data.jsonl"
    )
    assert module.is_allowed_download_url(
        "https://raw.githubusercontent.com/o/r/"
        + "b" * 40
        + "/data.jsonl"
    )
    assert module.is_allowed_download_url(
        "https://cas-bridge.xethub.hf.co/x"
    )
    assert not module.is_allowed_download_url(
        "http://huggingface.co/data.jsonl"
    )
    assert not module.is_allowed_download_url(
        "https://user:pass@huggingface.co/data.jsonl"
    )
    assert not module.is_allowed_download_url(
        "https://huggingface.co/data.jsonl#fragment"
    )
    assert not module.is_allowed_download_url(
        "https://huggingface.co.evil.invalid/data.jsonl"
    )

    lock = ROOT / "configs" / "p2-source-lock.vn97fetchdef1.json"
    definition = module.load_fetch_definition(lock)
    assert len(definition.sources) == 3
    assert [item.source_id for item in definition.sources] == [
        "vi-dialogue-hoanghai2110",
        "databricks-dolly-15k",
        "openai-gsm8k-train",
    ]
    assert definition.sources[0].revision == (
        "321a852edc7e9d36e8456fb7d7df583b520645a9"
    )
    assert definition.sources[0].expected_records == 2009
    assert definition.sources[1].expected_sha256 == (
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    )
    assert definition.sources[1].expected_bytes == 13085339
    assert definition.sources[2].expected_sha256 == (
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
    )
    assert definition.sources[2].expected_bytes == 4166206
    assert definition.sources[2].expected_records == 7473

    fetched = tuple(
        module.VN97P2FetchedSource(
            source_id=item.source_id,
            revision=item.revision,
            filename=item.filename,
            bytes=100 + index,
            records=item.expected_records,
            sha256=(str(index + 1) * 64)[:64],
            final_url=(
                "https://huggingface.co/file"
                if "huggingface.co" in item.download_url
                else "https://raw.githubusercontent.com/file"
            ),
        )
        for index, item in enumerate(definition.sources)
    )
    receipt = json.loads(
        module.build_fetch_receipt(
            definition,
            fetched,
        ).decode("utf-8")
    )
    assert receipt["schema"] == "VN97P2FETCH1"
    assert len(receipt["receipt_id"]) == 64
    assert receipt["definition_sha256"] == definition.definition_sha256
    assert len(receipt["sources"]) == 3

    expect_failure(
        "receipt source order mismatch",
        lambda: module.build_fetch_receipt(
            definition,
            tuple(reversed(fetched)),
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        bad = json.loads(lock.read_text(encoding="utf-8"))
        bad["sources"][0]["download_url"] = (
            "https://evil.invalid/"
            + bad["sources"][0]["revision"]
            + "/data.jsonl"
        )
        path.write_text(
            json.dumps(bad),
            encoding="utf-8",
        )
        expect_failure(
            "untrusted host",
            lambda: module.load_fetch_definition(path),
        )

    print(
        "VN97 P2 pinned source fetch contract PASS "
        f"definition={definition.definition_sha256}"
    )


if __name__ == "__main__":
    main()
