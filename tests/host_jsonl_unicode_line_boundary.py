from __future__ import annotations

import importlib
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

corpus_io = importlib.import_module("vn97.corpus_io")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "unicode-lines.jsonl"
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Giữ U+2028 ở trong chuỗi: A B",
                },
                {
                    "role": "assistant",
                    "content": "Và U+2029 cũng phải nguyên vẹn: C D",
                },
            ]
        }
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        assert " " in text
        assert " " in text
        path.write_text(text, encoding="utf-8")

        records, _ = corpus_io.load_records(
            [path],
            mode="chat",
            max_input_bytes=1024 * 1024,
            max_examples=10,
        )
        assert len(records) == 1
        messages = records[0]
        assert messages[0].content.endswith("A B")
        assert messages[1].content.endswith("C D")

    for relative in (
        "corpus_io.py",
        "training_cli.py",
        "p2_corpus_intake_cli.py",
        "p3_corpus_intake_cli.py",
    ):
        source = (SRC / relative).read_text(encoding="utf-8")
        assert 'text.split("\\n")' in source
        assert "text.splitlines()" not in source

    bundle = (SRC / "p2_corpus_bundle.py").read_text(encoding="utf-8")
    assert 'data.split(b"\\n")' in bundle

    handoff = (SRC / "p2_pilot_handoff.py").read_text(encoding="utf-8")
    assert '.read_bytes().split(b"\\n")' in handoff

    print("VN97 JSONL Unicode line-boundary regression PASS")


if __name__ == "__main__":
    main()
