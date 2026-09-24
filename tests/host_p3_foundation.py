from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.p2_corpus_intake",
    SRC / "p2_corpus_intake.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p2_corpus_intake.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p2_corpus_intake"] = module
spec.loader.exec_module(module)


def main() -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": f"Câu hỏi {index}"},
                {"role": "assistant", "content": f"Câu trả lời {index}"},
            ]
        }
        for index in range(20)
    ]

    default = module.prepare_source(
        "vi-dialogue-hoanghai2110",
        records,
        raw_sha256="a" * 64,
        raw_bytes=1000,
    )
    expanded = module.prepare_source(
        "vi-dialogue-hoanghai2110",
        records,
        raw_sha256="a" * 64,
        raw_bytes=1000,
        max_records=17,
    )
    assert default.accepted_records == 20
    assert expanded.accepted_records == 17
    assert expanded.rejected_records == 3

    campaign = json.loads(
        (
            ROOT
            / "configs"
            / "p3-language-production.vn97campdef1.json"
        ).read_text(encoding="utf-8")
    )
    assert campaign["schema"] == "VN97CAMPDEF1"
    assert len(campaign["candidates"]) == 4
    assert campaign["candidates"][0]["d_model"] == 256
    assert campaign["candidates"][-1]["d_model"] == 448
    ids = {
        (
            item["d_model"],
            item["n_layers"],
            item["d_state"],
            item["embedding_rank"],
            item["seed"],
        )
        for item in campaign["candidates"]
    }
    assert len(ids) == 4

    cli = (
        ROOT / "src" / "vn97" / "p3_corpus_intake_cli.py"
    ).read_text(encoding="utf-8")
    for source_id, limit in (
        ("vi-dialogue-hoanghai2110", 2009),
        ("databricks-dolly-15k", 15011),
        ("openai-gsm8k-train", 7473),
    ):
        assert source_id in cli
        assert str(limit) in cli

    result_doc = (
        ROOT
        / "docs"
        / "P2_REAL_PILOT_RESULT_P3_ENTRY.md"
    ).read_text(encoding="utf-8")
    assert "e0509769f3f819516c7e41c28769053673f634e41051d8e469b95bd8421db5a2" in result_doc
    assert "1edf7d43c019b303676babb7722693f278f6a102a54ec80848664e4f93761675" in result_doc

    print("VN97 P2 closeout / P3 foundation host contract PASS")


if __name__ == "__main__":
    main()
