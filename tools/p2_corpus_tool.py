from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

TOOLS = {
    "source-fetch": "p2_source_fetch_cli",
    "corpus-intake": "p2_corpus_intake_cli",
    "corpus-seal": "production_corpus_cli",
    "corpus-bundle": "p2_corpus_bundle_cli",
}


def main(argv: list[str] | None = None) -> int:
    values = list(
        sys.argv[1:]
        if argv is None
        else argv
    )
    if not values or values[0] not in TOOLS:
        names = ", ".join(sorted(TOOLS))
        raise SystemExit(
            f"usage: p2_corpus_tool.py <{names}> [args...]"
        )

    package = types.ModuleType("vn97")
    package.__path__ = [str(SRC)]
    sys.modules["vn97"] = package

    module = importlib.import_module(
        f"vn97.{TOOLS[values[0]]}"
    )
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(
            f"P2 corpus tool has no main(): {values[0]}"
        )
    result = entry(values[1:])
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
