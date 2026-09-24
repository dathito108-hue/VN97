from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys

from .production_corpus import (
    VN97ProductionCorpusError,
    prepare_corpus,
)
from .corpus_io import (
    atomic_write as _atomic_write,
    load_records as _load_records,
    read_bounded_regular_file as _read_bounded_regular_file,
)


_MAX_DEFINITION_BYTES = 1024 * 1024
_MAX_SOURCES = 512
_MAX_PATH_BYTES = 4096


def _strict_json(data: bytes) -> object:
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(raw)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VN97ProductionCorpusError(
            "corpus definition must be strict UTF-8 JSON"
        ) from exc
    if duplicates:
        raise VN97ProductionCorpusError(
            "corpus definition contains duplicate object keys"
        )
    return value


def _load_definition(path: Path) -> tuple[str, list[dict[str, object]]]:
    raw = _read_bounded_regular_file(
        path,
        max_bytes=_MAX_DEFINITION_BYTES,
    )
    root = _strict_json(raw)
    if (
        not isinstance(root, dict)
        or set(root) != {"profile_id", "schema", "sources"}
        or root["schema"] != "VN97CORPUSDEF1"
        or not isinstance(root["profile_id"], str)
        or not isinstance(root["sources"], list)
    ):
        raise VN97ProductionCorpusError(
            "corpus definition schema is invalid"
        )
    sources = root["sources"]
    if not 3 <= len(sources) <= _MAX_SOURCES:
        raise VN97ProductionCorpusError(
            "corpus definition must contain 3..512 sources"
        )
    required = {
        "license",
        "license_approved",
        "mode",
        "origin",
        "path",
        "source_id",
        "split",
    }
    parsed: list[dict[str, object]] = []
    for item in sources:
        if not isinstance(item, dict) or set(item) != required:
            raise VN97ProductionCorpusError(
                "corpus source fields are invalid"
            )
        if item["license_approved"] is not True:
            raise VN97ProductionCorpusError(
                "every corpus source requires explicit license_approved=true"
            )
        for key in (
            "license",
            "mode",
            "origin",
            "path",
            "source_id",
            "split",
        ):
            if not isinstance(item[key], str):
                raise VN97ProductionCorpusError(
                    f"corpus source {key} must be a string"
                )
        if not item["source_id"] or not item["origin"] or not item["license"]:
            raise VN97ProductionCorpusError(
                "corpus source identity/origin/license must be non-empty"
            )
        if item["mode"] not in {"text", "chat"}:
            raise VN97ProductionCorpusError(
                "corpus source mode must be text or chat"
            )
        if item["split"] not in {"training", "validation", "release"}:
            raise VN97ProductionCorpusError(
                "corpus source split is invalid"
            )
        path_bytes = item["path"].encode("utf-8")
        if not path_bytes or len(path_bytes) > _MAX_PATH_BYTES:
            raise VN97ProductionCorpusError(
                "corpus source path is outside bounds"
            )
        parsed.append(item)
    if not root["profile_id"] or len(root["profile_id"].encode("utf-8")) > 128:
        raise VN97ProductionCorpusError(
            "corpus profile_id must be 1..128 UTF-8 bytes"
        )
    return root["profile_id"], parsed


def _resolve_source_path(definition: Path, raw: str) -> Path:
    source = Path(raw)
    if not source.is_absolute():
        source = definition.parent / source
    try:
        info = os.stat(source, follow_symlinks=False)
    except OSError as exc:
        raise VN97ProductionCorpusError(
            f"corpus source is unavailable: {source}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise VN97ProductionCorpusError(
            f"corpus source must be a regular non-symlink file: {source}"
        )
    return source.resolve(strict=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal licensed local data into deterministic VN97CORPUS1 "
            "training/validation/release JSONL splits."
        )
    )
    parser.add_argument("--definition", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-source-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-source-examples",
        type=int,
        default=1_000_000,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_source_bytes <= 0 or args.max_source_examples <= 0:
        raise VN97ProductionCorpusError(
            "corpus source bounds must be positive"
        )

    definition = Path(args.definition).resolve(strict=True)
    profile_id, source_defs = _load_definition(definition)

    identities: set[tuple[int, int]] = set()
    rows = []
    for item in source_defs:
        source_path = _resolve_source_path(
            definition,
            str(item["path"]),
        )
        info = os.stat(source_path, follow_symlinks=False)
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in identities:
            raise VN97ProductionCorpusError(
                "corpus definition references the same physical file twice"
            )
        identities.add(identity)

        raw = _read_bounded_regular_file(
            source_path,
            max_bytes=args.max_source_bytes,
        )
        records, _ = _load_records(
            [source_path],
            mode=str(item["mode"]),
            max_input_bytes=args.max_source_bytes,
            max_examples=args.max_source_examples,
        )
        rows.append(
            (
                str(item["source_id"]),
                str(item["origin"]),
                str(item["license"]),
                str(item["split"]),
                str(item["mode"]),
                raw,
                records,
            )
        )

    manifest, prepared = prepare_corpus(
        rows,
        profile_id=profile_id,
    )

    output = Path(args.output_dir)
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise VN97ProductionCorpusError(
                "corpus output-dir must be a real directory"
            )
        if any(output.iterdir()):
            raise VN97ProductionCorpusError(
                "corpus output-dir must be new or empty"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    if output.is_symlink():
        raise VN97ProductionCorpusError(
            "corpus output-dir must not be a symlink"
        )

    filenames = {
        "training": "training.jsonl",
        "validation": "validation.jsonl",
        "release": "release.jsonl",
    }
    for split, filename in filenames.items():
        _atomic_write(
            output / filename,
            prepared[split].bytes,
        )
    _atomic_write(
        output / "corpus.vn97corpus1.json",
        manifest.to_bytes(),
    )

    print(
        "VN97CORPUS1 "
        f"id={manifest.manifest_id} "
        f"training={manifest.training.records} "
        f"validation={manifest.validation.records} "
        f"release={manifest.release.records}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-corpus-seal: {exc}", file=sys.stderr)
        raise
