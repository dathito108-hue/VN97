from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
import sys

from .p2_source_fetch import (
    VN97P2FetchedSource,
    VN97P2SourceFetchError,
    build_fetch_receipt,
    is_allowed_download_url,
    load_fetch_definition,
)


_CHUNK_BYTES = 1024 * 1024


class _StrictRedirectHandler(
    urllib.request.HTTPRedirectHandler
):
    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        if not is_allowed_download_url(newurl):
            raise VN97P2SourceFetchError(
                "source redirect left allowed HTTPS hosts"
            )
        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )


def _atomic_write(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise VN97P2SourceFetchError(
            f"fetch output already exists: {path.name}"
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise VN97P2SourceFetchError(
                f"fetch output raced with another writer: {path.name}"
            ) from exc
        finally:
            temp.unlink(missing_ok=True)
        dir_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise VN97P2SourceFetchError(
            f"fetch output verification failed: {path.name}"
        )


def _jsonl_record_count(
    path: Path,
) -> int:
    count = 0
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        parse_constant=lambda raw: (
                            _ for _ in ()
                        ).throw(ValueError(raw)),
                    )
                except (
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    raise VN97P2SourceFetchError(
                        "downloaded source contains invalid JSONL "
                        f"at {path.name}:{line_number}"
                    ) from exc
                if not isinstance(value, dict):
                    raise VN97P2SourceFetchError(
                        "downloaded JSONL record must be an object "
                        f"at {path.name}:{line_number}"
                    )
                count += 1
    except UnicodeDecodeError as exc:
        raise VN97P2SourceFetchError(
            f"downloaded source is not strict UTF-8: {path.name}"
        ) from exc
    if count <= 0:
        raise VN97P2SourceFetchError(
            f"downloaded source has no records: {path.name}"
        )
    return count


def _download_one(
    source,
    *,
    output_dir: Path,
    timeout_seconds: float,
) -> VN97P2FetchedSource:
    if not is_allowed_download_url(
        source.download_url
    ):
        raise VN97P2SourceFetchError(
            "source URL is not allowed"
        )

    target = output_dir / source.filename
    if target.exists() or target.is_symlink():
        raise VN97P2SourceFetchError(
            f"source target already exists: {source.filename}"
        )

    opener = urllib.request.build_opener(
        _StrictRedirectHandler()
    )
    request = urllib.request.Request(
        source.download_url,
        method="GET",
        headers={
            "Accept-Encoding": "identity",
            "User-Agent":
                "VN97-P2-source-fetch/1",
        },
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{source.filename}.",
        suffix=".download",
        dir=output_dir,
    )
    temp = Path(temp_name)
    digest = hashlib.sha256()
    consumed = 0
    final_url = source.download_url
    try:
        try:
            response = opener.open(
                request,
                timeout=timeout_seconds,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
        ) as exc:
            raise VN97P2SourceFetchError(
                f"source download failed: {source.source_id}"
            ) from exc

        try:
            final_url = response.geturl()
            if not is_allowed_download_url(
                final_url
            ):
                raise VN97P2SourceFetchError(
                    "source final URL is outside allowed HTTPS hosts"
                )
            length = response.headers.get(
                "Content-Length"
            )
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise VN97P2SourceFetchError(
                        "source Content-Length is invalid"
                    ) from exc
                if (
                    declared <= 0
                    or declared > source.max_bytes
                ):
                    raise VN97P2SourceFetchError(
                        "source declared size is outside bound"
                    )

            with os.fdopen(
                fd,
                "wb",
                closefd=True,
            ) as output:
                while True:
                    chunk = response.read(
                        _CHUNK_BYTES
                    )
                    if not chunk:
                        break
                    consumed += len(chunk)
                    if consumed > source.max_bytes:
                        raise VN97P2SourceFetchError(
                            "source exceeded max_bytes while downloading"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            response.close()

        if consumed <= 0:
            raise VN97P2SourceFetchError(
                "source download was empty"
            )
        sha256 = digest.hexdigest()
        if (
            source.expected_bytes is not None
            and consumed != source.expected_bytes
        ):
            raise VN97P2SourceFetchError(
                f"source byte identity mismatch: {source.source_id}"
            )
        if (
            source.expected_sha256 is not None
            and sha256
            != source.expected_sha256
        ):
            raise VN97P2SourceFetchError(
                f"source SHA-256 mismatch: {source.source_id}"
            )

        os.close(fd) if False else None
        records = _jsonl_record_count(temp)
        if records != source.expected_records:
            raise VN97P2SourceFetchError(
                f"source record-count mismatch: {source.source_id}"
            )

        try:
            os.link(temp, target)
        except FileExistsError as exc:
            raise VN97P2SourceFetchError(
                f"source target raced with another writer: {source.filename}"
            ) from exc
        dir_fd = os.open(
            output_dir,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

        return VN97P2FetchedSource(
            source_id=source.source_id,
            revision=source.revision,
            filename=source.filename,
            bytes=consumed,
            records=records,
            sha256=sha256,
            final_url=final_url,
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        temp.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the exact pinned public sources for the VN97 P2 "
            "medium-pilot corpus and emit VN97P2FETCH1."
        )
    )
    parser.add_argument(
        "--definition",
        default=(
            "configs/"
            "p2-source-lock.vn97fetchdef1.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--approve-source-license",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help=(
            "Explicitly acknowledge one source license; repeat "
            "for every source in the pinned definition."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if (
        args.timeout_seconds <= 0
        or args.timeout_seconds > 600
    ):
        raise VN97P2SourceFetchError(
            "fetch timeout must be in (0, 600] seconds"
        )

    definition_path = Path(
        args.definition
    ).resolve(strict=True)
    definition = load_fetch_definition(
        definition_path
    )
    required = {
        item.source_id
        for item in definition.sources
    }
    approvals = set(
        args.approve_source_license
    )
    if approvals != required:
        missing = sorted(required - approvals)
        extra = sorted(approvals - required)
        raise VN97P2SourceFetchError(
            "source-license approvals must exactly match "
            f"pinned sources; missing={missing} extra={extra}"
        )

    output = Path(args.output_dir)
    if output.exists():
        if (
            output.is_symlink()
            or not output.is_dir()
        ):
            raise VN97P2SourceFetchError(
                "fetch output-dir must be a real directory"
            )
        if any(output.iterdir()):
            raise VN97P2SourceFetchError(
                "fetch output-dir must be new or empty"
            )
    else:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )
    if output.is_symlink():
        raise VN97P2SourceFetchError(
            "fetch output-dir must not be a symlink"
        )
    output = output.resolve(strict=True)

    fetched = tuple(
        _download_one(
            source,
            output_dir=output,
            timeout_seconds=float(
                args.timeout_seconds
            ),
        )
        for source in definition.sources
    )
    receipt = build_fetch_receipt(
        definition,
        fetched,
    )
    _atomic_write(
        output
        / "source-fetch.vn97p2fetch1.json",
        receipt,
    )

    print(
        "VN97P2FETCH1 "
        f"sources={len(fetched)} "
        f"definition_sha256="
        f"{definition.definition_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p2-source-fetch: {exc}",
            file=sys.stderr,
        )
        raise
