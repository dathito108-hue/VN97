from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


@dataclass(frozen=True)
class VN97CorpusChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(
                "chat role must be system, user, or assistant"
            )
        if (
            not isinstance(self.content, str)
            or not self.content
        ):
            raise ValueError(
                "chat content must be non-empty text"
            )


def read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"input could not be opened safely: {path}"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"input must be a regular file: {path}"
            )
        if info.st_size < 0 or info.st_size > max_bytes:
            raise ValueError(
                "input exceeds configured byte bound"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - len(out),
                ),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise ValueError(
                f"input changed while being read: {path}"
            )
        return bytes(out)
    finally:
        os.close(fd)


def atomic_write(
    path: Path,
    data: bytes,
) -> None:
    if path.is_symlink():
        raise ValueError(
            f"output target must not be a symlink: {path}"
        )
    parent = path.parent
    parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if parent.is_symlink():
        raise ValueError(
            f"output parent must not be a symlink: {parent}"
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if path.is_symlink():
            raise ValueError(
                f"output target became a symlink: {path}"
            )
        os.replace(
            temp_path,
            path,
        )
        dir_fd = os.open(
            parent.resolve(strict=True),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp_path.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise IOError(
            f"output post-write verification failed: {path}"
        )


def load_records(
    paths: list[Path],
    *,
    mode: str,
    max_input_bytes: int,
    max_examples: int,
) -> tuple[list[object], str]:
    if mode not in {"text", "chat"}:
        raise ValueError(
            "dataset mode must be text or chat"
        )
    if not paths:
        raise ValueError(
            "at least one input path is required"
        )
    total = 0
    records: list[object] = []
    digest = hashlib.sha256()

    for path in paths:
        remaining = max_input_bytes - total
        if remaining < 0:
            raise ValueError(
                "inputs exceed configured byte bound"
            )
        data = read_bounded_regular_file(
            path,
            max_bytes=remaining,
        )
        total += len(data)
        digest.update(
            len(data).to_bytes(8, "little")
        )
        digest.update(data)

        try:
            text = data.decode(
                "utf-8",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"input is not UTF-8: {path}"
            ) from exc

        for line_number, line in enumerate(
            text.split("\n"),
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
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "JSONL record must be object "
                    f"at {path}:{line_number}"
                )

            if mode == "text":
                if (
                    set(value) != {"text"}
                    or not isinstance(
                        value["text"],
                        str,
                    )
                    or not value["text"]
                ):
                    raise ValueError(
                        "text record must be exactly "
                        '{"text": non-empty string} '
                        f"at {path}:{line_number}"
                    )
                records.append(
                    value["text"]
                )
            else:
                if (
                    set(value) != {"messages"}
                    or not isinstance(
                        value["messages"],
                        list,
                    )
                ):
                    raise ValueError(
                        "chat record must be exactly "
                        '{"messages": [...]} '
                        f"at {path}:{line_number}"
                    )
                messages: list[
                    VN97CorpusChatMessage
                ] = []
                for raw in value["messages"]:
                    if (
                        not isinstance(raw, dict)
                        or set(raw)
                        != {"role", "content"}
                    ):
                        raise ValueError(
                            "chat message keys are invalid "
                            f"at {path}:{line_number}"
                        )
                    messages.append(
                        VN97CorpusChatMessage(
                            role=raw["role"],
                            content=raw["content"],
                        )
                    )
                if not messages:
                    raise ValueError(
                        "chat record has no messages "
                        f"at {path}:{line_number}"
                    )
                records.append(
                    tuple(messages)
                )

            if len(records) > max_examples:
                raise ValueError(
                    "record count exceeds configured example bound"
                )

    if not records:
        raise ValueError(
            "inputs contain no usable records"
        )
    return records, digest.hexdigest()
