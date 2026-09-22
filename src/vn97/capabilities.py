from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
import ssl
import stat
from typing import Callable, Iterable, Iterator, Protocol, Sequence
from urllib.parse import urlsplit

from .authority import (
    ActionOutcome,
    AuthorizedAction,
    CapabilityDescriptor,
    ExternalActionRequest,
    TypedCapabilityRegistry,
)


CAP_FILE_READ = "file.read"
CAP_FILE_WRITE = "file.write"
CAP_WEB_FETCH = "web.fetch"
CAP_APP_LAUNCH = "app.launch"
CAP_CLIPBOARD_WRITE = "device.clipboard.write"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ANDROID_PACKAGE_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+$"
)
_MAX_ACTION_RESULT_UTF8_BYTES = 64 * 1024


class CapabilityImplementationError(RuntimeError):
    pass


class CapabilityPayloadError(CapabilityImplementationError):
    pass


class FileScopeError(CapabilityImplementationError):
    pass


class NetworkScopeError(CapabilityImplementationError):
    pass


class PlatformCapabilityError(CapabilityImplementationError):
    pass


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bounded_result_json(value: object) -> str:
    result = _canonical_json(value)
    if len(result.encode("utf-8")) > _MAX_ACTION_RESULT_UTF8_BYTES:
        raise CapabilityImplementationError("capability result exceeds byte limit")
    return result


def _exact_payload(request: ExternalActionRequest, keys: set[str]) -> dict[str, object]:
    value = request.payload_object()
    if set(value) != keys:
        raise CapabilityPayloadError(
            f"{request.capability_id} payload keys must be exactly {sorted(keys)}"
        )
    return value


def _optional_payload(
    request: ExternalActionRequest,
    *,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    value = request.payload_object()
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise CapabilityPayloadError(
            f"{request.capability_id} payload keys violate capability schema"
        )
    return value


def _require_text(value: object, *, label: str, max_utf8_bytes: int) -> str:
    if not isinstance(value, str):
        raise CapabilityPayloadError(f"{label} must be a string")
    if len(value.encode("utf-8")) > max_utf8_bytes:
        raise CapabilityPayloadError(f"{label} exceeds byte limit")
    return value


def _scope_value(request: ExternalActionRequest, key: str) -> str:
    value = request.scope.as_dict().get(key)
    if value is None:
        raise CapabilityPayloadError(f"missing scope key: {key}")
    return value


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if not relative_path or len(relative_path.encode("utf-8")) > 2048:
        raise FileScopeError("file path must be non-empty and bounded")
    if "\\" in relative_path or "\x00" in relative_path:
        raise FileScopeError("file path contains unsupported characters")
    if relative_path.startswith("/") or relative_path.endswith("/"):
        raise FileScopeError("file path must be canonical relative POSIX")
    parts = tuple(relative_path.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise FileScopeError("path traversal is not allowed")
    return parts


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    view = memoryview(data)
    while offset < len(view):
        count = os.write(fd, view[offset:])
        if count <= 0:
            raise OSError("short file write")
        offset += count


def _bounded_fd_sha256(fd: int, *, max_bytes: int) -> tuple[str, int]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    seen = 0
    while True:
        chunk = os.read(fd, min(64 * 1024, max_bytes - seen + 1))
        if not chunk:
            break
        seen += len(chunk)
        if seen > max_bytes:
            raise FileScopeError("file exceeds configured byte limit")
        digest.update(chunk)
    return digest.hexdigest(), seen


@dataclass(frozen=True)
class FileRoot:
    root_id: str
    path: Path
    readable: bool = True
    writable: bool = True
    max_read_bytes: int = 64 * 1024
    max_write_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not self.root_id or len(self.root_id.encode("utf-8")) > 128:
            raise ValueError("root_id must be non-empty and bounded")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        if any(char not in allowed for char in self.root_id):
            raise ValueError("root_id contains unsupported characters")
        path = Path(self.path)
        if not path.is_absolute() or not path.exists() or not path.is_dir():
            raise ValueError("file root must be an existing absolute directory")
        if path.is_symlink():
            raise ValueError("file root must not be a symlink")
        if self.max_read_bytes <= 0 or self.max_write_bytes <= 0:
            raise ValueError("file root byte limits must be positive")
        object.__setattr__(self, "path", path)


class ConfinedFileStore:
    """Linux/Android root-confined UTF-8 file access with no-follow path traversal."""

    def __init__(self, roots: Iterable[FileRoot]) -> None:
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise RuntimeError("confined files require O_NOFOLLOW and O_DIRECTORY")
        self._roots: dict[str, FileRoot] = {}
        for root in roots:
            if root.root_id in self._roots:
                raise ValueError("duplicate file root_id")
            self._roots[root.root_id] = root
        if not self._roots:
            raise ValueError("at least one file root is required")

    def root(self, root_id: str) -> FileRoot:
        root = self._roots.get(root_id)
        if root is None:
            raise FileScopeError("unknown file root")
        return root

    @contextmanager
    def _open_parent(
        self,
        root: FileRoot,
        relative_path: str,
    ) -> Iterator[tuple[int, str]]:
        parts = _relative_parts(relative_path)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        opened: list[int] = []
        try:
            current = os.open(root.path, flags)
            opened.append(current)
            for part in parts[:-1]:
                current = os.open(part, flags, dir_fd=current)
                opened.append(current)
            yield current, parts[-1]
        except OSError as exc:
            raise FileScopeError("unsafe or unavailable file path") from exc
        finally:
            for fd in reversed(opened):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def read_text(self, root_id: str, relative_path: str) -> tuple[str, str, int]:
        root = self.root(root_id)
        if not root.readable:
            raise FileScopeError("file root is not readable")
        with self._open_parent(root, relative_path) as (parent_fd, leaf):
            try:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            except OSError as exc:
                raise FileScopeError("read target is unavailable") from exc
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise FileScopeError("read target is not a regular file")
                digest, size = _bounded_fd_sha256(fd, max_bytes=root.max_read_bytes)
                os.lseek(fd, 0, os.SEEK_SET)
                data = bytearray()
                while len(data) < size:
                    chunk = os.read(fd, min(64 * 1024, size - len(data)))
                    if not chunk:
                        break
                    data.extend(chunk)
            finally:
                os.close(fd)
        try:
            text = bytes(data).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise FileScopeError("read target is not strict UTF-8") from exc
        return text, digest, size

    def atomic_write_text(
        self,
        root_id: str,
        relative_path: str,
        text: str,
        *,
        expected_sha256: str | None,
    ) -> tuple[str, int, bool]:
        root = self.root(root_id)
        if not root.writable:
            raise FileScopeError("file root is not writable")
        data = text.encode("utf-8")
        if len(data) > root.max_write_bytes:
            raise FileScopeError("write payload exceeds configured byte limit")
        if expected_sha256 is not None and not _SHA256_RE.fullmatch(expected_sha256):
            raise FileScopeError("expected_sha256 must be lowercase SHA-256 hex")

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Linux/Android production target
            raise FileScopeError("platform lacks advisory file locking") from exc

        with self._open_parent(root, relative_path) as (parent_fd, leaf):
            fcntl.flock(parent_fd, fcntl.LOCK_EX)
            temp_name = f".vn97-{secrets.token_hex(12)}.tmp"
            temp_created = False
            try:
                exists = False
                current_sha = ""
                try:
                    current_fd = os.open(
                        leaf,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    current_fd = -1
                except OSError as exc:
                    raise FileScopeError("write target is unsafe or unavailable") from exc
                if current_fd >= 0:
                    exists = True
                    try:
                        info = os.fstat(current_fd)
                        if not stat.S_ISREG(info.st_mode):
                            raise FileScopeError("write target is not a regular file")
                        current_sha, _ = _bounded_fd_sha256(
                            current_fd,
                            max_bytes=root.max_write_bytes,
                        )
                    finally:
                        os.close(current_fd)

                if exists:
                    if expected_sha256 is None:
                        raise FileScopeError("replace requires expected_sha256")
                    if current_sha != expected_sha256:
                        raise FileScopeError("replace compare-and-swap hash mismatch")
                elif expected_sha256 is not None:
                    raise FileScopeError("create must not provide expected_sha256")

                temp_fd = os.open(
                    temp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                temp_created = True
                try:
                    _write_all(temp_fd, data)
                    os.fsync(temp_fd)
                finally:
                    os.close(temp_fd)

                if exists:
                    # Recheck while the VN97 directory lock is held, then atomically replace.
                    check_fd = os.open(
                        leaf,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=parent_fd,
                    )
                    try:
                        check_sha, _ = _bounded_fd_sha256(
                            check_fd,
                            max_bytes=root.max_write_bytes,
                        )
                    finally:
                        os.close(check_fd)
                    if check_sha != expected_sha256:
                        raise FileScopeError("replace target changed during write")
                    os.replace(
                        temp_name,
                        leaf,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                    )
                    temp_created = False
                else:
                    try:
                        os.link(
                            temp_name,
                            leaf,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise FileScopeError("create target appeared during write") from exc
                    os.unlink(temp_name, dir_fd=parent_fd)
                    temp_created = False
                os.fsync(parent_fd)
            finally:
                if temp_created:
                    try:
                        os.unlink(temp_name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        pass
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
        return _sha256_bytes(data), len(data), exists


@dataclass(frozen=True)
class HttpsResponse:
    status: int
    final_url: str
    content_type: str
    body: bytes


Resolver = Callable[[str, int], Sequence[str]]
Transport = Callable[[str, str, str, float, int], HttpsResponse]


def _default_resolver(host: str, port: int) -> Sequence[str]:
    infos = socket.getaddrinfo(
        host,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    return tuple(dict.fromkeys(info[4][0].split("%", 1)[0] for info in infos))


def _default_https_transport(
    url: str,
    host: str,
    address: str,
    timeout_s: float,
    max_bytes: int,
) -> HttpsResponse:
    parts = urlsplit(url)
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    context = ssl.create_default_context()
    host_header = f"[{host}]" if ":" in host else host
    raw = socket.create_connection((address, 443), timeout=timeout_s)
    try:
        tls = context.wrap_socket(raw, server_hostname=host)
        raw = None
        try:
            request = (
                f"GET {target} HTTP/1.1\r\n"
                f"Host: {host_header}\r\n"
                "User-Agent: VN97/0.1 sovereign-fetch\r\n"
                "Accept: text/plain, text/html, application/json, application/*+json\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            tls.sendall(request)
            response = http.client.HTTPResponse(tls)
            response.begin()
            if 300 <= response.status < 400:
                raise NetworkScopeError("HTTPS redirects are disabled")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise NetworkScopeError("HTTPS response exceeds byte limit")
            return HttpsResponse(
                status=int(response.status),
                final_url=url,
                content_type=response.getheader("Content-Type", ""),
                body=body,
            )
        finally:
            tls.close()
    except NetworkScopeError:
        raise
    except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
        raise NetworkScopeError("HTTPS transport failed") from exc
    finally:
        if raw is not None:
            raw.close()


class SovereignHttpsFetcher:
    def __init__(
        self,
        *,
        resolver: Resolver = _default_resolver,
        transport: Transport = _default_https_transport,
        timeout_s: float = 10.0,
        max_response_bytes: int = 64 * 1024,
        max_result_utf8_bytes: int = 64 * 1024,
    ) -> None:
        if timeout_s <= 0 or timeout_s > 60:
            raise ValueError("HTTPS timeout must be in (0, 60]")
        if max_response_bytes <= 0 or max_response_bytes > 1024 * 1024:
            raise ValueError("HTTPS response limit must be in (0, 1 MiB]")
        if max_result_utf8_bytes <= 0:
            raise ValueError("HTTPS result limit must be positive")
        self._resolver = resolver
        self._transport = transport
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)
        self.max_result_utf8_bytes = int(max_result_utf8_bytes)

    def validate_url(self, url: str) -> tuple[str, int, tuple[str, ...]]:
        if not isinstance(url, str) or not url or len(url.encode("utf-8")) > 4096:
            raise NetworkScopeError("HTTPS URL must be non-empty and bounded")
        if not url.isascii() or any(char.isspace() or ord(char) == 0x7F for char in url):
            raise NetworkScopeError("HTTPS URL must be ASCII/percent-encoded without whitespace")
        try:
            parts = urlsplit(url)
            port = parts.port or 443
        except ValueError as exc:
            raise NetworkScopeError("HTTPS URL is malformed") from exc
        if parts.scheme.lower() != "https":
            raise NetworkScopeError("only HTTPS is allowed")
        if not parts.hostname:
            raise NetworkScopeError("HTTPS URL requires a host")
        if parts.username is not None or parts.password is not None:
            raise NetworkScopeError("URL credentials are forbidden")
        if parts.fragment:
            raise NetworkScopeError("URL fragments are forbidden")
        if port != 443:
            raise NetworkScopeError("only HTTPS port 443 is allowed")
        host = parts.hostname.rstrip(".").lower()
        if not host.isascii() or host == "localhost" or not host:
            raise NetworkScopeError("local or non-ASCII HTTPS hosts are not allowed")
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            addresses = self._resolver(host, port)
        else:
            addresses = (str(literal),)
        if not addresses:
            raise NetworkScopeError("HTTPS host did not resolve")
        normalized: list[str] = []
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError as exc:
                raise NetworkScopeError("resolver returned an invalid IP address") from exc
            if not ip.is_global:
                raise NetworkScopeError("HTTPS host resolves to a non-global IP address")
            normalized.append(str(ip))
        return host, port, tuple(sorted(set(normalized)))

    def fetch_text(self, url: str) -> dict[str, object]:
        host, _port, addresses = self.validate_url(url)
        response = self._transport(
            url,
            host,
            addresses[0],
            self.timeout_s,
            self.max_response_bytes,
        )
        if response.final_url != url:
            raise NetworkScopeError("HTTPS transport changed URL or followed a redirect")
        if not 200 <= response.status < 300:
            raise NetworkScopeError("HTTPS response status is not successful")
        if len(response.body) > self.max_response_bytes:
            raise NetworkScopeError("HTTPS response exceeds byte limit")
        media_type = response.content_type.split(";", 1)[0].strip().lower()
        if not (
            media_type.startswith("text/")
            or media_type == "application/json"
            or media_type.endswith("+json")
        ):
            raise NetworkScopeError("HTTPS capability accepts text/JSON responses only")
        charset = "utf-8"
        for parameter in response.content_type.split(";")[1:]:
            key, sep, value = parameter.strip().partition("=")
            if sep and key.lower() == "charset":
                charset = value.strip().strip('"').lower()
        if charset not in {"utf-8", "utf8", "us-ascii", "ascii"}:
            raise NetworkScopeError("HTTPS response charset is unsupported")
        try:
            body_text = response.body.decode(
                "utf-8" if charset in {"utf-8", "utf8"} else "ascii",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            raise NetworkScopeError("HTTPS response text decoding failed") from exc
        result = {
            "status": response.status,
            "url": response.final_url,
            "content_type": media_type,
            "body": body_text,
            "sha256": _sha256_bytes(response.body),
            "utf8_bytes": len(response.body),
        }
        if len(_canonical_json(result).encode("utf-8")) > self.max_result_utf8_bytes:
            raise NetworkScopeError("HTTPS result exceeds planner result byte limit")
        return result


class AppDeviceAdapter(Protocol):
    def launch_package(self, package: str) -> str:
        ...

    def write_clipboard(self, text: str) -> str:
        ...


@dataclass(frozen=True)
class CapabilityPackConfig:
    file_store: ConfinedFileStore | None = None
    https_fetcher: SovereignHttpsFetcher | None = None
    platform: AppDeviceAdapter | None = None


def _validate_file_read(request: ExternalActionRequest) -> None:
    _exact_payload(request, set())
    _scope_value(request, "root")
    _relative_parts(_scope_value(request, "path"))


def _validate_file_write(request: ExternalActionRequest) -> None:
    payload = _optional_payload(
        request,
        required={"text"},
        optional={"expected_sha256"},
    )
    _require_text(payload["text"], label="file text", max_utf8_bytes=256 * 1024)
    expected = payload.get("expected_sha256")
    if expected is not None and (
        not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected)
    ):
        raise CapabilityPayloadError("expected_sha256 must be lowercase SHA-256 hex")
    _scope_value(request, "root")
    _relative_parts(_scope_value(request, "path"))


def _validate_web_fetch(request: ExternalActionRequest) -> None:
    _exact_payload(request, set())
    _scope_value(request, "url")


def _validate_app_launch(request: ExternalActionRequest) -> None:
    _exact_payload(request, set())
    package = _scope_value(request, "package")
    if not package.isascii() or not _ANDROID_PACKAGE_RE.fullmatch(package):
        raise CapabilityPayloadError("package must be an ASCII Android package name")


def _validate_clipboard_write(request: ExternalActionRequest) -> None:
    payload = _exact_payload(request, {"text"})
    _require_text(payload["text"], label="clipboard text", max_utf8_bytes=16 * 1024)
    if _scope_value(request, "channel") != "system":
        raise CapabilityPayloadError("clipboard scope channel must equal 'system'")


def register_m6b_capabilities(
    registry: TypedCapabilityRegistry,
    config: CapabilityPackConfig,
) -> tuple[str, ...]:
    """Register concrete M6B handlers behind the existing M6A authority registry."""
    registered: list[str] = []

    if config.file_store is not None:
        store = config.file_store

        def file_read(action: AuthorizedAction) -> ActionOutcome:
            request = action.request
            root_id = _scope_value(request, "root")
            relative = _scope_value(request, "path")
            text, digest, byte_count = store.read_text(root_id, relative)
            return ActionOutcome(
                True,
                _bounded_result_json(
                    {
                        "root": root_id,
                        "path": relative,
                        "text": text,
                        "sha256": digest,
                        "utf8_bytes": byte_count,
                    }
                ),
            )

        def file_write(action: AuthorizedAction) -> ActionOutcome:
            request = action.request
            payload = request.payload_object()
            root_id = _scope_value(request, "root")
            relative = _scope_value(request, "path")
            digest, byte_count, replaced = store.atomic_write_text(
                root_id,
                relative,
                str(payload["text"]),
                expected_sha256=(
                    None
                    if "expected_sha256" not in payload
                    else str(payload["expected_sha256"])
                ),
            )
            return ActionOutcome(
                True,
                _bounded_result_json(
                    {
                        "root": root_id,
                        "path": relative,
                        "sha256": digest,
                        "utf8_bytes": byte_count,
                        "replaced": replaced,
                    }
                ),
            )

        registry.register(
            CapabilityDescriptor(
                CAP_FILE_READ,
                frozenset({"root", "path"}),
                approval_required=True,
                max_payload_utf8_bytes=2,
                max_lease_uses=1,
            ),
            file_read,
            validator=_validate_file_read,
        )
        registry.register(
            CapabilityDescriptor(
                CAP_FILE_WRITE,
                frozenset({"root", "path"}),
                approval_required=True,
                max_payload_utf8_bytes=300 * 1024,
                max_lease_uses=1,
            ),
            file_write,
            validator=_validate_file_write,
        )
        registered.extend((CAP_FILE_READ, CAP_FILE_WRITE))

    if config.https_fetcher is not None:
        fetcher = config.https_fetcher

        def web_fetch(action: AuthorizedAction) -> ActionOutcome:
            url = _scope_value(action.request, "url")
            return ActionOutcome(True, _bounded_result_json(fetcher.fetch_text(url)))

        def validate_web(request: ExternalActionRequest) -> None:
            _validate_web_fetch(request)
            fetcher.validate_url(_scope_value(request, "url"))

        registry.register(
            CapabilityDescriptor(
                CAP_WEB_FETCH,
                frozenset({"url"}),
                approval_required=True,
                max_payload_utf8_bytes=2,
                max_lease_uses=1,
            ),
            web_fetch,
            validator=validate_web,
        )
        registered.append(CAP_WEB_FETCH)

    if config.platform is not None:
        platform = config.platform

        def app_launch(action: AuthorizedAction) -> ActionOutcome:
            package = _scope_value(action.request, "package")
            try:
                result = platform.launch_package(package)
            except Exception as exc:
                raise PlatformCapabilityError("app adapter failed") from exc
            if not isinstance(result, str) or not result:
                raise PlatformCapabilityError("app adapter returned an invalid result")
            if len(result.encode("utf-8")) > 4096:
                raise PlatformCapabilityError("app adapter result exceeds byte limit")
            return ActionOutcome(True, result)

        def clipboard_write(action: AuthorizedAction) -> ActionOutcome:
            payload = action.request.payload_object()
            try:
                result = platform.write_clipboard(str(payload["text"]))
            except Exception as exc:
                raise PlatformCapabilityError("clipboard adapter failed") from exc
            if not isinstance(result, str) or not result:
                raise PlatformCapabilityError("clipboard adapter returned an invalid result")
            if len(result.encode("utf-8")) > 4096:
                raise PlatformCapabilityError("clipboard adapter result exceeds byte limit")
            return ActionOutcome(True, result)

        registry.register(
            CapabilityDescriptor(
                CAP_APP_LAUNCH,
                frozenset({"package"}),
                approval_required=True,
                max_payload_utf8_bytes=2,
                max_lease_uses=1,
            ),
            app_launch,
            validator=_validate_app_launch,
        )
        registry.register(
            CapabilityDescriptor(
                CAP_CLIPBOARD_WRITE,
                frozenset({"channel"}),
                approval_required=True,
                max_payload_utf8_bytes=18 * 1024,
                max_lease_uses=1,
            ),
            clipboard_write,
            validator=_validate_clipboard_write,
        )
        registered.extend((CAP_APP_LAUNCH, CAP_CLIPBOARD_WRITE))

    return tuple(registered)
