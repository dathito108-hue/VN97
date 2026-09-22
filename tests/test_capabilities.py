from pathlib import Path
import json
import os

import pytest

from vn97.authority import (
    AuthorizedAction,
    CapabilityLease,
    CapabilityScope,
    ExternalActionRequest,
    TypedCapabilityRegistry,
)
from vn97.capabilities import (
    CAP_APP_LAUNCH,
    CAP_CLIPBOARD_WRITE,
    CAP_FILE_READ,
    CAP_FILE_WRITE,
    CAP_WEB_FETCH,
    CapabilityPackConfig,
    CapabilityPayloadError,
    ConfinedFileStore,
    FileRoot,
    FileScopeError,
    HttpsResponse,
    NetworkScopeError,
    SovereignHttpsFetcher,
    register_m6b_capabilities,
)


def req(capability, scope, payload=None):
    return ExternalActionRequest.create(
        plan_id="p",
        step_id=1,
        objective="external",
        capability_id=capability,
        scope=CapabilityScope.from_mapping(scope),
        payload=payload,
    )


def action(request):
    return AuthorizedAction(
        request=request,
        principal="runtime.user",
        lease=CapabilityLease(capability_id=request.capability_id),
    )


def execute(registry, request):
    registry.validate(request)
    return registry._execute_authorized(action(request))


def test_confined_read_and_symlink_escape_rejected(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")
    outside = tmp_path / "outside.txt"; outside.write_text("secret", encoding="utf-8")
    store = ConfinedFileStore([FileRoot("docs", root)])
    text, digest, size = store.read_text("docs", "note.txt")
    assert text == "hello" and size == 5 and len(digest) == 64
    with pytest.raises(FileScopeError):
        store.read_text("docs", "../outside.txt")
    if hasattr(os, "symlink"):
        os.symlink(outside, root / "link.txt")
        with pytest.raises(FileScopeError):
            store.read_text("docs", "link.txt")


def test_atomic_create_and_compare_and_swap_replace(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    store = ConfinedFileStore([FileRoot("docs", root)])
    first, _, replaced = store.atomic_write_text(
        "docs", "note.txt", "one", expected_sha256=None
    )
    assert not replaced and (root / "note.txt").read_text() == "one"
    with pytest.raises(FileScopeError, match="hash mismatch"):
        store.atomic_write_text(
            "docs", "note.txt", "bad", expected_sha256="0" * 64
        )
    assert (root / "note.txt").read_text() == "one"
    second, _, replaced = store.atomic_write_text(
        "docs", "note.txt", "two", expected_sha256=first
    )
    assert replaced and second != first and (root / "note.txt").read_text() == "two"


def test_create_does_not_overwrite_existing_target(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    (root / "note.txt").write_text("original", encoding="utf-8")
    store = ConfinedFileStore([FileRoot("docs", root)])
    with pytest.raises(FileScopeError, match="replace requires"):
        store.atomic_write_text("docs", "note.txt", "new", expected_sha256=None)
    assert (root / "note.txt").read_text() == "original"


def test_file_capabilities_match_real_m6a_registry_shape(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")
    registry = TypedCapabilityRegistry()
    names = register_m6b_capabilities(
        registry,
        CapabilityPackConfig(file_store=ConfinedFileStore([FileRoot("docs", root)])),
    )
    assert names == (CAP_FILE_READ, CAP_FILE_WRITE)
    assert registry.descriptor(CAP_FILE_READ).approval_required
    assert registry.descriptor(CAP_FILE_WRITE).approval_required
    read = req(CAP_FILE_READ, {"root":"docs", "path":"note.txt"})
    body = json.loads(execute(registry, read).result)
    assert body["text"] == "hello" and body["path"] == "note.txt"
    write = req(CAP_FILE_WRITE, {"root":"docs", "path":"new.txt"}, {"text":"new"})
    result = json.loads(execute(registry, write).result)
    assert result["replaced"] is False and (root / "new.txt").read_text() == "new"


def test_replace_hashing_is_bounded_by_write_limit(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    target = root / "large.txt"; target.write_text("x" * 20, encoding="utf-8")
    store = ConfinedFileStore([FileRoot("docs", root, max_write_bytes=8)])
    with pytest.raises(FileScopeError, match="exceeds configured"):
        store.atomic_write_text("docs", "large.txt", "small", expected_sha256="0" * 64)
    assert target.read_text() == "x" * 20


def test_https_validator_blocks_http_credentials_loopback_and_non_global_dns():
    fetcher = SovereignHttpsFetcher(
        resolver=lambda host, port: ("10.0.0.1",),
        transport=lambda u,h,a,t,m: None,
    )
    for url in [
        "http://example.com",
        "https://u:p@example.com/",
        "https://127.0.0.1/",
        "https://example.com/",
        "https://example.com/a b",
        "https://example.com/é",
    ]:
        with pytest.raises(NetworkScopeError):
            fetcher.validate_url(url)


def test_https_text_fetch_is_bounded_and_rejects_redirect_or_binary():
    resolver = lambda host, port: ("8.8.8.8",)
    good = SovereignHttpsFetcher(
        resolver=resolver,
        transport=lambda url, host, address, timeout, limit: HttpsResponse(
            200, url, "text/plain; charset=utf-8", b"hello"
        ),
        max_response_bytes=32,
    )
    result = good.fetch_text("https://example.com/a")
    assert result["body"] == "hello" and result["utf8_bytes"] == 5
    redirected = SovereignHttpsFetcher(
        resolver=resolver,
        transport=lambda url,h,a,t,m: HttpsResponse(
            200, "https://example.com/b", "text/plain", b"x"
        ),
    )
    with pytest.raises(NetworkScopeError, match="changed URL"):
        redirected.fetch_text("https://example.com/a")
    binary = SovereignHttpsFetcher(
        resolver=resolver,
        transport=lambda url,h,a,t,m: HttpsResponse(
            200, url, "application/octet-stream", b"x"
        ),
    )
    with pytest.raises(NetworkScopeError, match="text/JSON"):
        binary.fetch_text("https://example.com/a")


def test_https_transport_receives_only_prevalidated_global_ip():
    seen = []
    fetcher = SovereignHttpsFetcher(
        resolver=lambda host, port: ("8.8.4.4", "8.8.8.8"),
        transport=lambda url, host, address, timeout, limit: (
            seen.append((host, address))
            or HttpsResponse(200, url, "text/plain", b"ok")
        ),
    )
    fetcher.fetch_text("https://example.com/a")
    assert seen == [("example.com", "8.8.4.4")]


def test_web_capability_uses_exact_url_scope_and_requires_approval():
    fetcher = SovereignHttpsFetcher(
        resolver=lambda host, port: ("8.8.8.8",),
        transport=lambda url,h,a,t,m: HttpsResponse(
            200, url, "application/json", b'{"ok":true}'
        ),
    )
    registry = TypedCapabilityRegistry()
    names = register_m6b_capabilities(
        registry,
        CapabilityPackConfig(https_fetcher=fetcher),
    )
    assert names == (CAP_WEB_FETCH,)
    assert registry.descriptor(CAP_WEB_FETCH).approval_required
    request = req(CAP_WEB_FETCH, {"url":"https://example.com/api"})
    result = json.loads(execute(registry, request).result)
    assert result["url"] == "https://example.com/api"


class Platform:
    def __init__(self): self.calls=[]
    def launch_package(self, package):
        self.calls.append(("launch", package)); return "launched"
    def write_clipboard(self, text):
        self.calls.append(("clipboard", text)); return "copied"


def test_app_and_clipboard_adapters_are_typed_and_approval_protected():
    platform = Platform(); registry = TypedCapabilityRegistry()
    names = register_m6b_capabilities(
        registry,
        CapabilityPackConfig(platform=platform),
    )
    assert names == (CAP_APP_LAUNCH, CAP_CLIPBOARD_WRITE)
    assert registry.descriptor(CAP_APP_LAUNCH).approval_required
    assert registry.descriptor(CAP_CLIPBOARD_WRITE).approval_required
    app = req(CAP_APP_LAUNCH, {"package":"com.example.app"})
    assert execute(registry, app).result == "launched"
    with pytest.raises(CapabilityPayloadError):
        registry.validate(req(CAP_APP_LAUNCH, {"package":"com.exämple.app"}))
    clip = req(CAP_CLIPBOARD_WRITE, {"channel":"system"}, {"text":"hello"})
    assert execute(registry, clip).result == "copied"
    assert platform.calls == [
        ("launch", "com.example.app"),
        ("clipboard", "hello"),
    ]


def test_registry_can_seal_after_pack_registration(tmp_path: Path):
    root = tmp_path / "root"; root.mkdir()
    registry = TypedCapabilityRegistry()
    register_m6b_capabilities(
        registry,
        CapabilityPackConfig(file_store=ConfinedFileStore([FileRoot("app", root)])),
    )
    registry.seal()
    assert registry.sealed


def test_app_package_segments_must_start_with_ascii_letter():
    registry = TypedCapabilityRegistry()
    register_m6b_capabilities(registry, CapabilityPackConfig(platform=Platform()))
    with pytest.raises(CapabilityPayloadError):
        registry.validate(req(CAP_APP_LAUNCH, {"package":"com._private.app"}))
