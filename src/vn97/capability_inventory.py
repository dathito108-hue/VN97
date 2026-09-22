from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Iterator

INVENTORY_SCHEMA = "VN97INV1"
INVENTORY_FILE = "inventory.vn97inv1.json"
MAX_INVENTORY_BYTES = 4 * 1024 * 1024
MAX_HISTORY = 4096
MAX_STACK_DEPTH = 64
_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,127}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
_REV_RE = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class CapabilityActivationError(RuntimeError):
    pass


class InventoryCorruptionError(CapabilityActivationError):
    pass


@dataclass(frozen=True)
class CapabilityInventoryItem:
    activation_id: str
    capability_id: str
    capability_version: int
    package_sha256: str
    publisher_key_id: str
    signature_sha256: str
    profile_id: str
    profile_sha256: str
    plan_sha256: str
    runtime_api_version: int
    backend_id: str
    artifact_sha256: str
    runtime_revision: str
    source_origin: str
    source_sha256: str
    source_license: str


@dataclass(frozen=True)
class InventoryEvent:
    generation: int
    action: str
    capability_id: str
    activation_id: str
    package_sha256: str

    def __post_init__(self) -> None:
        if self.generation <= 0:
            raise ValueError("event generation must be positive")
        if self.action not in {"activate", "rollback", "recover_activate", "recover_rollback"}:
            raise ValueError("inventory event action is invalid")


@dataclass(frozen=True)
class InventorySnapshot:
    generation: int
    active: tuple[CapabilityInventoryItem, ...]
    history: tuple[InventoryEvent, ...]
    pending_operation: str | None

    def current(self, capability_id: str) -> CapabilityInventoryItem | None:
        return next((value for value in self.active if value.capability_id == capability_id), None)


@dataclass(frozen=True)
class _StoredActivation:
    item: CapabilityInventoryItem
    backend_token: str


@dataclass
class _InventoryState:
    generation: int
    stacks: dict[str, list[_StoredActivation]]
    history: list[InventoryEvent]
    pending: dict[str, object] | None

    def snapshot(self) -> InventorySnapshot:
        active = tuple(
            stack[-1].item
            for _, stack in sorted(self.stacks.items())
            if stack
        )
        return InventorySnapshot(
            generation=self.generation,
            active=active,
            history=tuple(self.history),
            pending_operation=None if self.pending is None else str(self.pending["operation"]),
        )


def _canon(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InventoryCorruptionError("inventory value is not canonical JSON") from exc


def _strict_object(data: bytes) -> dict[str, object]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InventoryCorruptionError("inventory is not strict UTF-8 JSON") from exc
    if duplicates or not isinstance(value, dict) or _canon(value) != data:
        raise InventoryCorruptionError("inventory must be canonical object JSON")
    return value


def _require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise InventoryCorruptionError(f"{label} is invalid")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise InventoryCorruptionError(f"{label} is invalid")
    return value


def _require_text(value: object, label: str, limit: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise InventoryCorruptionError(f"{label} is invalid")
    return value


_PLAN_IDENTITY_KEYS = {
    "capability_id",
    "capability_version",
    "package_sha256",
    "plan_sha256",
    "profile_id",
    "profile_sha256",
    "publisher_key_id",
    "runtime_api_version",
    "signature_sha256",
    "source_license",
    "source_origin",
    "source_sha256",
}


def _validate_plan_identity(
    value: object,
    *,
    expected_capability_id: str | None = None,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PLAN_IDENTITY_KEYS:
        raise InventoryCorruptionError("pending activation identity keys mismatch")
    capability_id = _require_id(value["capability_id"], "pending identity capability_id")
    if expected_capability_id is not None and capability_id != expected_capability_id:
        raise InventoryCorruptionError("pending identity capability mismatch")
    version = value["capability_version"]
    runtime_api = value["runtime_api_version"]
    if type(version) is not int or not 1 <= version <= 0xFFFFFFFF:
        raise InventoryCorruptionError("pending identity capability_version invalid")
    if type(runtime_api) is not int or runtime_api <= 0:
        raise InventoryCorruptionError("pending identity runtime_api_version invalid")
    for label in (
        "package_sha256",
        "plan_sha256",
        "profile_sha256",
        "signature_sha256",
        "source_sha256",
    ):
        _require_sha(value[label], f"pending identity {label}")
    for label in ("publisher_key_id", "profile_id"):
        _require_id(value[label], f"pending identity {label}")
    _require_text(value["source_origin"], "pending identity source_origin")
    _require_text(value["source_license"], "pending identity source_license", 128)
    return value


def _validate_item(item: CapabilityInventoryItem) -> None:
    _require_sha(item.activation_id, "activation_id")
    _require_id(item.capability_id, "capability_id")
    if type(item.capability_version) is not int or not 1 <= item.capability_version <= 0xFFFFFFFF:
        raise InventoryCorruptionError("capability_version is invalid")
    for label, value in (
        ("package_sha256", item.package_sha256),
        ("signature_sha256", item.signature_sha256),
        ("profile_sha256", item.profile_sha256),
        ("plan_sha256", item.plan_sha256),
        ("artifact_sha256", item.artifact_sha256),
        ("source_sha256", item.source_sha256),
    ):
        _require_sha(value, label)
    for label, value in (
        ("publisher_key_id", item.publisher_key_id),
        ("profile_id", item.profile_id),
        ("backend_id", item.backend_id),
    ):
        _require_id(value, label)
    if type(item.runtime_api_version) is not int or item.runtime_api_version <= 0:
        raise InventoryCorruptionError("runtime_api_version is invalid")
    if not isinstance(item.runtime_revision, str) or not _REV_RE.fullmatch(item.runtime_revision):
        raise InventoryCorruptionError("runtime_revision is invalid")
    _require_text(item.source_origin, "source_origin")
    _require_text(item.source_license, "source_license", 128)


def _stored_to_obj(stored: _StoredActivation) -> dict[str, object]:
    item = stored.item
    return {
        "activation_id": item.activation_id,
        "artifact_sha256": item.artifact_sha256,
        "backend_id": item.backend_id,
        "backend_token": stored.backend_token,
        "capability_id": item.capability_id,
        "capability_version": item.capability_version,
        "package_sha256": item.package_sha256,
        "plan_sha256": item.plan_sha256,
        "profile_id": item.profile_id,
        "profile_sha256": item.profile_sha256,
        "publisher_key_id": item.publisher_key_id,
        "runtime_api_version": item.runtime_api_version,
        "runtime_revision": item.runtime_revision,
        "signature_sha256": item.signature_sha256,
        "source_license": item.source_license,
        "source_origin": item.source_origin,
        "source_sha256": item.source_sha256,
    }


def _stored_from_obj(value: object) -> _StoredActivation:
    if not isinstance(value, dict):
        raise InventoryCorruptionError("activation record must be object")
    keys = {
        "activation_id",
        "artifact_sha256",
        "backend_id",
        "backend_token",
        "capability_id",
        "capability_version",
        "package_sha256",
        "plan_sha256",
        "profile_id",
        "profile_sha256",
        "publisher_key_id",
        "runtime_api_version",
        "runtime_revision",
        "signature_sha256",
        "source_license",
        "source_origin",
        "source_sha256",
    }
    if set(value) != keys:
        raise InventoryCorruptionError("activation record keys mismatch")
    try:
        item = CapabilityInventoryItem(
            activation_id=value["activation_id"],
            capability_id=value["capability_id"],
            capability_version=value["capability_version"],
            package_sha256=value["package_sha256"],
            publisher_key_id=value["publisher_key_id"],
            signature_sha256=value["signature_sha256"],
            profile_id=value["profile_id"],
            profile_sha256=value["profile_sha256"],
            plan_sha256=value["plan_sha256"],
            runtime_api_version=value["runtime_api_version"],
            backend_id=value["backend_id"],
            artifact_sha256=value["artifact_sha256"],
            runtime_revision=value["runtime_revision"],
            source_origin=value["source_origin"],
            source_sha256=value["source_sha256"],
            source_license=value["source_license"],
        )
    except TypeError as exc:
        raise InventoryCorruptionError("activation record field type mismatch") from exc
    _validate_item(item)
    token = value["backend_token"]
    if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
        raise InventoryCorruptionError("backend token is invalid")
    return _StoredActivation(item, token)


def _event_to_obj(event: InventoryEvent) -> dict[str, object]:
    return {
        "action": event.action,
        "activation_id": event.activation_id,
        "capability_id": event.capability_id,
        "generation": event.generation,
        "package_sha256": event.package_sha256,
    }


def _event_from_obj(value: object) -> InventoryEvent:
    if not isinstance(value, dict) or set(value) != {
        "action",
        "activation_id",
        "capability_id",
        "generation",
        "package_sha256",
    }:
        raise InventoryCorruptionError("inventory event is invalid")
    try:
        event = InventoryEvent(
            generation=value["generation"],
            action=value["action"],
            capability_id=value["capability_id"],
            activation_id=value["activation_id"],
            package_sha256=value["package_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise InventoryCorruptionError("inventory event fields are invalid") from exc
    _require_id(event.capability_id, "event capability_id")
    _require_sha(event.activation_id, "event activation_id")
    _require_sha(event.package_sha256, "event package_sha256")
    return event


class CapabilityInventoryStore:
    def __init__(self, root: Path, *, max_inventory_bytes: int = MAX_INVENTORY_BYTES) -> None:
        root = Path(root)
        if not root.is_absolute() or not root.exists() or not root.is_dir() or root.is_symlink():
            raise ValueError("inventory root must be existing absolute non-symlink directory")
        if max_inventory_bytes < 1024:
            raise ValueError("max_inventory_bytes is too small")
        self.root = root
        self.max_inventory_bytes = max_inventory_bytes

    @contextmanager
    def locked(self) -> Iterator[int]:
        try:
            import fcntl
        except ImportError as exc:
            raise CapabilityActivationError("platform lacks advisory locking") from exc
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX)
            yield root_fd
        finally:
            os.close(root_fd)

    def load(self) -> InventorySnapshot:
        with self.locked() as root_fd:
            return self._load(root_fd).snapshot()

    def _load(self, root_fd: int) -> _InventoryState:
        try:
            fd = os.open(INVENTORY_FILE, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            return _InventoryState(0, {}, [], None)
        except OSError as exc:
            raise InventoryCorruptionError("inventory file unavailable or unsafe") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > self.max_inventory_bytes:
                raise InventoryCorruptionError("inventory file type/size invalid")
            data = bytearray()
            while len(data) < info.st_size:
                chunk = os.read(fd, min(65536, info.st_size - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) != info.st_size:
                raise InventoryCorruptionError("inventory read truncated")
        finally:
            os.close(fd)

        root = _strict_object(bytes(data))
        if set(root) != {"schema", "generation", "stacks", "history", "pending"} or root["schema"] != INVENTORY_SCHEMA:
            raise InventoryCorruptionError("inventory schema/keys mismatch")
        generation = root["generation"]
        if type(generation) is not int or generation < 0:
            raise InventoryCorruptionError("generation invalid")
        raw_stacks = root["stacks"]
        raw_history = root["history"]
        pending = root["pending"]
        if not isinstance(raw_stacks, list) or not isinstance(raw_history, list) or len(raw_history) > MAX_HISTORY:
            raise InventoryCorruptionError("inventory collections invalid")

        stacks: dict[str, list[_StoredActivation]] = {}
        for raw in raw_stacks:
            if not isinstance(raw, dict) or set(raw) != {"capability_id", "records"}:
                raise InventoryCorruptionError("inventory stack invalid")
            capability_id = _require_id(raw["capability_id"], "stack capability_id")
            records = raw["records"]
            if capability_id in stacks or not isinstance(records, list) or not 1 <= len(records) <= MAX_STACK_DEPTH:
                raise InventoryCorruptionError("inventory stack depth/identity invalid")
            decoded = [_stored_from_obj(record) for record in records]
            if any(value.item.capability_id != capability_id for value in decoded):
                raise InventoryCorruptionError("stack record capability mismatch")
            stacks[capability_id] = decoded

        history = [_event_from_obj(value) for value in raw_history]
        if history:
            generations = [event.generation for event in history]
            if any(right <= left for left, right in zip(generations, generations[1:])) or generations[-1] > generation:
                raise InventoryCorruptionError("inventory history generation sequence invalid")
        if pending is not None:
            self._validate_pending(pending)
            if pending["generation_base"] != generation:
                raise InventoryCorruptionError("pending generation does not match inventory generation")
        return _InventoryState(generation, stacks, history, pending)

    def _validate_pending(self, value: object) -> None:
        if not isinstance(value, dict):
            raise InventoryCorruptionError("pending transaction must be object")
        keys = {
            "tx_id",
            "operation",
            "phase",
            "capability_id",
            "backend_id",
            "generation_base",
            "identity",
            "backend_token",
            "artifact_sha256",
            "activation_id",
        }
        if set(value) != keys:
            raise InventoryCorruptionError("pending keys mismatch")
        _require_sha(value["tx_id"], "pending tx_id")
        _require_id(value["capability_id"], "pending capability_id")
        _require_id(value["backend_id"], "pending backend_id")
        if value["operation"] not in {"activate", "rollback"} or value["phase"] not in {"reserved", "prepared"}:
            raise InventoryCorruptionError("pending operation/phase invalid")
        if type(value["generation_base"]) is not int or value["generation_base"] < 0:
            raise InventoryCorruptionError("pending generation invalid")

        if value["operation"] == "activate":
            _validate_plan_identity(value["identity"], expected_capability_id=value["capability_id"])
            if value["activation_id"] is not None:
                raise InventoryCorruptionError("activation pending transaction must not predeclare activation_id")
        else:
            if value["identity"] != {} or value["phase"] != "prepared":
                raise InventoryCorruptionError("rollback pending identity/phase invalid")
            _require_sha(value["activation_id"], "pending activation_id")

        if value["phase"] == "reserved":
            if value["backend_token"] is not None or value["artifact_sha256"] is not None or value["activation_id"] is not None:
                raise InventoryCorruptionError("reserved transaction has prepared fields")
        else:
            if not isinstance(value["backend_token"], str) or not _TOKEN_RE.fullmatch(value["backend_token"]):
                raise InventoryCorruptionError("pending token invalid")
            _require_sha(value["artifact_sha256"], "pending artifact_sha256")

    def _write(self, root_fd: int, state: _InventoryState) -> None:
        stacks = [
            {"capability_id": capability_id, "records": [_stored_to_obj(value) for value in stack]}
            for capability_id, stack in sorted(state.stacks.items())
            if stack
        ]
        value = {
            "generation": state.generation,
            "history": [_event_to_obj(event) for event in state.history[-MAX_HISTORY:]],
            "pending": state.pending,
            "schema": INVENTORY_SCHEMA,
            "stacks": stacks,
        }
        data = _canon(value)
        if len(data) > self.max_inventory_bytes:
            raise CapabilityActivationError("inventory exceeds byte limit")
        temp_name = ".vn97inv-" + secrets.token_hex(12) + ".tmp"
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise CapabilityActivationError("short inventory write")
                written += count
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(temp_name, INVENTORY_FILE, src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            try:
                os.unlink(temp_name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
