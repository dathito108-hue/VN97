from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable
import xml.etree.ElementTree as ET

from .device_evidence_campaign import (
    ANDROID_ACTIVITY,
    ANDROID_PACKAGE,
    VN97AdbClient,
    VN97PhysicalEvidenceCampaignError,
    _ensure_physical_device,
)
from .production_materialization import (
    VN97FinalReleaseReceipt,
    parse_final_release_receipt,
)
from .release_attestation import (
    parse_apk_attestation,
)


SCHEMA = "VN97ACCEPT1"
SELFTEST_SCHEMA = "VN97SELFTEST1"
MAX_RECEIPT_BYTES = 256 * 1024

ACCEPTANCE_RECEIVER = (
    "ai.vn97.app/.VN97TurnkeyAcceptanceReceiver"
)
ACTION_BEGIN = (
    "ai.vn97.app.action.M19M_ACCEPTANCE_BEGIN"
)
ACTION_VERIFY_AFTER_REBOOT = (
    "ai.vn97.app.action.M19M_ACCEPTANCE_VERIFY_AFTER_REBOOT"
)
EXTRA_EXPECTED_PACKAGE_SHA256 = (
    "ai.vn97.app.extra.M19M_EXPECTED_PACKAGE_SHA256"
)
EXTRA_NONCE = (
    "ai.vn97.app.extra.M19M_NONCE"
)

_CHECK_NAMES = (
    "apk_identity",
    "autonomous_goal_created",
    "autonomous_goal_persisted_after_reboot",
    "bootstrap_ready_first_launch",
    "clean_install",
    "floating_service_before_reboot",
    "floating_service_recovered_after_reboot",
    "microphone_ingress_permission",
    "notification_permission",
    "overlay_permission",
    "physical_device",
    "post_reboot_bootstrap_ready",
    "real_reboot_observed",
    "turnkey_controls_hidden",
)


class VN97TurnkeyAcceptanceError(
    RuntimeError
):
    pass


def _require_hex(
    value: object,
    *,
    length: int,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(
            ch not in "0123456789abcdef"
            for ch in value
        )
    ):
        raise VN97TurnkeyAcceptanceError(
            f"{label} must be lowercase hex"
        )
    return value


@dataclass(frozen=True)
class VN97AcceptanceDevice:
    manufacturer: str
    model: str
    sdk_int: int
    abi: str

    def __post_init__(self) -> None:
        for value, label in (
            (
                self.manufacturer,
                "manufacturer",
            ),
            (
                self.model,
                "model",
            ),
            (
                self.abi,
                "ABI",
            ),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(
                    ord(ch) < 0x20
                    for ch in value
                )
            ):
                raise VN97TurnkeyAcceptanceError(
                    f"acceptance device {label} is invalid"
                )
        if (
            type(self.sdk_int) is not int
            or self.sdk_int <= 0
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance device SDK is invalid"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "abi": self.abi,
            "manufacturer":
                self.manufacturer,
            "model": self.model,
            "sdk_int": self.sdk_int,
        }


@dataclass(frozen=True)
class VN97TurnkeyAcceptanceReceipt:
    final_release_receipt_sha256: str
    repository_commit: str
    release_attestation_sha256: str
    bootstrap_package_sha256: str
    apk_sha256: str
    application_id: str
    version_code: int
    version_name: str
    device: VN97AcceptanceDevice
    pre_boot_id_sha256: str
    post_boot_id_sha256: str
    pre_selftest_sha256: str
    post_selftest_sha256: str
    autonomous_nonce_sha256: str
    checks: tuple[
        tuple[str, bool],
        ...,
    ]
    cleanup_state: str

    def __post_init__(self) -> None:
        for value, label in (
            (
                self.final_release_receipt_sha256,
                "final release receipt SHA-256",
            ),
            (
                self.release_attestation_sha256,
                "release attestation SHA-256",
            ),
            (
                self.bootstrap_package_sha256,
                "bootstrap package SHA-256",
            ),
            (
                self.apk_sha256,
                "APK SHA-256",
            ),
            (
                self.pre_boot_id_sha256,
                "pre-reboot boot-id SHA-256",
            ),
            (
                self.post_boot_id_sha256,
                "post-reboot boot-id SHA-256",
            ),
            (
                self.pre_selftest_sha256,
                "pre-reboot self-test SHA-256",
            ),
            (
                self.post_selftest_sha256,
                "post-reboot self-test SHA-256",
            ),
            (
                self.autonomous_nonce_sha256,
                "autonomous nonce SHA-256",
            ),
        ):
            _require_hex(
                value,
                length=64,
                label=label,
            )
        _require_hex(
            self.repository_commit,
            length=40,
            label="repository commit",
        )
        if (
            self.application_id
            != ANDROID_PACKAGE
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance application id mismatch"
            )
        if (
            type(self.version_code) is not int
            or self.version_code <= 0
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance version code is invalid"
            )
        if (
            not isinstance(
                self.version_name,
                str,
            )
            or not self.version_name
            or len(
                self.version_name
            ) > 128
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance version name is invalid"
            )
        if (
            self.pre_boot_id_sha256
            == self.post_boot_id_sha256
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance receipt must prove a real reboot"
            )
        expected_checks = tuple(
            (
                name,
                dict(self.checks)
                .get(name),
            )
            for name in
                _CHECK_NAMES
        )
        if (
            self.checks
            != expected_checks
            or any(
                value is not True
                for _, value in
                    self.checks
            )
        ):
            raise VN97TurnkeyAcceptanceError(
                "acceptance checks must be complete, ordered and true"
            )
        if self.cleanup_state not in {
            "UNINSTALLED",
            "KEPT_INSTALLED",
        }:
            raise VN97TurnkeyAcceptanceError(
                "acceptance cleanup state is invalid"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "apk_sha256":
                self.apk_sha256,
            "application_id":
                self.application_id,
            "autonomous_nonce_sha256":
                self
                .autonomous_nonce_sha256,
            "bootstrap_package_sha256":
                self
                .bootstrap_package_sha256,
            "checks": {
                name: value
                for name, value in
                    self.checks
            },
            "cleanup_state":
                self.cleanup_state,
            "device":
                self.device
                .canonical_object(),
            "final_release_receipt_sha256":
                self
                .final_release_receipt_sha256,
            "post_boot_id_sha256":
                self.post_boot_id_sha256,
            "post_selftest_sha256":
                self.post_selftest_sha256,
            "pre_boot_id_sha256":
                self.pre_boot_id_sha256,
            "pre_selftest_sha256":
                self.pre_selftest_sha256,
            "release_attestation_sha256":
                self
                .release_attestation_sha256,
            "repository_commit":
                self.repository_commit,
            "schema": SCHEMA,
            "status": "ACCEPTED",
            "version_code":
                self.version_code,
            "version_name":
                self.version_name,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_turnkey_acceptance_receipt(
    data: bytes,
) -> VN97TurnkeyAcceptanceReceipt:
    if (
        not 0 < len(data)
        <= MAX_RECEIPT_BYTES
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 byte size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(
        pairs: list[
            tuple[str, Any]
        ],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        root = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda raw:
                (_ for _ in ())
                .throw(
                    ValueError(raw)
                ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(
        root,
        dict,
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 must use canonical JSON"
        )
    expected = {
        "apk_sha256",
        "application_id",
        "autonomous_nonce_sha256",
        "bootstrap_package_sha256",
        "checks",
        "cleanup_state",
        "device",
        "final_release_receipt_sha256",
        "post_boot_id_sha256",
        "post_selftest_sha256",
        "pre_boot_id_sha256",
        "pre_selftest_sha256",
        "release_attestation_sha256",
        "repository_commit",
        "schema",
        "status",
        "version_code",
        "version_name",
    }
    if (
        set(root) != expected
        or root["schema"] != SCHEMA
        or root["status"]
        != "ACCEPTED"
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 schema/keys/status mismatch"
        )
    raw_device = root["device"]
    raw_checks = root["checks"]
    if (
        not isinstance(
            raw_device,
            dict,
        )
        or set(raw_device)
        != {
            "abi",
            "manufacturer",
            "model",
            "sdk_int",
        }
        or not isinstance(
            raw_checks,
            dict,
        )
        or tuple(
            raw_checks.keys()
        )
        != tuple(
            sorted(
                raw_checks.keys()
            )
        )
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 nested objects are invalid"
        )
    try:
        return VN97TurnkeyAcceptanceReceipt(
            final_release_receipt_sha256=
                root[
                    "final_release_receipt_sha256"
                ],
            repository_commit=
                root[
                    "repository_commit"
                ],
            release_attestation_sha256=
                root[
                    "release_attestation_sha256"
                ],
            bootstrap_package_sha256=
                root[
                    "bootstrap_package_sha256"
                ],
            apk_sha256=
                root["apk_sha256"],
            application_id=
                root["application_id"],
            version_code=
                root["version_code"],
            version_name=
                root["version_name"],
            device=
                VN97AcceptanceDevice(
                    manufacturer=
                        raw_device[
                            "manufacturer"
                        ],
                    model=
                        raw_device[
                            "model"
                        ],
                    sdk_int=
                        raw_device[
                            "sdk_int"
                        ],
                    abi=
                        raw_device[
                            "abi"
                        ],
                ),
            pre_boot_id_sha256=
                root[
                    "pre_boot_id_sha256"
                ],
            post_boot_id_sha256=
                root[
                    "post_boot_id_sha256"
                ],
            pre_selftest_sha256=
                root[
                    "pre_selftest_sha256"
                ],
            post_selftest_sha256=
                root[
                    "post_selftest_sha256"
                ],
            autonomous_nonce_sha256=
                root[
                    "autonomous_nonce_sha256"
                ],
            checks=tuple(
                (
                    name,
                    raw_checks[name],
                )
                for name in
                    _CHECK_NAMES
            ),
            cleanup_state=
                root["cleanup_state"],
        )
    except (
        TypeError,
        ValueError,
        VN97TurnkeyAcceptanceError,
    ) as exc:
        if isinstance(
            exc,
            VN97TurnkeyAcceptanceError,
        ):
            raise
        raise VN97TurnkeyAcceptanceError(
            str(exc)
        ) from exc


@dataclass(frozen=True)
class VN97TurnkeySelfTest:
    active_package_sha256: str
    assistant_open: bool
    autonomous_goal_present: bool
    autonomous_goal_state: str
    floating_enabled: bool
    microphone_permission: bool
    notification_permission: bool
    overlay_permission: bool
    phase: str
    turnkey_required: bool

    def __post_init__(self) -> None:
        _require_hex(
            self.active_package_sha256,
            length=64,
            label="self-test active package SHA-256",
        )
        for value, label in (
            (
                self.assistant_open,
                "assistant_open",
            ),
            (
                self.autonomous_goal_present,
                "autonomous_goal_present",
            ),
            (
                self.floating_enabled,
                "floating_enabled",
            ),
            (
                self.microphone_permission,
                "microphone_permission",
            ),
            (
                self.notification_permission,
                "notification_permission",
            ),
            (
                self.overlay_permission,
                "overlay_permission",
            ),
            (
                self.turnkey_required,
                "turnkey_required",
            ),
        ):
            if type(value) is not bool:
                raise VN97TurnkeyAcceptanceError(
                    f"self-test {label} must be boolean"
                )
        if (
            self.phase
            not in {
                "PRE_REBOOT",
                "POST_REBOOT",
            }
        ):
            raise VN97TurnkeyAcceptanceError(
                "self-test phase is invalid"
            )
        if (
            not isinstance(
                self.autonomous_goal_state,
                str,
            )
            or not self.autonomous_goal_state
            or len(
                self.autonomous_goal_state
            ) > 128
        ):
            raise VN97TurnkeyAcceptanceError(
                "self-test autonomous goal state is invalid"
            )
        if not all(
            (
                self.assistant_open,
                self.autonomous_goal_present,
                self.floating_enabled,
                self.microphone_permission,
                self.notification_permission,
                self.overlay_permission,
                self.turnkey_required,
            )
        ):
            raise VN97TurnkeyAcceptanceError(
                "turnkey self-test reported a false required condition"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "active_package_sha256":
                self.active_package_sha256,
            "assistant_open":
                self.assistant_open,
            "autonomous_goal_present":
                self
                .autonomous_goal_present,
            "autonomous_goal_state":
                self
                .autonomous_goal_state,
            "floating_enabled":
                self.floating_enabled,
            "microphone_permission":
                self
                .microphone_permission,
            "notification_permission":
                self
                .notification_permission,
            "overlay_permission":
                self.overlay_permission,
            "phase": self.phase,
            "schema":
                SELFTEST_SCHEMA,
            "turnkey_required":
                self.turnkey_required,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def parse_turnkey_selftest(
    data: bytes,
    *,
    expected_phase: str,
    expected_package_sha256: str,
) -> VN97TurnkeySelfTest:
    try:
        root = json.loads(
            data.decode(
                "utf-8",
                errors="strict",
            )
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise VN97TurnkeyAcceptanceError(
            "VN97SELFTEST1 is not strict UTF-8 JSON"
        ) from exc
    if (
        not isinstance(root, dict)
        or set(root)
        != {
            "active_package_sha256",
            "assistant_open",
            "autonomous_goal_present",
            "autonomous_goal_state",
            "floating_enabled",
            "microphone_permission",
            "notification_permission",
            "overlay_permission",
            "phase",
            "schema",
            "turnkey_required",
        }
        or root.get("schema")
        != SELFTEST_SCHEMA
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97SELFTEST1 schema/keys mismatch"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97TurnkeyAcceptanceError(
            "VN97SELFTEST1 must use canonical JSON"
        )
    test = VN97TurnkeySelfTest(
        active_package_sha256=
            root[
                "active_package_sha256"
            ],
        assistant_open=
            root["assistant_open"],
        autonomous_goal_present=
            root[
                "autonomous_goal_present"
            ],
        autonomous_goal_state=
            root[
                "autonomous_goal_state"
            ],
        floating_enabled=
            root["floating_enabled"],
        microphone_permission=
            root[
                "microphone_permission"
            ],
        notification_permission=
            root[
                "notification_permission"
            ],
        overlay_permission=
            root["overlay_permission"],
        phase=root["phase"],
        turnkey_required=
            root["turnkey_required"],
    )
    if test.phase != expected_phase:
        raise VN97TurnkeyAcceptanceError(
            "VN97SELFTEST1 phase mismatch"
        )
    if (
        test.active_package_sha256
        != expected_package_sha256
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97SELFTEST1 active package identity mismatch"
        )
    return test


def _regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97TurnkeyAcceptanceError(
            f"{label} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not 0 < info.st_size
        <= max_bytes
    ):
        raise VN97TurnkeyAcceptanceError(
            f"{label} must be a bounded regular non-symlink file"
        )
    return path.resolve(
        strict=True
    )


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_repository(
    repository_root: Path,
    *,
    expected_commit: str,
) -> Path:
    try:
        root = repository_root.resolve(
            strict=True
        )
    except OSError as exc:
        raise VN97TurnkeyAcceptanceError(
            "VN97 repository root is unavailable"
        ) from exc
    if not root.is_dir():
        raise VN97TurnkeyAcceptanceError(
            "VN97 repository root is not a directory"
        )
    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise VN97TurnkeyAcceptanceError(
            "VN97 Git identity could not be verified"
        ) from exc
    if commit != expected_commit:
        raise VN97TurnkeyAcceptanceError(
            "VN97FINAL1 repository commit does not match acceptance checkout HEAD"
        )
    if dirty:
        raise VN97TurnkeyAcceptanceError(
            "M19M requires a clean tracked Git worktree"
        )

    current_root = Path(
        __file__
    ).resolve().parent
    for name in (
        "turnkey_acceptance.py",
        "production_materialization.py",
        "release_attestation.py",
    ):
        current = current_root / name
        bound = (
            root
            / "src/vn97"
            / name
        )
        if (
            not current.is_file()
            or not bound.is_file()
            or _sha256_file(
                current
            )
            != _sha256_file(
                bound
            )
        ):
            raise VN97TurnkeyAcceptanceError(
                "running M19M source does not match accepted release checkout"
            )
    receiver = (
        root
        / "android/app/src/main/java/ai/vn97/app/VN97TurnkeyAcceptanceReceiver.kt"
    )
    if not receiver.is_file():
        raise VN97TurnkeyAcceptanceError(
            "M19M acceptance receiver source is missing"
        )
    return root


def _load_release_binding(
    *,
    final_receipt_path: Path,
    release_dir: Path,
) -> tuple[
    VN97FinalReleaseReceipt,
    bytes,
    Path,
    str,
    str,
]:
    receipt_file = _regular_file(
        final_receipt_path,
        label="VN97FINAL1 receipt",
        max_bytes=MAX_RECEIPT_BYTES,
    )
    receipt_bytes = (
        receipt_file.read_bytes()
    )
    receipt = (
        parse_final_release_receipt(
            receipt_bytes
        )
    )
    root = release_dir.resolve(
        strict=True
    )
    if not root.is_dir():
        raise VN97TurnkeyAcceptanceError(
            "release directory is invalid"
        )
    apk = _regular_file(
        root / "VN97-production.apk",
        label="VN97 production APK",
        max_bytes=2 * 1024 * 1024 * 1024,
    )
    if (
        apk.stat().st_size
        != receipt.apk_bytes
        or _sha256_file(apk)
        != receipt.apk_sha256
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97 production APK does not match VN97FINAL1"
        )
    attestation_path = _regular_file(
        root
        / "release-attestation.vn97apk1",
        label="VN97APK1 attestation",
        max_bytes=64 * 1024,
    )
    attestation_bytes = (
        attestation_path.read_bytes()
    )
    if (
        hashlib.sha256(
            attestation_bytes
        ).hexdigest()
        != receipt
        .release_attestation_sha256
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97APK1 does not match VN97FINAL1"
        )
    attestation = parse_apk_attestation(
        attestation_bytes
    )
    if (
        attestation.apk_sha256
        != receipt.apk_sha256
        or attestation.application_id
        != receipt.application_id
        or attestation.version_code
        != receipt.version_code
        or attestation.version_name
        != receipt.version_name
        or attestation
        .signer_certificate_sha256
        != receipt
        .signer_certificate_sha256
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97APK1 identity does not match VN97FINAL1"
        )
    return (
        receipt,
        receipt_bytes,
        apk,
        hashlib.sha256(
            attestation_bytes
        ).hexdigest(),
        attestation
        .bootstrap_package_sha256,
    )


def _device_profile(
    adb: VN97AdbClient,
    serial: str,
) -> VN97AcceptanceDevice:
    def prop(name: str) -> str:
        value = adb.text(
            serial,
            [
                "shell",
                "getprop",
                name,
            ],
        ).strip()
        if not value:
            raise VN97TurnkeyAcceptanceError(
                f"Android property {name} is unavailable"
            )
        return value

    try:
        sdk = int(
            prop(
                "ro.build.version.sdk"
            )
        )
    except ValueError as exc:
        raise VN97TurnkeyAcceptanceError(
            "Android SDK property is invalid"
        ) from exc
    return VN97AcceptanceDevice(
        manufacturer=prop(
            "ro.product.manufacturer"
        ),
        model=prop(
            "ro.product.model"
        ),
        sdk_int=sdk,
        abi=prop(
            "ro.product.cpu.abi"
        ),
    )


def _boot_id_sha256(
    adb: VN97AdbClient,
    serial: str,
) -> str:
    value = adb.text(
        serial,
        [
            "shell",
            "cat",
            "/proc/sys/kernel/random/boot_id",
        ],
    ).strip().lower()
    if (
        not re.fullmatch(
            r"[0-9a-f-]{36}",
            value,
        )
    ):
        raise VN97TurnkeyAcceptanceError(
            "Android kernel boot id is invalid"
        )
    return hashlib.sha256(
        value.encode("ascii")
    ).hexdigest()


def _grant_acceptance_permissions(
    adb: VN97AdbClient,
    serial: str,
    *,
    sdk_int: int,
) -> None:
    adb.run(
        serial,
        [
            "shell",
            "pm",
            "grant",
            ANDROID_PACKAGE,
            "android.permission.RECORD_AUDIO",
        ],
    )
    if sdk_int >= 33:
        adb.run(
            serial,
            [
                "shell",
                "pm",
                "grant",
                ANDROID_PACKAGE,
                "android.permission.POST_NOTIFICATIONS",
            ],
        )
    adb.run(
        serial,
        [
            "shell",
            "appops",
            "set",
            ANDROID_PACKAGE,
            "SYSTEM_ALERT_WINDOW",
            "allow",
        ],
    )
    overlay = adb.text(
        serial,
        [
            "shell",
            "appops",
            "get",
            ANDROID_PACKAGE,
            "SYSTEM_ALERT_WINDOW",
        ],
    ).lower()
    if "allow" not in overlay:
        raise VN97TurnkeyAcceptanceError(
            "overlay app-op was not granted"
        )


def _parse_installed_identity(
    text: str,
) -> tuple[int, str]:
    version_code = re.search(
        r"\bversionCode=(\d+)\b",
        text,
    )
    version_name = re.search(
        r"(?m)^\s*versionName=([^\r\n]+)$",
        text,
    )
    if (
        version_code is None
        or version_name is None
    ):
        raise VN97TurnkeyAcceptanceError(
            "installed package identity could not be parsed"
        )
    return (
        int(
            version_code.group(1)
        ),
        version_name
        .group(1)
        .strip(),
    )


def _ui_xml(
    adb: VN97AdbClient,
    serial: str,
) -> bytes:
    remote = (
        "/sdcard/vn97-m19m-window.xml"
    )
    adb.run(
        serial,
        [
            "shell",
            "uiautomator",
            "dump",
            remote,
        ],
        timeout=30.0,
    )
    try:
        raw = adb.run(
            serial,
            [
                "exec-out",
                "cat",
                remote,
            ],
            timeout=30.0,
        ).stdout
    finally:
        adb.run(
            serial,
            [
                "shell",
                "rm",
                "-f",
                remote,
            ],
            check=False,
        )
    try:
        ET.fromstring(raw)
    except ET.ParseError as exc:
        raise VN97TurnkeyAcceptanceError(
            "Android UI hierarchy is not valid XML"
        ) from exc
    return raw


def _ui_text_values(
    raw: bytes,
) -> set[str]:
    root = ET.fromstring(raw)
    values: set[str] = set()
    for node in root.iter():
        for key in (
            "text",
            "content-desc",
            "hint",
        ):
            value = node.attrib.get(
                key,
                "",
            ).strip()
            if value:
                values.add(value)
    return values


def _wait_turnkey_ui_ready(
    adb: VN97AdbClient,
    serial: str,
    *,
    timeout_seconds: int,
) -> bytes:
    deadline = (
        time.monotonic()
        + timeout_seconds
    )
    latest: bytes | None = None
    while time.monotonic() < deadline:
        try:
            latest = _ui_xml(
                adb,
                serial,
            )
            values = _ui_text_values(
                latest
            )
            ready = (
                "VN97 native model ready."
                in values
                or "Ready." in values
            )
            voice = (
                "Microphone voice enabled"
                in values
            )
            turnkey_hidden = (
                "IMPORT VN97 MODEL"
                not in values
                and
                "Collect VN97 mobile evidence"
                not in values
            )
            if (
                ready
                and voice
                and turnkey_hidden
            ):
                return latest
        except (
            VN97TurnkeyAcceptanceError,
            VN97PhysicalEvidenceCampaignError,
        ):
            pass
        time.sleep(1.0)
    detail = (
        sorted(
            _ui_text_values(
                latest
            )
        )[:40]
        if latest
        else []
    )
    raise VN97TurnkeyAcceptanceError(
        "turnkey UI did not reach READY state: "
        + repr(detail)
    )


def _wait_service(
    adb: VN97AdbClient,
    serial: str,
    *,
    timeout_seconds: int,
) -> None:
    deadline = (
        time.monotonic()
        + timeout_seconds
    )
    while time.monotonic() < deadline:
        try:
            output = adb.text(
                serial,
                [
                    "shell",
                    "dumpsys",
                    "activity",
                    "services",
                    (
                        ANDROID_PACKAGE
                        + "/.VN97FloatingAssistantService"
                    ),
                ],
                check=False,
            )
        except VN97PhysicalEvidenceCampaignError:
            output = ""
        if (
            "VN97FloatingAssistantService"
            in output
            and "ServiceRecord"
            in output
        ):
            return
        time.sleep(1.0)
    raise VN97TurnkeyAcceptanceError(
        "floating assistant service did not become active"
    )


def _decode_broadcast_result(
    output: str,
) -> tuple[int, bytes]:
    match = re.search(
        r"Broadcast completed:\s*result=(-?\d+)(?:,\s*data=\"([^\"]*)\")?",
        output,
    )
    if match is None:
        raise VN97TurnkeyAcceptanceError(
            "acceptance broadcast result could not be parsed"
        )
    code = int(
        match.group(1)
    )
    encoded = (
        match.group(2)
        or ""
    )
    try:
        data = base64.b64decode(
            encoded,
            validate=True,
        )
    except Exception as exc:
        raise VN97TurnkeyAcceptanceError(
            "acceptance broadcast data is not valid base64"
        ) from exc
    return code, data


def _run_selftest(
    adb: VN97AdbClient,
    serial: str,
    *,
    action: str,
    phase: str,
    package_sha256: str,
    nonce: str,
    timeout_seconds: int,
) -> VN97TurnkeySelfTest:
    output = adb.text(
        serial,
        [
            "shell",
            "am",
            "broadcast",
            "--receiver-foreground",
            "-n",
            ACCEPTANCE_RECEIVER,
            "-a",
            action,
            "--es",
            EXTRA_EXPECTED_PACKAGE_SHA256,
            package_sha256,
            "--es",
            EXTRA_NONCE,
            nonce,
        ],
        timeout=float(
            timeout_seconds
        ),
        check=False,
    )
    code, data = (
        _decode_broadcast_result(
            output
        )
    )
    if code != 0:
        detail = data.decode(
            "utf-8",
            errors="replace",
        )[:768]
        raise VN97TurnkeyAcceptanceError(
            "turnkey acceptance probe failed: "
            + detail
        )
    return parse_turnkey_selftest(
        data,
        expected_phase=phase,
        expected_package_sha256=
            package_sha256,
    )


def _wait_for_reboot(
    adb: VN97AdbClient,
    serial: str,
    *,
    old_boot_id_sha256: str,
    timeout_seconds: int,
) -> str:
    adb.run(
        serial,
        ["reboot"],
        timeout=10.0,
        check=False,
    )
    adb.run(
        serial,
        ["wait-for-device"],
        timeout=float(
            timeout_seconds
        ),
    )
    deadline = (
        time.monotonic()
        + timeout_seconds
    )
    latest_boot: str | None = None
    while time.monotonic() < deadline:
        try:
            boot_completed = adb.text(
                serial,
                [
                    "shell",
                    "getprop",
                    "sys.boot_completed",
                ],
                timeout=10.0,
                check=False,
            ).strip()
        except VN97PhysicalEvidenceCampaignError:
            boot_completed = ""
        if boot_completed == "1":
            try:
                latest_boot = (
                    _boot_id_sha256(
                        adb,
                        serial,
                    )
                )
            except Exception:
                latest_boot = None
            if (
                latest_boot is not None
                and latest_boot
                != old_boot_id_sha256
            ):
                adb.run(
                    serial,
                    [
                        "shell",
                        "wm",
                        "dismiss-keyguard",
                    ],
                    check=False,
                )
                return latest_boot
        time.sleep(2.0)
    raise VN97TurnkeyAcceptanceError(
        "physical device did not complete a distinct reboot"
    )


def _atomic_create(
    path: Path,
    data: bytes,
) -> None:
    if (
        path.exists()
        or path.is_symlink()
    ):
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 output must not already exist"
        )
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if path.parent.is_symlink():
        raise VN97TurnkeyAcceptanceError(
            "VN97ACCEPT1 parent must not be a symlink"
        )
    fd, temporary = (
        tempfile.mkstemp(
            prefix=".vn97-accept-",
            dir=path.parent,
        )
    )
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(
                stream.fileno()
            )
        try:
            os.link(
                temporary,
                path,
            )
        except FileExistsError as exc:
            raise VN97TurnkeyAcceptanceError(
                "VN97ACCEPT1 output must not already exist"
            ) from exc
        except OSError as exc:
            raise VN97TurnkeyAcceptanceError(
                "VN97ACCEPT1 could not be published atomically"
            ) from exc
        os.unlink(
            temporary
        )
        if path.read_bytes() != data:
            raise VN97TurnkeyAcceptanceError(
                "VN97ACCEPT1 post-write verification failed"
            )
    finally:
        if os.path.exists(
            temporary
        ):
            os.unlink(
                temporary
            )


def run_clean_device_acceptance(
    *,
    final_receipt_path: Path,
    release_dir: Path,
    repository_root: Path,
    serial: str,
    output_path: Path | None = None,
    adb_executable: str = "adb",
    keep_installed: bool = False,
    launch_timeout_seconds: int = 120,
    reboot_timeout_seconds: int = 240,
    selftest_timeout_seconds: int = 60,
    adb_client: VN97AdbClient | None = None,
) -> VN97TurnkeyAcceptanceReceipt:
    (
        final,
        final_bytes,
        apk,
        attestation_sha256,
        bootstrap_package_sha256,
    ) = _load_release_binding(
        final_receipt_path=
            final_receipt_path,
        release_dir=release_dir,
    )
    _verify_repository(
        repository_root,
        expected_commit=
            final.repository_commit,
    )

    adb = (
        adb_client
        or VN97AdbClient(
            adb_executable
        )
    )
    _ensure_physical_device(
        adb,
        serial,
    )
    profile = _device_profile(
        adb,
        serial,
    )
    if adb.app_installed(serial):
        raise VN97TurnkeyAcceptanceError(
            "clean-device acceptance requires ai.vn97.app to be absent before install"
        )
    pre_boot = _boot_id_sha256(
        adb,
        serial,
    )
    nonce = hashlib.sha256(
        final_bytes
        + (
            profile.manufacturer
            + "\n"
            + profile.model
            + "\n"
            + str(
                profile.sdk_int
            )
            + "\n"
            + profile.abi
        ).encode("utf-8")
    ).hexdigest()[:16]

    installed = False
    primary_failure: BaseException | None = None
    pre_selftest: VN97TurnkeySelfTest | None = None
    post_selftest: VN97TurnkeySelfTest | None = None
    post_boot: str | None = None
    try:
        install = adb.run(
            serial,
            [
                "install",
                "--no-streaming",
                str(apk),
            ],
            timeout=300.0,
        )
        if (
            b"Success"
            not in install.stdout
            and b"Success"
            not in install.stderr
        ):
            raise VN97TurnkeyAcceptanceError(
                "Android install did not report Success"
            )
        installed = True

        package_dump = adb.text(
            serial,
            [
                "shell",
                "dumpsys",
                "package",
                ANDROID_PACKAGE,
            ],
        )
        installed_version = (
            _parse_installed_identity(
                package_dump
            )
        )
        if installed_version != (
            final.version_code,
            final.version_name,
        ):
            raise VN97TurnkeyAcceptanceError(
                "installed application version does not match VN97FINAL1"
            )
        if (
            "VN97TurnkeyAcceptanceReceiver"
            not in package_dump
            or "android.permission.DUMP"
            not in package_dump
        ):
            raise VN97TurnkeyAcceptanceError(
                "installed production APK is missing the protected M19M acceptance surface"
            )

        _grant_acceptance_permissions(
            adb,
            serial,
            sdk_int=
                profile.sdk_int,
        )
        adb.run(
            serial,
            [
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                ANDROID_ACTIVITY,
            ],
            timeout=60.0,
        )
        first_ui = (
            _wait_turnkey_ui_ready(
                adb,
                serial,
                timeout_seconds=
                    launch_timeout_seconds,
            )
        )
        first_values = (
            _ui_text_values(
                first_ui
            )
        )
        if (
            "IMPORT VN97 MODEL"
            in first_values
            or
            "Collect VN97 mobile evidence"
            in first_values
        ):
            raise VN97TurnkeyAcceptanceError(
                "developer provisioning/evidence controls are visible in turnkey APK"
            )

        pre_selftest = _run_selftest(
            adb,
            serial,
            action=ACTION_BEGIN,
            phase="PRE_REBOOT",
            package_sha256=
                bootstrap_package_sha256,
            nonce=nonce,
            timeout_seconds=
                selftest_timeout_seconds,
        )
        _wait_service(
            adb,
            serial,
            timeout_seconds=60,
        )

        post_boot = _wait_for_reboot(
            adb,
            serial,
            old_boot_id_sha256=
                pre_boot,
            timeout_seconds=
                reboot_timeout_seconds,
        )
        if not adb.app_installed(
            serial
        ):
            raise VN97TurnkeyAcceptanceError(
                "VN97 package disappeared after reboot"
            )

        _wait_service(
            adb,
            serial,
            timeout_seconds=90,
        )
        adb.run(
            serial,
            [
                "shell",
                "am",
                "start",
                "-W",
                "-n",
                ANDROID_ACTIVITY,
            ],
            timeout=60.0,
        )
        _wait_turnkey_ui_ready(
            adb,
            serial,
            timeout_seconds=
                launch_timeout_seconds,
        )
        post_selftest = _run_selftest(
            adb,
            serial,
            action=
                ACTION_VERIFY_AFTER_REBOOT,
            phase="POST_REBOOT",
            package_sha256=
                bootstrap_package_sha256,
            nonce=nonce,
            timeout_seconds=
                selftest_timeout_seconds,
        )

        checks = tuple(
            (
                name,
                True,
            )
            for name in
                _CHECK_NAMES
        )

    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        cleanup_state = (
            "KEPT_INSTALLED"
            if keep_installed
            else "UNINSTALLED"
        )
        if (
            installed
            and not keep_installed
        ):
            cleanup = adb.run(
                serial,
                [
                    "uninstall",
                    ANDROID_PACKAGE,
                ],
                timeout=120.0,
                check=False,
            )
            if cleanup.returncode != 0:
                detail = cleanup.stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip().replace(
                    "\n",
                    " ",
                )[:512]
                if primary_failure is None:
                    raise VN97TurnkeyAcceptanceError(
                        "M19M acceptance passed but cleanup uninstall failed"
                        + (
                            f": {detail}"
                            if detail
                            else ""
                        )
                    )
                if hasattr(
                    primary_failure,
                    "add_note",
                ):
                    primary_failure.add_note(
                        "M19M cleanup uninstall failed"
                    )

    if (
        pre_selftest is None
        or post_selftest is None
        or post_boot is None
    ):
        raise VN97TurnkeyAcceptanceError(
            "M19M acceptance ended without complete observations"
        )

    receipt = (
        VN97TurnkeyAcceptanceReceipt(
            final_release_receipt_sha256=
                hashlib.sha256(
                    final_bytes
                ).hexdigest(),
            repository_commit=
                final.repository_commit,
            release_attestation_sha256=
                attestation_sha256,
            bootstrap_package_sha256=
                bootstrap_package_sha256,
            apk_sha256=
                final.apk_sha256,
            application_id=
                final.application_id,
            version_code=
                final.version_code,
            version_name=
                final.version_name,
            device=profile,
            pre_boot_id_sha256=
                pre_boot,
            post_boot_id_sha256=
                post_boot,
            pre_selftest_sha256=
                hashlib.sha256(
                    pre_selftest
                    .to_bytes()
                ).hexdigest(),
            post_selftest_sha256=
                hashlib.sha256(
                    post_selftest
                    .to_bytes()
                ).hexdigest(),
            autonomous_nonce_sha256=
                hashlib.sha256(
                    nonce.encode(
                        "ascii"
                    )
                ).hexdigest(),
            checks=checks,
            cleanup_state=
                cleanup_state,
        )
    )
    if output_path is not None:
        _atomic_create(
            output_path,
            receipt.to_bytes(),
        )
        parsed = (
            parse_turnkey_acceptance_receipt(
                output_path
                .read_bytes()
            )
        )
        if parsed != receipt:
            raise VN97TurnkeyAcceptanceError(
                "persisted VN97ACCEPT1 does not round-trip"
            )
    return receipt
