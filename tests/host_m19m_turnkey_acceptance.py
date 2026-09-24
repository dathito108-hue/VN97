from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg


def load_module(name: str, filename: str):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        name,
        SRC / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load {filename}"
        )
    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load dependency graph using normal package-relative import resolution.
sys.path.insert(0, str(ROOT / "src"))
MAT = load_module(
    "vn97.production_materialization",
    "production_materialization.py",
)
ATT = load_module(
    "vn97.release_attestation",
    "release_attestation.py",
)
ACC = load_module(
    "vn97.turnkey_acceptance",
    "turnkey_acceptance.py",
)
DEV = load_module(
    "vn97.device_evidence_campaign",
    "device_evidence_campaign.py",
)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


class FakeExecutor:
    def __init__(
        self,
        *,
        apk_version_code: int,
        apk_version_name: str,
        package_sha256: str,
        emulator: bool = False,
        transient_ui_failures: int = 0,
        transient_boot_failures: int = 0,
        transient_service_failures: int = 0,
        fail_pre_selftest: bool = false,
    ) -> None:
        self.apk_version_code = (
            apk_version_code
        )
        self.apk_version_name = (
            apk_version_name
        )
        self.package_sha256 = (
            package_sha256
        )
        self.emulator = emulator
        self.transient_ui_failures = (
            transient_ui_failures
        )
        self.transient_boot_failures = (
            transient_boot_failures
        )
        self.transient_service_failures = (
            transient_service_failures
        )
        self.fail_pre_selftest = (
            fail_pre_selftest
        )
        self.installed = False
        self.service = False
        self.rebooted = False
        self.commands: list[
            list[str]
        ] = []

    def _selftest(
        self,
        phase: str,
    ) -> bytes:
        test = ACC.VN97TurnkeySelfTest(
            active_package_sha256=
                self.package_sha256,
            assistant_open=True,
            autonomous_goal_present=True,
            autonomous_goal_state=(
                "SCHEDULED"
                if phase == "PRE_REBOOT"
                else "COMPLETED"
            ),
            floating_enabled=True,
            microphone_permission=True,
            notification_permission=True,
            overlay_permission=True,
            phase=phase,
            turnkey_required=True,
        )
        return test.to_bytes()

    def __call__(
        self,
        command: list[str],
        input_bytes: bytes | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[bytes]:
        del input_bytes, timeout
        self.commands.append(
            list(command)
        )
        text = " ".join(command)

        if text.endswith(" get-state"):
            return subprocess.CompletedProcess(
                command, 0, b"device\n", b""
            )
        if (
            "getprop ro.kernel.qemu"
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    b"1\n"
                    if self.emulator
                    else b"0\n"
                ),
                b"",
            )
        if (
            "getprop ro.boot.qemu"
            in text
        ):
            return subprocess.CompletedProcess(
                command, 0, b"0\n", b""
            )
        props = {
            "ro.product.manufacturer":
                "VN97Fixture",
            "ro.product.model":
                "TurnkeyPhone",
            "ro.build.version.sdk":
                "37",
            "ro.product.cpu.abi":
                "arm64-v8a",
            "sys.boot_completed":
                "1",
        }
        for key, value in props.items():
            if (
                "getprop " + key
                in text
            ):
                if (
                    key == "sys.boot_completed"
                    and self.rebooted
                    and self.transient_boot_failures > 0
                ):
                    self.transient_boot_failures -= 1
                    raise subprocess.TimeoutExpired(
                        command,
                        1,
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    (value + "\n")
                    .encode(),
                    b"",
                )

        if (
            "cat /proc/sys/kernel/random/boot_id"
            in text
        ):
            boot = (
                "22222222-2222-2222-2222-222222222222"
                if self.rebooted
                else "11111111-1111-1111-1111-111111111111"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                (boot + "\n").encode(),
                b"",
            )

        if (
            "pm path ai.vn97.app"
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    b"package:/data/app/ai.vn97.app/base.apk\n"
                    if self.installed
                    else b""
                ),
                b"",
            )

        if " install " in (
            " " + text + " "
        ):
            self.installed = True
            return subprocess.CompletedProcess(
                command, 0, b"Success\n", b""
            )

        if (
            "dumpsys package ai.vn97.app"
            in text
        ):
            payload = (
                "Package [ai.vn97.app]\n"
                + "  versionCode="
                + str(
                    self.apk_version_code
                )
                + " minSdk=29 targetSdk=37\n"
                + "  versionName="
                + self.apk_version_name
                + "\n"
                + "  receiver: VN97TurnkeyAcceptanceReceiver\n"
                + "  permission=android.permission.DUMP\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                payload.encode(),
                b"",
            )

        if (
            "pm grant ai.vn97.app"
            in text
            or
            "appops set ai.vn97.app"
            in text
        ):
            return subprocess.CompletedProcess(
                command, 0, b"", b""
            )
        if (
            "appops get ai.vn97.app SYSTEM_ALERT_WINDOW"
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                b"SYSTEM_ALERT_WINDOW: allow\n",
                b"",
            )

        if (
            "am start -W -n ai.vn97.app/.VN97MainActivity"
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                b"Status: ok\n",
                b"",
            )

        if (
            "uiautomator dump"
            in text
        ):
            if self.transient_ui_failures > 0:
                self.transient_ui_failures -= 1
                raise subprocess.TimeoutExpired(
                    command,
                    1,
                )
            return subprocess.CompletedProcess(
                command,
                0,
                b"UI hierchary dumped\n",
                b"",
            )
        if (
            "exec-out cat /sdcard/vn97-m19m-window.xml"
            in text
        ):
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<hierarchy rotation="0">'
                '<node text="VN97 native model ready." />'
                '<node text="Microphone voice enabled" />'
                '<node text="VN97 model active." />'
                '</hierarchy>'
            )
            return subprocess.CompletedProcess(
                command,
                0,
                xml.encode(),
                b"",
            )
        if (
            "rm -f /sdcard/vn97-m19m-window.xml"
            in text
        ):
            return subprocess.CompletedProcess(
                command, 0, b"", b""
            )

        if (
            "am broadcast"
            in text
            and ACC.ACTION_BEGIN
            in text
        ):
            self.service = True
            if self.fail_pre_selftest:
                encoded = base64.b64encode(
                    b"injected pre-reboot failure"
                ).decode()
                output = (
                    'Broadcasting\n'
                    'Broadcast completed: result=1, data="'
                    + encoded
                    + '"\n'
                )
            else:
                encoded = base64.b64encode(
                    self._selftest(
                        "PRE_REBOOT"
                    )
                ).decode()
                output = (
                    'Broadcasting\n'
                    'Broadcast completed: result=0, data="'
                    + encoded
                    + '"\n'
                )
            return subprocess.CompletedProcess(
                command,
                0,
                output.encode(),
                b"",
            )
        if (
            "am broadcast"
            in text
            and
            ACC.ACTION_VERIFY_AFTER_REBOOT
            in text
        ):
            encoded = base64.b64encode(
                self._selftest(
                    "POST_REBOOT"
                )
            ).decode()
            output = (
                'Broadcasting\n'
                'Broadcast completed: result=0, data="'
                + encoded
                + '"\n'
            )
            return subprocess.CompletedProcess(
                command,
                0,
                output.encode(),
                b"",
            )

        if (
            "dumpsys activity services"
            in text
        ):
            if self.transient_service_failures > 0:
                self.transient_service_failures -= 1
                raise subprocess.TimeoutExpired(
                    command,
                    1,
                )
            payload = (
                "ServiceRecord{123 "
                "ai.vn97.app/.VN97FloatingAssistantService}\n"
                if self.service
                else "nothing\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                payload.encode(),
                b"",
            )

        if text.endswith(" reboot"):
            self.rebooted = True
            # persisted preference + boot receiver restarts the floating service
            self.service = True
            return subprocess.CompletedProcess(
                command, 0, b"", b""
            )
        if text.endswith(" wait-for-device"):
            return subprocess.CompletedProcess(
                command, 0, b"", b""
            )
        if (
            "wm dismiss-keyguard"
            in text
        ):
            return subprocess.CompletedProcess(
                command, 0, b"", b""
            )

        if (
            "uninstall ai.vn97.app"
            in text
        ):
            self.installed = False
            self.service = False
            return subprocess.CompletedProcess(
                command, 0, b"Success\n", b""
            )

        return subprocess.CompletedProcess(
            command, 0, b"", b""
        )


def fixture_release(
    root: Path,
    *,
    repository_commit: str,
):
    release = root / "release"
    release.mkdir()
    apk = release / "VN97-production.apk"
    apk.write_bytes(
        b"VN97 production APK fixture"
    )
    apk_sha = hashlib.sha256(
        apk.read_bytes()
    ).hexdigest()
    signer = "99" * 32
    package_sha = "44" * 32
    readiness_bytes = (
        b'{"fixture":"VN97READY1"}'
    )
    bootstrap_bytes = (
        b'{"fixture":"VN97BOOTREL6"}'
    )
    (
        release
        / "production-readiness.vn97ready1"
    ).write_bytes(
        readiness_bytes
    )
    (
        release
        / "bootstrap-release.vn97bootrel6.json"
    ).write_bytes(
        bootstrap_bytes
    )
    readiness_sha = hashlib.sha256(
        readiness_bytes
    ).hexdigest()
    bootstrap_sha = hashlib.sha256(
        bootstrap_bytes
    ).hexdigest()

    attestation = ATT.VN97ApkAttestation(
        application_id="ai.vn97.app",
        version_code=190100,
        version_name="1.0.0-rc1",
        apk_bytes=apk.stat().st_size,
        apk_sha256=apk_sha,
        signer_certificate_sha256=(
            signer,
        ),
        release_candidate_manifest_sha256=
            "33" * 32,
        bootstrap_release_report_sha256=
            bootstrap_sha,
        bootstrap_package_sha256=
            package_sha,
        bootstrap_signature_sha256=
            "66" * 32,
        bootstrap_publisher_sha256=
            "77" * 32,
        release_manifest_sha256=
            "88" * 32,
    )
    attestation_bytes = (
        attestation.to_bytes()
    )
    (
        release
        / "release-attestation.vn97apk1"
    ).write_bytes(
        attestation_bytes
    )

    final = MAT.VN97FinalReleaseReceipt(
        manifest_sha256="11" * 32,
        repository_commit=
            repository_commit,
        closure_report_sha256=
            "22" * 32,
        readiness_report_sha256=
            readiness_sha,
        release_candidate_manifest_sha256=
            "33" * 32,
        release_attestation_sha256=
            hashlib.sha256(
                attestation_bytes
            ).hexdigest(),
        bootstrap_release_report_sha256=
            bootstrap_sha,
        release_manifest_sha256=
            "88" * 32,
        application_id="ai.vn97.app",
        version_code=190100,
        version_name="1.0.0-rc1",
        apk_bytes=apk.stat().st_size,
        apk_sha256=apk_sha,
        signer_certificate_sha256=(
            signer,
        ),
    )
    final_path = root / "final.vn97final1"
    final_path.write_bytes(
        final.to_bytes()
    )
    return (
        release,
        final_path,
        package_sha,
        final,
    )


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower()

    selftest = ACC.VN97TurnkeySelfTest(
        active_package_sha256=
            "44" * 32,
        assistant_open=True,
        autonomous_goal_present=True,
        autonomous_goal_state=
            "SCHEDULED",
        floating_enabled=True,
        microphone_permission=True,
        notification_permission=True,
        overlay_permission=True,
        phase="PRE_REBOOT",
        turnkey_required=True,
    )
    assert (
        ACC.parse_turnkey_selftest(
            selftest.to_bytes(),
            expected_phase=
                "PRE_REBOOT",
            expected_package_sha256=
                "44" * 32,
        )
        == selftest
    )

    with tempfile.TemporaryDirectory() as tmp:
        target = (
            Path(tmp)
            / "raced.vn97accept1"
        )
        real_link = ACC.os.link

        def racing_link(
            source,
            destination,
        ):
            Path(destination).write_bytes(
                b"competing-acceptance-receipt"
            )
            return real_link(
                source,
                destination,
            )

        ACC.os.link = racing_link
        try:
            expect_failure(
                "acceptance receipt publication race",
                lambda: ACC._atomic_create(
                    target,
                    selftest.to_bytes(),
                ),
            )
        finally:
            ACC.os.link = real_link
        assert target.read_bytes() == (
            b"competing-acceptance-receipt"
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (
            release,
            final_path,
            package_sha,
            final,
        ) = fixture_release(
            root,
            repository_commit=commit,
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
        )
        (
            release
            / "unexpected.txt"
        ).write_text(
            "unexpected",
            encoding="utf-8",
        )
        expect_failure(
            "release bundle extra entry",
            lambda:
                ACC.run_clean_device_acceptance(
                    final_receipt_path=
                        final_path,
                    release_dir=release,
                    repository_root=ROOT,
                    serial="physical-001",
                    adb_client=
                        DEV.VN97AdbClient(
                            "adb",
                            executor=fake,
                        ),
                ),
        )
        assert not fake.commands

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (
            release,
            final_path,
            package_sha,
            final,
        ) = fixture_release(
            root,
            repository_commit=commit,
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
        )
        (
            release
            / "production-readiness.vn97ready1"
        ).write_bytes(
            b'{"tampered":true}'
        )
        expect_failure(
            "release readiness tamper",
            lambda:
                ACC.run_clean_device_acceptance(
                    final_receipt_path=
                        final_path,
                    release_dir=release,
                    repository_root=ROOT,
                    serial="physical-001",
                    adb_client=
                        DEV.VN97AdbClient(
                            "adb",
                            executor=fake,
                        ),
                ),
        )
        assert not fake.commands

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (
            release,
            final_path,
            package_sha,
            final,
        ) = fixture_release(
            root,
            repository_commit=commit,
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
        )
        expect_failure(
            "acceptance receipt inside release directory",
            lambda:
                ACC.run_clean_device_acceptance(
                    final_receipt_path=
                        final_path,
                    release_dir=release,
                    repository_root=ROOT,
                    serial="physical-001",
                    output_path=
                        release
                        / "device.vn97accept1",
                    adb_client=
                        DEV.VN97AdbClient(
                            "adb",
                            executor=fake,
                        ),
                ),
        )
        assert not fake.commands

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (
            release,
            final_path,
            package_sha,
            final,
        ) = fixture_release(
            root,
            repository_commit=commit,
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
            transient_ui_failures=1,
            transient_boot_failures=1,
            transient_service_failures=1,
        )
        adb = DEV.VN97AdbClient(
            "adb",
            executor=fake,
        )
        output = root / "accepted.vn97accept1"
        receipt = (
            ACC.run_clean_device_acceptance(
                final_receipt_path=
                    final_path,
                release_dir=release,
                repository_root=ROOT,
                serial="physical-001",
                output_path=output,
                adb_client=adb,
                launch_timeout_seconds=10,
                reboot_timeout_seconds=10,
                selftest_timeout_seconds=10,
            )
        )
        assert receipt.cleanup_state == "UNINSTALLED"
        assert (
            ACC.parse_turnkey_acceptance_receipt(
                output.read_bytes()
            )
            == receipt
        )
        assert not fake.installed
        assert (
            receipt
            .bootstrap_package_sha256
            == package_sha
        )
        assert (
            receipt.pre_boot_id_sha256
            != receipt.post_boot_id_sha256
        )
        commands = [
            " ".join(item)
            for item in fake.commands
        ]
        assert any(
            "install --no-streaming"
            in item
            for item in commands
        )
        assert any(
            ACC.ACTION_BEGIN
            in item
            for item in commands
        )
        assert any(
            ACC.ACTION_VERIFY_AFTER_REBOOT
            in item
            for item in commands
        )
        assert any(
            item.endswith(" reboot")
            for item in commands
        )
        assert any(
            "uninstall ai.vn97.app"
            in item
            for item in commands
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (
            release,
            final_path,
            package_sha,
            final,
        ) = fixture_release(
            root,
            repository_commit=commit,
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
            fail_pre_selftest=True,
        )
        output = (
            root
            / "failed.vn97accept1"
        )
        expect_failure(
            "failed keep-installed acceptance cleanup",
            lambda:
                ACC.run_clean_device_acceptance(
                    final_receipt_path=
                        final_path,
                    release_dir=release,
                    repository_root=ROOT,
                    serial="physical-001",
                    output_path=output,
                    keep_installed=True,
                    adb_client=
                        DEV.VN97AdbClient(
                            "adb",
                            executor=fake,
                        ),
                    launch_timeout_seconds=10,
                    reboot_timeout_seconds=10,
                    selftest_timeout_seconds=10,
                ),
        )
        assert not fake.installed
        assert not output.exists()
        commands = [
            " ".join(item)
            for item in fake.commands
        ]
        assert any(
            "uninstall ai.vn97.app"
            in item
            for item in commands
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        release, final_path, package_sha, final = (
            fixture_release(
                root,
                repository_commit=commit,
            )
        )
        fake = FakeExecutor(
            apk_version_code=
                final.version_code,
            apk_version_name=
                final.version_name,
            package_sha256=
                package_sha,
            emulator=True,
        )
        expect_failure(
            "emulator acceptance",
            lambda:
                ACC.run_clean_device_acceptance(
                    final_receipt_path=
                        final_path,
                    release_dir=release,
                    repository_root=ROOT,
                    serial="emulator-001",
                    adb_client=
                        DEV.VN97AdbClient(
                            "adb",
                            executor=fake,
                        ),
                    launch_timeout_seconds=10,
                    reboot_timeout_seconds=10,
                    selftest_timeout_seconds=10,
                ),
        )
        assert not any(
            " install " in
            (" " + " ".join(command) + " ")
            for command in fake.commands
        )

    manifest = (
        ROOT
        / "android/app/src/main/AndroidManifest.xml"
    ).read_text(
        encoding="utf-8"
    )
    assert (
        ".VN97TurnkeyAcceptanceReceiver"
        in manifest
    )
    assert (
        'android:permission="android.permission.DUMP"'
        in manifest
    )

    receiver = (
        ROOT
        / "android/app/src/main/java/ai/vn97/app/VN97TurnkeyAcceptanceReceiver.kt"
    ).read_text(
        encoding="utf-8"
    )
    turnkey_gate = receiver.index(
        "BuildConfig"
    )
    start_goal = receiver.index(
        ".startGoal("
    )
    fixed_goal = receiver.index(
        "private fun acceptanceGoal("
    )
    assert (
        turnkey_gate
        < start_goal
        < fixed_goal
    )
    assert "EXTRA_GOAL" not in receiver

    print(
        "M19M clean-device turnkey acceptance contracts: PASS"
    )


if __name__ == "__main__":
    main()
