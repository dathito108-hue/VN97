from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT /
    "src/vn97/release_attestation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vn97_release_attestation_m19d",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(
        "could not load release_attestation.py"
    )
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Attestation = MODULE.VN97ApkAttestation
parse_release_manifest = MODULE.parse_release_manifest
parse_apk_attestation = MODULE.parse_apk_attestation
verify_apk_payload = MODULE.verify_apk_payload
parse_aapt_badging = MODULE.parse_aapt_badging
parse_apksigner = (
    MODULE.parse_apksigner_certificate_sha256
)
sha = MODULE.sha256_bytes


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def release_manifest(
    package: bytes,
    signature: bytes,
    publisher: bytes,
) -> bytes:
    version_b64 = base64.b64encode(
        b"1.0.0-rc1"
    ).decode("ascii")
    return (
        "VN97REL1\n"
        "application_id=ai.vn97.app\n"
        "version_code=190100\n"
        f"version_name_b64={version_b64}\n"
        f"model_bytes={len(package)}\n"
        f"model_sha256={sha(package)}\n"
        f"signature_bytes={len(signature)}\n"
        f"signature_sha256={sha(signature)}\n"
        f"publisher_bytes={len(publisher)}\n"
        f"publisher_sha256={sha(publisher)}\n"
    ).encode("utf-8")


def write_apk(
    path: Path,
    *,
    package: bytes,
    signature: bytes,
    publisher: bytes,
    manifest: bytes,
) -> None:
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            MODULE.APK_BOOTSTRAP_PACKAGE,
            package,
        )
        archive.writestr(
            MODULE.APK_BOOTSTRAP_SIGNATURE,
            signature,
        )
        archive.writestr(
            MODULE.APK_BOOTSTRAP_PUBLISHER,
            publisher,
        )
        archive.writestr(
            MODULE.APK_RELEASE_MANIFEST,
            manifest,
        )


def main() -> None:
    package = (
        b"VN97CAP1" +
        b"x" * 128
    )
    signature = b"s" * 64
    publisher = b"p" * 32
    manifest = release_manifest(
        package,
        signature,
        publisher,
    )

    parsed = parse_release_manifest(
        manifest
    )
    assert parsed.application_id == "ai.vn97.app"
    assert parsed.version_code == 190100
    assert parsed.version_name == "1.0.0-rc1"
    assert parsed.model_sha256 == sha(package)

    with tempfile.TemporaryDirectory() as tmp:
        apk = Path(tmp) / "fixture.apk"
        write_apk(
            apk,
            package=package,
            signature=signature,
            publisher=publisher,
            manifest=manifest,
        )
        verified, manifest_bytes = (
            verify_apk_payload(
                apk,
                expected_application_id=
                    "ai.vn97.app",
                expected_version_code=190100,
                expected_version_name=
                    "1.0.0-rc1",
                expected_package=package,
                expected_signature=signature,
                expected_publisher=publisher,
            )
        )
        assert verified == parsed
        assert manifest_bytes == manifest

        expect_failure(
            "staged package mismatch",
            lambda: verify_apk_payload(
                apk,
                expected_application_id=
                    "ai.vn97.app",
                expected_version_code=190100,
                expected_version_name=
                    "1.0.0-rc1",
                expected_package=
                    package + b"x",
                expected_signature=signature,
                expected_publisher=publisher,
            ),
        )

        duplicate = (
            Path(tmp) /
            "duplicate.apk"
        )
        with zipfile.ZipFile(
            duplicate,
            "w",
        ) as archive:
            archive.writestr(
                MODULE.APK_BOOTSTRAP_PACKAGE,
                package,
            )
            archive.writestr(
                MODULE.APK_BOOTSTRAP_PACKAGE,
                package,
            )
            archive.writestr(
                MODULE.APK_BOOTSTRAP_SIGNATURE,
                signature,
            )
            archive.writestr(
                MODULE.APK_BOOTSTRAP_PUBLISHER,
                publisher,
            )
            archive.writestr(
                MODULE.APK_RELEASE_MANIFEST,
                manifest,
            )
        expect_failure(
            "duplicate APK ZIP entries",
            lambda: verify_apk_payload(
                duplicate,
                expected_application_id=
                    "ai.vn97.app",
                expected_version_code=190100,
                expected_version_name=
                    "1.0.0-rc1",
                expected_package=package,
                expected_signature=signature,
                expected_publisher=publisher,
            ),
        )

    identity = parse_aapt_badging(
        "package: name='ai.vn97.app' "
        "versionCode='190100' "
        "versionName='1.0.0-rc1' "
        "compileSdkVersion='37'\n"
    )
    assert identity == (
        "ai.vn97.app",
        190100,
        "1.0.0-rc1",
    )

    cert_a = "11" * 32
    cert_b = "22" * 32
    certificates = parse_apksigner(
        "Signer #2 certificate SHA-256 digest: "
        + cert_b
        + "\nSigner #1 certificate SHA-256 digest: "
        + cert_a
        + "\n"
    )
    assert certificates == (
        cert_a,
        cert_b,
    )

    attestation = Attestation(
        application_id="ai.vn97.app",
        version_code=190100,
        version_name="1.0.0-rc1",
        apk_bytes=1234,
        apk_sha256="33" * 32,
        signer_certificate_sha256=(
            cert_a,
        ),
        release_candidate_manifest_sha256=
            "44" * 32,
        bootstrap_release_report_sha256=
            "55" * 32,
        bootstrap_package_sha256=
            "66" * 32,
        bootstrap_signature_sha256=
            "77" * 32,
        bootstrap_publisher_sha256=
            "88" * 32,
        release_manifest_sha256=
            "99" * 32,
    )
    encoded = attestation.to_bytes()
    assert parse_apk_attestation(
        encoded
    ) == attestation
    assert b'"schema":"VN97APK1"' in encoded
    assert encoded == (
        __import__("json")
        .dumps(
            attestation.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        .encode("utf-8")
    )

    expect_failure(
        "trailing release manifest data",
        lambda: parse_release_manifest(
            manifest + b"x"
        ),
    )

    print(
        "M19D signed APK attestation contracts: PASS"
    )


if __name__ == "__main__":
    main()
