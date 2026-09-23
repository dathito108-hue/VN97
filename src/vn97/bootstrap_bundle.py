from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Protocol

from .capability_package import (
    CapabilityDataSection,
    CapabilitySource,
    build_capability_package,
    parse_capability_package,
)
from .capability_trust import (
    SignatureEnvelope,
    parse_signature_envelope,
)
from .model import VN97LanguageCore
from .model_image import build_model_image
from .tokenizer import VN97TokenizerPackage


BOOTSTRAP_PACKAGE_NAME = "model.vn97cap1"
BOOTSTRAP_SIGNATURE_NAME = "model.vn97sig1"
BOOTSTRAP_PUBLISHER_KEY_NAME = "publisher.ed25519"
BOOTSTRAP_CAPABILITY_ID = "model.language"
BOOTSTRAP_SECTION_ROLE = "model_image"
BOOTSTRAP_SECTION_FORMAT = "VN97MI1"


class VN97BootstrapBundleError(RuntimeError):
    pass


class VN97BootstrapSigner(Protocol):
    @property
    def key_id(self) -> str: ...

    @property
    def public_key(self) -> bytes: ...

    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True)
class VN97BootstrapBundle:
    model_image: bytes
    capability_package: bytes
    signature_envelope: bytes
    publisher_public_key: bytes
    model_image_sha256: str
    package_sha256: str
    capability_version: int
    publisher_key_id: str

    def __post_init__(self) -> None:
        if not self.model_image.startswith(b"VN97MI1\0"):
            raise ValueError("bootstrap model image must be VN97MI1")
        if len(self.publisher_public_key) != 32:
            raise ValueError("publisher public key must be 32-byte Ed25519")
        if hashlib.sha256(self.model_image).hexdigest() != self.model_image_sha256:
            raise ValueError("model image SHA-256 does not match bundle bytes")
        if hashlib.sha256(self.capability_package).hexdigest() != self.package_sha256:
            raise ValueError("capability package SHA-256 does not match bundle bytes")


class Ed25519PrivateKeySigner:
    """Optional production signer backed by the cryptography package."""

    def __init__(self, key_id: str, raw_private_key: bytes) -> None:
        if not isinstance(raw_private_key, bytes) or len(raw_private_key) != 32:
            raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError as exc:
            raise VN97BootstrapBundleError(
                "Ed25519 signing requires the optional 'cryptography' package"
            ) from exc

        # SignatureEnvelope validates the publisher key_id syntax for us.
        SignatureEnvelope(
            key_id=key_id,
            package_sha256="0" * 64,
            capability_id=BOOTSTRAP_CAPABILITY_ID,
            capability_version=1,
            signature=bytes(64),
        )
        self._key_id = key_id
        self._private_key = Ed25519PrivateKey.from_private_bytes(
            raw_private_key
        )
        self._public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key(self) -> bytes:
        return self._public_key

    def sign(self, message: bytes) -> bytes:
        return self._private_key.sign(message)


def build_bootstrap_bundle(
    model: VN97LanguageCore,
    *,
    tokenizer: VN97TokenizerPackage,
    source: CapabilitySource,
    capability_version: int,
    signer: VN97BootstrapSigner,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> VN97BootstrapBundle:
    """Build the exact signed model bundle consumed by the M10J APK bootstrap path.

    The caller owns training/checkpoint semantics. This function never creates or
    mutates model weights; it only serializes the supplied canonical VN97 model.
    """
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if not isinstance(tokenizer, VN97TokenizerPackage):
        raise TypeError("tokenizer must be VN97TokenizerPackage")
    if type(capability_version) is not int or not 1 <= capability_version <= 0xFFFFFFFF:
        raise ValueError("capability_version must be in unsigned 32-bit range")

    public_key = signer.public_key
    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("signer public_key must be exactly 32 bytes")

    image = build_model_image(
        model,
        tokenizer=tokenizer,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    image_bytes = image.data
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    package = build_capability_package(
        capability_id=BOOTSTRAP_CAPABILITY_ID,
        capability_version=capability_version,
        kind="weights",
        source=source,
        sections=(
            CapabilityDataSection(
                BOOTSTRAP_SECTION_ROLE,
                BOOTSTRAP_SECTION_FORMAT,
                image_bytes,
            ),
        ),
    )
    parsed = parse_capability_package(package)
    if (
        parsed.manifest.capability_id != BOOTSTRAP_CAPABILITY_ID
        or parsed.manifest.capability_version != capability_version
        or parsed.manifest.kind != "weights"
        or len(parsed.sections) != 1
        or parsed.sections[0] != image_bytes
    ):
        raise VN97BootstrapBundleError(
            "built capability package failed bootstrap identity self-check"
        )

    unsigned = SignatureEnvelope(
        key_id=signer.key_id,
        package_sha256=parsed.package_sha256,
        capability_id=BOOTSTRAP_CAPABILITY_ID,
        capability_version=capability_version,
        signature=bytes(64),
    )
    signature = signer.sign(unsigned.signing_message())
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise VN97BootstrapBundleError(
            "bootstrap signer must return exactly 64 Ed25519 signature bytes"
        )
    envelope = SignatureEnvelope(
        key_id=signer.key_id,
        package_sha256=parsed.package_sha256,
        capability_id=BOOTSTRAP_CAPABILITY_ID,
        capability_version=capability_version,
        signature=signature,
    ).to_bytes()
    reparsed_envelope = parse_signature_envelope(envelope)
    if (
        reparsed_envelope.package_sha256 != parsed.package_sha256
        or reparsed_envelope.capability_id != BOOTSTRAP_CAPABILITY_ID
        or reparsed_envelope.capability_version != capability_version
        or reparsed_envelope.key_id != signer.key_id
    ):
        raise VN97BootstrapBundleError(
            "signature envelope failed bootstrap binding self-check"
        )

    return VN97BootstrapBundle(
        model_image=image_bytes,
        capability_package=package,
        signature_envelope=envelope,
        publisher_public_key=bytes(public_key),
        model_image_sha256=image_sha256,
        package_sha256=parsed.package_sha256,
        capability_version=capability_version,
        publisher_key_id=signer.key_id,
    )


def write_bootstrap_assets(
    bundle: VN97BootstrapBundle,
    root: str | os.PathLike[str],
) -> Path:
    """Write the three exact M10J assets with bounded, fsynced atomic file replace."""
    if not isinstance(bundle, VN97BootstrapBundle):
        raise TypeError("bundle must be VN97BootstrapBundle")

    requested_root = Path(root)
    if requested_root.is_symlink():
        raise VN97BootstrapBundleError(
            "bootstrap asset root must not be a symlink"
        )
    requested_root.mkdir(parents=True, exist_ok=True)
    if requested_root.is_symlink():
        raise VN97BootstrapBundleError(
            "bootstrap asset root must not be a symlink"
        )
    root_path = requested_root.resolve(strict=True)
    if not root_path.is_dir():
        raise VN97BootstrapBundleError(
            "bootstrap asset root must be a real directory"
        )

    assets = (
        (BOOTSTRAP_PACKAGE_NAME, bundle.capability_package),
        (BOOTSTRAP_SIGNATURE_NAME, bundle.signature_envelope),
        (BOOTSTRAP_PUBLISHER_KEY_NAME, bundle.publisher_public_key),
    )
    temps: list[tuple[Path, Path]] = []
    try:
        for name, data in assets:
            target = root_path / name
            if target.is_symlink():
                raise VN97BootstrapBundleError(
                    f"bootstrap asset target must not be a symlink: {name}"
                )
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{name}.",
                suffix=".tmp",
                dir=root_path,
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "wb", closefd=True) as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
            temps.append((temp_path, target))

        for temp_path, target in temps:
            os.replace(temp_path, target)

        dir_fd = os.open(root_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        for temp_path, _ in temps:
            temp_path.unlink(missing_ok=True)

    for name, expected in assets:
        actual = (root_path / name).read_bytes()
        if actual != expected:
            raise VN97BootstrapBundleError(
                f"bootstrap asset post-write verification failed: {name}"
            )
    return root_path
