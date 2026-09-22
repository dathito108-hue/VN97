import hashlib
from pathlib import Path
import struct
import tempfile

import pytest

from vn97.capability_package import (
    CapabilityDataSection, CapabilityManifestError, CapabilityPackageIntegrityError,
    CapabilitySource, CapabilityStager, build_capability_package, parse_capability_package,
)

def package():
    source = CapabilitySource("local-import", hashlib.sha256(b"source").hexdigest(), "test-only")
    return build_capability_package(
        capability_id="vision.edge", capability_version=1, kind="multimodal", source=source,
        sections=(
            CapabilityDataSection("weights", "VN97T2", b"weights-data"),
            CapabilityDataSection("metadata", "VN97META1", b"meta-data"),
        ),
    )

def test_round_trip_and_manifest_binding():
    blob=package(); parsed=parse_capability_package(blob)
    assert parsed.manifest.capability_id=="vision.edge"
    assert parsed.manifest.kind=="multimodal"
    assert parsed.sections==(b"weights-data",b"meta-data")
    assert parsed.package_sha256==hashlib.sha256(blob).hexdigest()
    assert parsed.manifest.sections[0]["format"]=="VN97T2"

def test_tamper_fails_integrity():
    blob=bytearray(package()); blob[-1]^=1
    with pytest.raises(CapabilityPackageIntegrityError): parse_capability_package(bytes(blob))

def test_header_tamper_fails_integrity():
    blob=bytearray(package()); total=struct.unpack_from("<Q",blob,40)[0]; struct.pack_into("<Q",blob,40,total+1)
    with pytest.raises(Exception): parse_capability_package(bytes(blob))

def test_executable_roles_are_rejected():
    with pytest.raises(ValueError): CapabilityDataSection("executable","ELF",b"x")

def test_staging_is_content_addressed_idempotent_and_not_activation():
    blob=package()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td).resolve(); stager=CapabilityStager(root,max_package_bytes=1024*1024)
        first=stager.stage(blob); second=stager.stage(blob)
        assert first.path==second.path
        assert first.path.name==hashlib.sha256(blob).hexdigest()+".vn97cap1"
        assert first.path.read_bytes()==blob
        assert first.manifest.capability_id=="vision.edge"

def test_manifest_source_hash_and_id_are_strict():
    with pytest.raises(ValueError): CapabilitySource("x","ABC","license")
    source=CapabilitySource("x",hashlib.sha256(b"x").hexdigest(),"license")
    with pytest.raises(CapabilityManifestError):
        # Builder reaches strict manifest validation for an invalid capability ID.
        build_capability_package(capability_id="Bad ID",capability_version=1,kind="knowledge",source=source,sections=(CapabilityDataSection("data","RAW",b"x"),))
