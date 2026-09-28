"""Exercise real packaging code without importing the training package facade."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Only package namespaces are supplied: every tested module is real source.
for name, path in (("vn97", ROOT / "src/vn97"), ("vn97.r2", ROOT / "src/vn97/r2")):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

from vn97.r2.mamba2_android_runtime import build_g06_runtime_descriptor, compile_g06_runtime_descriptor
from vn97.r2.mamba2_bundle_contract import verify_g05_bundle

fixture = runpy.run_path(str(ROOT / "tests/test_r2_mamba2_g06_android.py"))["_manifest"]


def seal(root, manifest):
    body = dict(manifest)
    body.pop("manifest_id", None)
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    manifest["manifest_id"] = hashlib.sha256(b"VN97M2G05ONNX1\0" + encoded).hexdigest()
    path = root / "manifest.vn97m2g05.json"
    path.write_text(json.dumps(manifest), encoding="ascii")
    return path


def bundle(root, dtype="float32"):
    manifest = fixture()
    manifest["state_contract"]["dtype"] = dtype
    if dtype == "float16":
        manifest["state_contract"]["total_bytes_per_batch"] //= 2
    for record in manifest["graph_files"]:
        content = ("inventory fixture: " + record["filename"]).encode()
        (root / record["filename"]).write_bytes(content)
        record["bytes"] = len(content)
        record["sha256"] = hashlib.sha256(content).hexdigest()
    seal(root, manifest)
    return manifest


class RuntimeContractTests(unittest.TestCase):
    def test_valid_fp16_and_fp32_keep_identity_and_activation_closed(self):
        for dtype in ("float16", "float32"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = bundle(root, dtype)
                descriptor = build_g06_runtime_descriptor(bundle_dir=root)
                self.assertEqual(descriptor, compile_g06_runtime_descriptor(manifest))
                self.assertEqual(descriptor["g05_manifest_id"], manifest["manifest_id"])
                self.assertEqual(descriptor["state_dtype"], dtype)
                self.assertIs(descriptor["production_activation_authorized"], False)
                body = dict(descriptor)
                identity = body.pop("runtime_id")
                encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
                self.assertEqual(identity, hashlib.sha256(b"VN97M2G06RUNTIME1\0" + encoded).hexdigest())

    def test_android_incompatible_contracts_do_not_write_descriptor(self):
        mutations = (
            lambda m: m["state_contract"].update(dtype="bfloat16"),
            lambda m: m["state_contract"].pop("dtype"),
            lambda m: m["state_contract"].update(batch_size=True),
            lambda m: m["state_contract"].update(total_bytes_per_batch=1),
            lambda m: m["state_contract"]["ssm_state_shape"].__setitem__(1, True),
            lambda m: m["config"].update(n_groups=True),
            lambda m: m.update(valid_length_max=16),
            lambda m: m.update(production_activation_authorized=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = bundle(root)
                mutate(manifest)
                seal(root, manifest)
                with self.assertRaises(ValueError):
                    build_g06_runtime_descriptor(bundle_dir=root)
                self.assertFalse((root / "runtime.vn97m2g06.json").exists())

    def test_filename_escape_and_control_characters_are_rejected(self):
        for name in ("../outside", "/tmp/outside", "nested/file", "nested\\file", ".", "..", " ", "a\x00b", "a\nb", "C:outside"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = bundle(root)
                manifest["graph_files"][1]["filename"] = name
                seal(root, manifest)
                with self.assertRaisesRegex(ValueError, "filename"):
                    verify_g05_bundle(root)
                with self.assertRaisesRegex(ValueError, "filename"):
                    compile_g06_runtime_descriptor(manifest)

    def test_modified_payload_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = bundle(root)
            path = root / manifest["graph_files"][0]["filename"]
            path.write_bytes(b"x" * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_g06_runtime_descriptor(bundle_dir=root)

    def test_modified_manifest_fails_identity_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = bundle(root)
            manifest["source_weight_sha256"] = "1" * 64
            (root / "manifest.vn97m2g05.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_g05_bundle(root)

    def test_duplicate_keys_and_nonfinite_json_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = bundle(root)
            path = root / "manifest.vn97m2g05.json"
            path.write_text(json.dumps(manifest)[:-1] + ',"schema":"VN97M2G05ONNX1"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                verify_g05_bundle(root)
            path.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "non-finite"):
                verify_g05_bundle(root)

    def test_symlinks_rejected_for_manifest_and_payload(self):
        for manifest_link in (True, False):
            with self.subTest(manifest=manifest_link), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = bundle(root)
                name = "manifest.vn97m2g05.json" if manifest_link else manifest["graph_files"][0]["filename"]
                path = root / name
                moved = root / "saved"
                path.rename(moved)
                path.symlink_to(moved)
                with self.assertRaises(ValueError):
                    verify_g05_bundle(root)

    def test_boolean_size_not_accepted_as_one_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = bundle(root)
            record = manifest["graph_files"][0]
            (root / record["filename"]).write_bytes(b"x")
            record.update(bytes=True, sha256=hashlib.sha256(b"x").hexdigest())
            seal(root, manifest)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                verify_g05_bundle(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
