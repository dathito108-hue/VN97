"""Dependency-free regression gate for the single app intelligence route."""
from pathlib import Path
import ast
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class G06RouteTests(unittest.TestCase):
    def test_retired_implementations_cannot_be_loaded(self):
        for path in (
            "src/vn97/r2/model.py", "src/vn97/r2/ssm.py",
            "research/vn97_native/language_candidate.py",
            "research/vn97_native/portable_runtime.py",
            "android/runtime/src/main/java/ai/vn97/runtime/R2CognitionInference.kt",
            "android/runtime/src/main/java/ai/vn97/runtime/OrtProductionExecutor.kt",
            "android/runtime/src/main/java/ai/vn97/runtime/AdaptiveStateOrtCore.kt",
        ):
            self.assertFalse((ROOT / path).exists(), path)

    def test_retained_python_dependencies_and_entrypoints_resolve(self):
        base = ROOT / "src/vn97/r2"
        for path in base.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                    self.assertTrue((base / (node.module.split('.')[0] + '.py')).is_file(),
                                    (path.name, node.module))
        scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
        for command, target in scripts.items():
            if target.startswith("vn97.r2."):
                module = target.split(':')[0]
                self.assertTrue((ROOT / "src" / (module.replace('.', '/') + '.py')).is_file(), command)

    def test_live_routes_do_not_open_legacy_inference(self):
        files = [
            "android/platform/src/main/java/ai/vn97/platform/AndroidPlatformRuntime.kt",
            "android/platform/src/main/java/ai/vn97/platform/VN97AutonomousSeed.kt",
            "android/platform/src/main/java/ai/vn97/platform/VN97OnDeviceEvidence.kt",
            "android/app/src/main/java/ai/vn97/app/VN97AppAssistant.kt",
            "android/app/src/main/java/ai/vn97/app/VN97GameAgentService.kt",
            "android/app/src/main/java/ai/vn97/app/VN97PaperTradingSessionManager.kt",
            "android/app/src/main/java/ai/vn97/app/VN97AutonomousWorkManager.kt",
        ]
        for path in files:
            source = (ROOT / path).read_text()
            for forbidden in ("VN97R2CognitionInference", "VN97OrtProductionExecutor",
                              "NativeActivatedInventoryModelLoader", "NativeCognitionInferenceEngine",
                              "NativeRuntimeSession.create("):
                self.assertNotIn(forbidden, source, path)
        source = (ROOT / "android/runtime/src/main/java/ai/vn97/runtime/G06CognitionInference.kt").read_text()
        self.assertIn("promotion.requireCompatible", source)
        self.assertIn("binding.requireCompatible", source)
        self.assertIn("VN97Mamba2CognitionCandidate.open", source)

    def test_main_activity_has_no_legacy_model_activation_route(self):
        path = "android/app/src/main/java/ai/vn97/app/VN97MainActivity.kt"
        source = (ROOT / path).read_text()
        for forbidden in (
            "app.provisioner", "VN97MI1", "model.vn97cap1",
            "REQUEST_MODEL_ARCHIVE", "EXTRA_M19J_SPEECH_FRAMES",
            "VN97SelfImprovementActivity::class.java",
        ):
            self.assertNotIn(forbidden, source, (path, forbidden))
        self.assertIn("app.assistant.openIfActivated()", source)
        self.assertIn(".collectMobileEvidence(", source)

        evidence_path = (
            "android/platform/src/main/java/ai/vn97/platform/"
            "VN97OnDeviceEvidence.kt"
        )
        evidence = (ROOT / evidence_path).read_text()
        self.assertNotIn("speechFrames", evidence)
        self.assertNotIn("NativePreparedAudio", evidence)
        self.assertIn('speechPrefill = null', evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
