#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")

checks = []

def require(condition: bool, label: str) -> None:
    checks.append((label, condition))
    if not condition:
        raise SystemExit("R2-F7 FAIL: " + label)

bridge = text("android/runtime/src/main/java/ai/vn97/runtime/R2CognitionInference.kt")
platform = text("android/platform/src/main/java/ai/vn97/platform/AndroidPlatformRuntime.kt")
assistant = text("android/app/src/main/java/ai/vn97/app/VN97AppAssistant.kt")
game = text("android/app/src/main/java/ai/vn97/app/VN97GameAgentService.kt")
trade = text("android/platform/src/main/java/ai/vn97/platform/VN97PaperTradingAgent.kt")
manager = text("android/app/src/main/java/ai/vn97/app/VN97PaperTradingSessionManager.kt")
build = text("android/app/build.gradle.kts")
installer = text("android/app/src/main/java/ai/vn97/app/VN97R2BundledRuntime.kt")
continuity = text("android/app/src/main/java/ai/vn97/app/VN97AutonomousWorkManager.kt")

require("VN97R2F2BRIDGE1" in bridge, "F2 identity-bound bridge")
require("VN97OrtProductionExecutor.open" in bridge, "F2 uses F1 ORT executor")
require("executor.prefill" in bridge and "executor.step" in bridge, "F2 prefill/step generation")
require("NativeCognitionInferenceEngine" not in platform, "F3 platform has no legacy cognition")
require("VN97R2CognitionInference.open" in platform, "F3 assistant/planner uses R2")
require("NativeCognitionInferenceEngine" not in game, "F5 game has no legacy cognition")
require("VN97R2CognitionInference.open" in game, "F5 game reasoning uses R2")
require("NativeCognitionInferenceEngine" not in trade, "F5 trading has no legacy cognition")
require("NativeCognitionInferenceEngine" not in manager, "F5 trading memory has no legacy cognition")
require("WAITING_APPROVAL" in continuity, "F4 approval restoration remains present")
require("resumeProductionAssistantContinuation" in continuity, "F4 continuation remains present")
require("legacy inference is disabled" in assistant, "F3 multimodal fail-closed boundary")
require("verifyR2TurnkeyRuntime" in build, "F6 release requires R2 assets")
require(".vn97-r2.staging" in installer and ".vn97-r2.backup" in installer, "F6 crash-safe migration")
require("runtime/step.onnx" in installer, "F6 step graph package gate")
require("chunk-[1-9][0-9]*" in installer, "F6 chunk graph package gate")

print("VN97 R2 F2-F7 static validation: PASS")
print("Validated gates:")
for label, _ in checks:
    print(" - " + label)
print("Physical/device truth gates (not claimed by CI):")
print(" - Samsung Galaxy S21 FE benchmark: PENDING_PHYSICAL_DEVICE")
print(" - production dense checkpoint / 1B training claim: PENDING_REAL_CHECKPOINT")
print(" - semantic vision/audio ONNX package: PENDING_MULTIMODAL_ONNX")
