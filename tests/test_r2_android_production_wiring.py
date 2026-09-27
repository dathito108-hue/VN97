from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_production_assistant_has_no_legacy_inference_callsite() -> None:
    source = _read(
        "android/app/src/main/java/ai/vn97/app/VN97AppAssistant.kt"
    )
    assert "NativeCognitionInferenceEngine" not in source
    assert ".openCognition(opened)" in source
    assert "legacy model inference fallback is disabled" in source


def test_autonomous_continuity_is_bound_to_r2_identity() -> None:
    source = _read(
        "android/app/src/main/java/ai/vn97/app/VN97AutonomousWorkManager.kt"
    )
    assert "r2Continuity.bind(jobId)" in source
    assert "r2Continuity.requireCurrent(context.jobId)" in source
    assert "inference = r2Inference" in source


def test_paper_trading_does_not_construct_legacy_cognition_engine() -> None:
    source = _read(
        "android/app/src/main/java/ai/vn97/app/VN97PaperTradingSessionManager.kt"
    )
    assert "NativeCognitionInferenceEngine" not in source
    assert ".openCognition(model)" in source


def test_game_agent_uses_r2_and_two_clock_scheduler() -> None:
    source = _read(
        "android/app/src/main/java/ai/vn97/app/VN97GameAgentService.kt"
    )
    assert "NativeCognitionInferenceEngine" not in source
    assert ".openCognition(model)" in source
    assert "VN97RealtimeAgentScheduler" in source
    assert "VN97R2RealtimePerception" in source


def test_reflex_scheduler_has_no_model_dependency() -> None:
    source = _read(
        "android/app/src/main/java/ai/vn97/app/VN97RealtimeAgentScheduler.kt"
    )
    assert "Cognition" not in source.replace("cognition", "")
    assert "NativeCognition" not in source
    assert "VN97R2CognitionInference" not in source
    assert "16L..34L" in source
    assert "200L..1_000L" in source
