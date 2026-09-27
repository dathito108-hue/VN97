package ai.vn97.app

fun main() {
    val normal = VN97RealtimeCadencePolicy.choose(
        VN97RealtimePressure.NORMAL,
        VN97RealtimePressure.NORMAL,
    )
    check(normal == VN97RealtimeCadence(4, 60))
    val moderate = VN97RealtimeCadencePolicy.choose(
        VN97RealtimePressure.MODERATE,
        VN97RealtimePressure.NORMAL,
    )
    check(moderate == VN97RealtimeCadence(2, 45))
    val severe = VN97RealtimeCadencePolicy.choose(
        VN97RealtimePressure.NORMAL,
        VN97RealtimePressure.SEVERE,
    )
    check(severe == VN97RealtimeCadence(1, 30))
    val gate = VN97RealtimeGate(normal)
    check(gate.claimCognition(1_000_000_000L))
    check(!gate.claimCognition(1_000_000_001L))
    check(gate.claimCognition(1_250_000_000L))
    check(gate.claimReflex(2_000_000_000L))
    check(!gate.claimReflex(2_000_000_001L))
}
