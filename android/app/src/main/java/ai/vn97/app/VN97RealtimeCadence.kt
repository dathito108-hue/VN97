package ai.vn97.app

data class VN97RealtimeCadence(
    val cognitionHz: Int,
    val reflexHz: Int,
) {
    init {
        require(cognitionHz in 1..5)
        require(reflexHz in 30..60)
    }

    val cognitionPeriodNs: Long
        get() = 1_000_000_000L / cognitionHz

    val reflexPeriodNs: Long
        get() = 1_000_000_000L / reflexHz
}

enum class VN97RealtimePressure {
    NORMAL,
    MODERATE,
    SEVERE,
}

object VN97RealtimeCadencePolicy {
    fun choose(
        thermal: VN97RealtimePressure,
        memory: VN97RealtimePressure,
    ): VN97RealtimeCadence =
        when {
            thermal == VN97RealtimePressure.SEVERE ||
                memory == VN97RealtimePressure.SEVERE ->
                VN97RealtimeCadence(1, 30)
            thermal == VN97RealtimePressure.MODERATE ||
                memory == VN97RealtimePressure.MODERATE ->
                VN97RealtimeCadence(2, 45)
            else ->
                VN97RealtimeCadence(4, 60)
        }
}

class VN97RealtimeGate(
    private var cadence: VN97RealtimeCadence,
) {
    private var nextCognitionNs = 0L
    private var nextReflexNs = 0L

    @Synchronized
    fun update(value: VN97RealtimeCadence) {
        cadence = value
    }

    @Synchronized
    fun claimCognition(nowNs: Long): Boolean {
        require(nowNs >= 0L)
        if (nowNs < nextCognitionNs) return false
        nextCognitionNs =
            Math.addExact(nowNs, cadence.cognitionPeriodNs)
        return true
    }

    @Synchronized
    fun claimReflex(nowNs: Long): Boolean {
        require(nowNs >= 0L)
        if (nowNs < nextReflexNs) return false
        nextReflexNs =
            Math.addExact(nowNs, cadence.reflexPeriodNs)
        return true
    }
}
