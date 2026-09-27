package ai.vn97.app

import android.os.SystemClock

data class VN97RealtimeCadence(
    val cognitionIntervalMillis: Long,
    val reflexIntervalMillis: Long,
) {
    init {
        require(cognitionIntervalMillis in 200L..1_000L) {
            "realtime cognition cadence must stay between 1 and 5 Hz"
        }
        require(reflexIntervalMillis in 16L..34L) {
            "realtime reflex cadence must stay near 30-60 Hz"
        }
    }
}

object VN97RealtimeCadencePolicy {
    fun from(
        decision: VN97RuntimeResourceDecision,
    ): VN97RealtimeCadence {
        check(decision.runnable) {
            decision.reason
        }
        return if (decision.maxInteractiveAdvances >= 8) {
            VN97RealtimeCadence(
                cognitionIntervalMillis = 200L,
                reflexIntervalMillis = 16L,
            )
        } else {
            VN97RealtimeCadence(
                cognitionIntervalMillis = 500L,
                reflexIntervalMillis = 33L,
            )
        }
    }
}

/**
 * Two-clock scheduler:
 * - cognition is rate-limited to 1-5 Hz;
 * - deterministic safety/reflex checks run at ~30-60 Hz.
 *
 * The reflex callback MUST NOT invoke model inference. It only validates or
 * executes already-authorized deterministic native actions.
 */
class VN97RealtimeAgentScheduler {
    private var lastCognitionElapsedMillis = Long.MIN_VALUE

    fun awaitCognitionSlot(
        cadence: VN97RealtimeCadence,
        reflexTick: () -> Unit,
    ) {
        val previous = lastCognitionElapsedMillis
        if (previous != Long.MIN_VALUE) {
            val due = Math.addExact(
                previous,
                cadence.cognitionIntervalMillis,
            )
            while (true) {
                val now = SystemClock.elapsedRealtime()
                if (now >= due) break
                reflexTick()
                val remaining = due - now
                Thread.sleep(
                    minOf(
                        cadence.reflexIntervalMillis,
                        remaining,
                    )
                )
            }
        }
        reflexTick()
        lastCognitionElapsedMillis = SystemClock.elapsedRealtime()
    }
}
