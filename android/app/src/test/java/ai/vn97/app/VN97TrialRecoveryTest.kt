package ai.vn97.app

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class VN97TrialRecoveryTest {
    private fun report() = JSONObject().put("writer_pid", 101).put("status", "running")
        .put("execution_passed", false).put("generated_text", "one token")
    @Test fun recordsInterruptedNativeRunAndPreservesOutput() {
        val value = VN97TrialRecovery.recover(report(), 102, JSONObject().put("reason", "native_crash"))
        assertEquals("interrupted", value.getString("status"))
        assertFalse(value.getBoolean("execution_passed"))
        assertEquals("one token", value.getString("generated_text"))
        assertEquals("native_crash", value.getJSONObject("process_exit").getString("reason"))
    }
    @Test fun unknownReasonDoesNotClaimOutOfMemory() {
        val value = VN97TrialRecovery.recover(report(), 102, null)
        assertEquals("unavailable", value.getJSONObject("process_exit").getString("reason"))
    }
    @Test fun currentProcessIsNotMarkedDead() { assertFalse(VN97TrialRecovery.interrupted(report(), 101)) }
    @Test fun completedRunIsNotReclassified() {
        assertFalse(VN97TrialRecovery.interrupted(report().put("execution_passed", true).put("status", "completed"), 102))
    }
    @Test fun handledFailureAndCancellationAreNotReclassified() {
        for (state in listOf("failed", "cancelled")) assertFalse(VN97TrialRecovery.interrupted(report().put("status", state), 102))
    }
}
