package ai.vn97.app

import ai.vn97.runtime.VN97TrialMemory
import org.junit.Assert.*
import org.junit.Test

class VN97TrialMemoryTest {
    private val threshold = 128L * 1024 * 1024
    private val total = 8L * 1024 * 1024 * 1024
    @Test fun systemPressureAlwaysStops() {
        assertTrue(VN97TrialMemory.underPressure(VN97TrialMemory.Sample(total / 2, threshold, total, true)))
    }
    @Test fun reserveBoundaryIsExplicit() {
        val boundary = threshold + VN97TrialMemory.RESERVE_BYTES
        assertTrue(VN97TrialMemory.underPressure(VN97TrialMemory.Sample(boundary - 1, threshold, total, false)))
        assertFalse(VN97TrialMemory.underPressure(VN97TrialMemory.Sample(boundary, threshold, total, false)))
    }
    @Test fun adequateHeadroomContinues() {
        assertFalse(VN97TrialMemory.underPressure(VN97TrialMemory.Sample(total / 2, threshold, total, false)))
    }
}
