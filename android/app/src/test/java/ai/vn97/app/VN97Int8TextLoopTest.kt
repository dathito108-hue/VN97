package ai.vn97.app

import ai.vn97.runtime.VN97Int8TextLoop
import java.util.concurrent.CancellationException
import org.junit.Assert.*
import org.junit.Test

class VN97Int8TextLoopTest {
    private fun row(id: Int) = FloatArray(50288).also { it[id] = 10f }
    @Test fun chunkingContinuationAndEos() {
        val seen = mutableListOf<IntArray>()
        val result = VN97Int8TextLoop.generate(IntArray(10) { it + 100 }, 0, { false }, {
            seen.add(it)
            row(if (seen.size < 3) 42 else 0)
        }, {})
        assertArrayEquals(intArrayOf(42), result)
        assertEquals(listOf(8, 2, 1), seen.map { it.size })
        assertArrayEquals(intArrayOf(108, 109), seen[1])
        assertArrayEquals(intArrayOf(42), seen[2])
    }
    @Test fun paddedVocabularyCannotBeGeneratedAndOutputIsBounded() {
        var calls = 0
        val result = VN97Int8TextLoop.generate(intArrayOf(100), 0, { false }, {
            calls++
            row(42).also { it[50287] = 100f }
        }, {})
        assertEquals(16, result.size)
        assertTrue(result.all { it == 42 })
        assertEquals(16, calls)
    }
    @Test fun cancellationDoesNotStartNextInference() {
        var stop = false; var calls = 0
        assertThrows(CancellationException::class.java) {
            VN97Int8TextLoop.generate(intArrayOf(100), 0, { stop }, { calls++; row(42) }, { stop = true })
        }
        assertEquals(1, calls)
    }
    @Test fun oversizedPromptRejectedBeforeInference() {
        assertThrows(IllegalArgumentException::class.java) {
            VN97Int8TextLoop.generate(IntArray(129), 0, { false }, { error("Must not execute") }, {})
        }
    }
    @Test fun nonfiniteLogitsRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            VN97Int8TextLoop.generate(intArrayOf(100), 0, { false }, { row(42).also { it[3] = Float.NaN } }, {})
        }
    }
    @Test fun eosDoesNotLeakIntoOutput() {
        val result = VN97Int8TextLoop.generate(intArrayOf(100), 0, { false }, { row(0) }, { error("No output expected") })
        assertEquals(0, result.size)
    }
}
