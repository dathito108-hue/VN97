package ai.vn97.app

import java.io.File
import java.nio.file.Files

private fun expectM15DFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M15D failure: $label"
    }
}

private val SESSION_ID = "11".repeat(32)
private val MODEL_ID = "22".repeat(32)

private fun baseRecord(
    jobId: Int = 0x60000001,
): VN97PaperTradingSessionRecord =
    VN97PaperTradingSessionRecord(
        jobId = jobId,
        sessionId = SESSION_ID,
        modelIdHex = MODEL_ID,
        endpoint = "https://market.example/feed",
        sourceId = "fixture.market",
        symbols = listOf("ABC", "XYZ"),
        userGoal = "Paper trade only when bounded evidence is sufficient.",
        state = VN97PaperTradingSessionState.SCHEDULED,
        createdWallTimeMillis = 1_000L,
        updatedWallTimeMillis = 1_000L,
        intervalMillis = 60_000L,
        nextRunWallTimeMillis = 1_000L,
        deadlineWallTimeMillis = 1_000_000L,
        maxEpisodes = 4,
        requiresBatteryNotLow = true,
        requiresCharging = false,
    )

fun main() {
    val root = Files.createTempDirectory("vn97-m15d-store").toFile()
    try {
        val store = VN97PaperTradingSessionStore(root)
        val initial = baseRecord()
        store.save(initial)
        check(store.loadOrNull(initial.jobId) == initial)
        check(store.list() == listOf(initial))
        check(store.contains(initial.jobId))

        val running = initial.copy(
            state = VN97PaperTradingSessionState.RUNNING,
            updatedWallTimeMillis = 2_000L,
            wakeCount = 1,
            scheduleAttemptCount = 1,
            lastScheduledWallTimeMillis = 1_500L,
        )
        store.save(running)

        val consumed = running.copy(
            updatedWallTimeMillis = 3_000L,
            episodesAttempted = 1,
            lastSnapshotId = "33".repeat(32),
            lastObservedNs = 9_000L,
            lastOutcome = "SNAPSHOT_CONSUMED",
        )
        store.save(consumed)
        check(store.loadOrNull(initial.jobId) == consumed)

        val paused = consumed.copy(
            state = VN97PaperTradingSessionState.PAUSED,
            updatedWallTimeMillis = 4_000L,
        )
        store.save(paused)
        val resumed = paused.copy(
            state = VN97PaperTradingSessionState.SCHEDULED,
            updatedWallTimeMillis = 5_000L,
            nextRunWallTimeMillis = 65_000L,
        )
        store.save(resumed)

        expectM15DFailure("identity mutation") {
            store.save(
                resumed.copy(
                    updatedWallTimeMillis = 6_000L,
                    endpoint = "https://other.example/feed",
                )
            )
        }
        expectM15DFailure("wake counter rollback") {
            store.save(
                resumed.copy(
                    updatedWallTimeMillis = 6_000L,
                    wakeCount = 0,
                )
            )
        }
        expectM15DFailure("episode rollback") {
            store.save(
                resumed.copy(
                    updatedWallTimeMillis = 6_000L,
                    episodesAttempted = 0,
                )
            )
        }
        expectM15DFailure("observation rollback") {
            store.save(
                resumed.copy(
                    updatedWallTimeMillis = 6_000L,
                    lastSnapshotId = "44".repeat(32),
                    lastObservedNs = 8_999L,
                )
            )
        }

        val stopped = resumed.copy(
            state = VN97PaperTradingSessionState.STOPPED,
            updatedWallTimeMillis = 7_000L,
            terminalReason = "paper session stopped by user",
        )
        store.save(stopped)
        check(store.loadOrNull(initial.jobId) == stopped)
        expectM15DFailure("terminal mutation") {
            store.save(
                stopped.copy(
                    updatedWallTimeMillis = 8_000L,
                    terminalReason = "changed terminal reason",
                )
            )
        }

        expectM15DFailure("cleartext endpoint") {
            baseRecord(jobId = 0x60000002).copy(
                endpoint = "http://market.example/feed",
            )
        }
        expectM15DFailure("URL credentials") {
            baseRecord(jobId = 0x60000002).copy(
                endpoint = "https://user:pass@market.example/feed",
            )
        }
        expectM15DFailure("non-canonical symbols") {
            baseRecord(jobId = 0x60000002).copy(
                symbols = listOf("XYZ", "ABC"),
            )
        }
        expectM15DFailure("terminal without reason") {
            baseRecord(jobId = 0x60000002).copy(
                state = VN97PaperTradingSessionState.FAILED,
            )
        }

        val tamperRoot =
            Files.createTempDirectory("vn97-m15d-tamper").toFile()
        try {
            val tamperStore = VN97PaperTradingSessionStore(tamperRoot)
            val record = baseRecord(jobId = 0x60000003)
            tamperStore.save(record)
            val file = File(
                File(tamperRoot, record.jobId.toString()),
                "session.vn97pts1",
            )
            val bytes = file.readBytes()
            bytes[bytes.lastIndex - 3] =
                (bytes[bytes.lastIndex - 3].toInt() xor 1).toByte()
            file.writeBytes(bytes)
            expectM15DFailure("digest tamper") {
                tamperStore.loadOrNull(record.jobId)
            }
        } finally {
            tamperRoot.deleteRecursively()
        }

        println("M15D durable paper session ledger contracts: PASS")
    } finally {
        root.deleteRecursively()
    }
}
