package ai.vn97.platform

import java.nio.file.Files
import java.nio.file.StandardOpenOption

private inline fun expectM18CFailure(
    label: String,
    block: () -> Unit,
) {
    check(runCatching(block).isFailure) {
        "expected M18C failure: $label"
    }
}

fun main() {
    val root =
        Files.createTempDirectory(
            "vn97-m18c-health"
        ).toFile()
    try {
        val store =
            VN97ExecutionHealthStore(root)

        var now = 1_000_000L
        val first =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .CONTINUATION_JOB,
                key = "97",
                nowWallTimeMillis = now,
                maxRunMillis = 10_000L,
            )
        check(!first.suppressed)
        val firstLease =
            checkNotNull(first.lease)
        check(first.record.running)
        check(first.record.generation == 1L)

        val overlap =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .CONTINUATION_JOB,
                key = "97",
                nowWallTimeMillis =
                    now + 1_000L,
                maxRunMillis = 10_000L,
            )
        check(overlap.suppressed)
        check(overlap.lease == null)
        check(
            overlap.record.generation ==
                firstLease.generation
        )

        val success =
            store.finish(
                lease = firstLease,
                state =
                    VN97ExecutionHealthState
                        .SUCCEEDED,
                nowWallTimeMillis =
                    now + 2_000L,
            )
        check(
            success.state ==
                VN97ExecutionHealthState
                    .SUCCEEDED
        )
        check(success.consecutiveFailures == 0)

        fun failOnce(
            generationExpected: Long,
        ): VN97ExecutionHealthRecord {
            now += 20_000L
            val begin =
                store.begin(
                    domain =
                        VN97ExecutionHealthDomain
                            .CONTINUATION_JOB,
                    key = "97",
                    nowWallTimeMillis = now,
                    maxRunMillis = 5_000L,
                )
            check(!begin.suppressed)
            check(
                begin.record.generation ==
                    generationExpected
            )
            return store.finish(
                lease =
                    checkNotNull(
                        begin.lease
                    ),
                state =
                    VN97ExecutionHealthState
                        .FAILED,
                nowWallTimeMillis =
                    now + 100L,
                detail = "synthetic failure",
            )
        }

        check(
            failOnce(2L)
                .consecutiveFailures == 1
        )
        check(
            failOnce(3L)
                .consecutiveFailures == 2
        )
        val thirdFailure =
            failOnce(4L)
        check(
            thirdFailure
                .consecutiveFailures == 3
        )
        check(
            thirdFailure
                .suppressUntilWallTimeMillis >
                thirdFailure
                    .updatedWallTimeMillis
        )

        val cooled =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .CONTINUATION_JOB,
                key = "97",
                nowWallTimeMillis =
                    thirdFailure
                        .updatedWallTimeMillis +
                        1_000L,
                maxRunMillis = 5_000L,
            )
        check(cooled.suppressed)

        now =
            thirdFailure
                .suppressUntilWallTimeMillis +
                1L
        val afterCooldown =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .CONTINUATION_JOB,
                key = "97",
                nowWallTimeMillis = now,
                maxRunMillis = 5_000L,
            )
        check(!afterCooldown.suppressed)
        val cancelled =
            store.finish(
                lease =
                    checkNotNull(
                        afterCooldown.lease
                    ),
                state =
                    VN97ExecutionHealthState
                        .CANCELLED,
                nowWallTimeMillis =
                    now + 100L,
                detail = "system stop",
            )
        check(
            cancelled.consecutiveFailures ==
                thirdFailure
                    .consecutiveFailures
        )

        val otherKey =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .PAPER_TRADING_JOB,
                key = "123",
                nowWallTimeMillis = now,
                maxRunMillis = 1_000L,
            )
        val staleAt =
            checkNotNull(
                otherKey.lease
            ).deadlineWallTimeMillis +
                VN97ExecutionHealthPolicy
                    .STALE_GRACE_MILLIS +
                1L
        val recovered =
            store.begin(
                domain =
                    VN97ExecutionHealthDomain
                        .PAPER_TRADING_JOB,
                key = "123",
                nowWallTimeMillis = staleAt,
                maxRunMillis = 1_000L,
            )
        check(!recovered.suppressed)
        check(recovered.record.generation == 2L)
        check(
            recovered.record
                .consecutiveFailures == 1
        )

        val hugeDetail =
            "é".repeat(10_000)
        val bounded =
            store.finish(
                lease =
                    checkNotNull(
                        recovered.lease
                    ),
                state =
                    VN97ExecutionHealthState
                        .FAILED,
                nowWallTimeMillis =
                    staleAt + 10L,
                detail = hugeDetail,
            )
        check(
            bounded.detail.toByteArray(
                Charsets.UTF_8
            ).size <=
                VN97ExecutionHealthRecord
                    .MAX_DETAIL_BYTES
        )

        val target =
            root.listFiles()
                ?.firstOrNull {
                    it.name.endsWith(
                        ".vn97health1"
                    )
                }
                ?: error(
                    "health evidence missing"
                )
        Files.write(
            target.toPath(),
            byteArrayOf(0x41),
            StandardOpenOption.APPEND,
        )
        expectM18CFailure(
            "tampered health evidence"
        ) {
            val key =
                if (
                    target.name ==
                        root.listFiles()
                            ?.firstOrNull()
                            ?.name
                ) {
                    "97"
                } else {
                    "123"
                }
            store.loadOrNull(
                VN97ExecutionHealthDomain
                    .CONTINUATION_JOB,
                key,
            )
        }

        println(
            "M18C execution health contracts: PASS"
        )
    } finally {
        root.deleteRecursively()
    }
}
