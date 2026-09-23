package ai.vn97.platform

import java.nio.charset.StandardCharsets

private fun expectM15CFailure(label: String, block: () -> Unit) {
    check(runCatching(block).isFailure) {
        "expected M15C failure: $label"
    }
}

private fun feed(
    source: String = "fixture.market",
    observedNs: Long = 1_000L,
    symbol: String = "ABC",
    bid: Long = 99_000_000L,
    ask: Long = 100_000_000L,
    quoteTimestampNs: Long = 995L,
): ByteArray =
    (
        "VN97MKTFEED1\n" +
            "source=$source\n" +
            "observed_ns=$observedNs\n" +
            "Q|$symbol|$bid|$ask|$quoteTimestampNs\n"
    ).toByteArray(StandardCharsets.UTF_8)

private fun snapshot(
    observedNs: Long,
    bid: Long,
    ask: Long,
    quoteTimestampNs: Long,
): VN97MarketSnapshot =
    VN97MarketSnapshot.create(
        sourceId = "fixture.market",
        observedNs = observedNs,
        quotes = listOf(
            VN97MarketQuote(
                symbol = "ABC",
                bidPriceMicros = bid,
                askPriceMicros = ask,
                timestampNs = quoteTimestampNs,
            )
        ),
        maxQuoteAgeNs = 20L,
    )

fun main() {
    val policy = VN97MarketDataPolicy(
        sourceId = "fixture.market",
        allowedSymbols = setOf("ABC", "XYZ"),
        maxResponseBytes = 4 * 1024,
        maxObservationAgeNs = 50L,
        maxQuoteAgeNs = 20L,
        connectTimeoutMs = 1_000,
        readTimeoutMs = 1_000,
    )

    var transportCalls = 0
    val transport = VN97MarketDataTransport {
            endpoint,
            maxBytes,
            connectTimeoutMs,
            readTimeoutMs,
        ->
        transportCalls += 1
        check(endpoint.toString() == "https://market.example/feed")
        check(maxBytes == policy.maxResponseBytes)
        check(connectTimeoutMs == policy.connectTimeoutMs)
        check(readTimeoutMs == policy.readTimeoutMs)
        feed()
    }
    val source = VN97HttpsMarketDataSource(
        endpoint = "https://market.example/feed",
        policy = policy,
        transport = transport,
    )
    val decoded = source.fetch(nowNs = 1_010L)
    check(transportCalls == 1)
    check(decoded.sourceId == "fixture.market")
    check(decoded.observedNs == 1_000L)
    check(decoded.quote("ABC")?.askPriceMicros == 100_000_000L)

    expectM15CFailure("cleartext endpoint") {
        VN97HttpsMarketDataSource(
            endpoint = "http://market.example/feed",
            policy = policy,
            transport = transport,
        )
    }
    expectM15CFailure("endpoint userinfo") {
        VN97HttpsMarketDataSource(
            endpoint = "https://user:pass@market.example/feed",
            policy = policy,
            transport = transport,
        )
    }
    expectM15CFailure("wrong source") {
        VN97MarketFeedCodec.decode(
            bytes = feed(source = "other.market"),
            policy = policy,
            nowNs = 1_010L,
        )
    }
    expectM15CFailure("unallowlisted symbol") {
        VN97MarketFeedCodec.decode(
            bytes = feed(symbol = "BAD"),
            policy = policy,
            nowNs = 1_010L,
        )
    }
    expectM15CFailure("stale observation") {
        VN97MarketFeedCodec.decode(
            bytes = feed(observedNs = 1_000L),
            policy = policy,
            nowNs = 1_051L,
        )
    }
    expectM15CFailure("future observation") {
        VN97MarketFeedCodec.decode(
            bytes = feed(observedNs = 1_100L, quoteTimestampNs = 1_095L),
            policy = policy,
            nowNs = 1_050L,
        )
    }
    expectM15CFailure("stale quote") {
        VN97MarketFeedCodec.decode(
            bytes = feed(quoteTimestampNs = 970L),
            policy = policy,
            nowNs = 1_010L,
        )
    }
    expectM15CFailure("invalid UTF-8") {
        VN97MarketFeedCodec.decode(
            bytes = byteArrayOf(0xC3.toByte(), 0x28),
            policy = policy,
            nowNs = 1_010L,
        )
    }

    val oversizedTransport = VN97MarketDataTransport { _, maxBytes, _, _ ->
        ByteArray(maxBytes + 1) { 'A'.code.toByte() }
    }
    expectM15CFailure("transport byte-bound violation") {
        VN97HttpsMarketDataSource(
            endpoint = "https://market.example/feed",
            policy = policy,
            transport = oversizedTransport,
        ).fetch(1_010L)
    }

    val first = snapshot(
        observedNs = 1_000L,
        bid = 99_000_000L,
        ask = 100_000_000L,
        quoteTimestampNs = 995L,
    )
    val second = snapshot(
        observedNs = 2_000L,
        bid = 100_000_000L,
        ask = 101_000_000L,
        quoteTimestampNs = 1_995L,
    )
    val snapshots = ArrayDeque(listOf(first, second))
    val runner = VN97PaperTradingEpisodeRunner(
        source = VN97MarketDataSource {
            check(snapshots.isNotEmpty())
            snapshots.removeFirst()
        },
        evaluator = { marketSnapshot, evaluatedNs ->
            marketSnapshot.snapshotId + ":" + evaluatedNs
        },
        limits = VN97PaperTradingEpisodeLimits(
            maxEpisodeAttempts = 2,
            minEpisodeSpacingNs = 100L,
        ),
    )

    val firstEpisode = runner.runNext(1_010L)
    check(firstEpisode.attempt == 1)
    check(firstEpisode.snapshotId == first.snapshotId)
    check(runner.attemptsUsed() == 1)
    check(runner.attemptsRemaining() == 1)

    expectM15CFailure("minimum episode spacing") {
        runner.runNext(1_050L)
    }
    check(runner.attemptsUsed() == 1)

    val secondEpisode = runner.runNext(2_010L)
    check(secondEpisode.attempt == 2)
    check(secondEpisode.snapshotId == second.snapshotId)
    check(runner.attemptsRemaining() == 0)

    expectM15CFailure("episode attempt budget") {
        runner.runNext(3_010L)
    }

    val duplicateRunner = VN97PaperTradingEpisodeRunner(
        source = VN97MarketDataSource { first },
        evaluator = { marketSnapshot, _ -> marketSnapshot.snapshotId },
        limits = VN97PaperTradingEpisodeLimits(
            maxEpisodeAttempts = 3,
            minEpisodeSpacingNs = 0L,
        ),
    )
    duplicateRunner.runNext(1_010L)
    expectM15CFailure("duplicate snapshot replay") {
        duplicateRunner.runNext(1_020L)
    }

    val sameTimeNewSnapshot = snapshot(
        observedNs = 1_000L,
        bid = 98_000_000L,
        ask = 99_000_000L,
        quoteTimestampNs = 995L,
    )
    val observationQueue = ArrayDeque(listOf(first, sameTimeNewSnapshot))
    val monotonicRunner = VN97PaperTradingEpisodeRunner(
        source = VN97MarketDataSource { observationQueue.removeFirst() },
        evaluator = { marketSnapshot, _ -> marketSnapshot.snapshotId },
        limits = VN97PaperTradingEpisodeLimits(
            maxEpisodeAttempts = 2,
            minEpisodeSpacingNs = 0L,
        ),
    )
    monotonicRunner.runNext(1_010L)
    expectM15CFailure("observation did not advance") {
        monotonicRunner.runNext(1_020L)
    }

    println("M15C read-only market data and fresh episode contracts: PASS")
}
