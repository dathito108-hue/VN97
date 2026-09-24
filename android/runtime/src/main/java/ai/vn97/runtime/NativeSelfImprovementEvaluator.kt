package ai.vn97.runtime

import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.ln

object NativeSelfImprovementEvaluator {
    fun evaluateActivatedModel(
        model: NativeActivatedModel,
        suite: List<VN97HeldOutCase> =
            VN97HeldOutSuite.cases,
    ): VN97ModelEvaluationMetrics {
        require(suite.isNotEmpty()) {
            "held-out suite must not be empty"
        }
        require(model.info.hasTokenizer) {
            "held-out evaluation requires VN97TK1 tokenizer"
        }

        var targetTokens = 0L
        var targetUtf8Bytes = 0L
        var totalNll = 0.0
        var top1Correct = 0L
        val prefillNanos =
            ArrayList<Long>(suite.size)
        val started = System.nanoTime()

        for (item in suite) {
            val promptIds =
                model.encodeUtf8(
                    text = item.prompt,
                    addBos = true,
                    addText = true,
                    addEos = false,
                )
            require(promptIds.isNotEmpty()) {
                "held-out prompt encoded to no tokens: ${item.id}"
            }

            val targetIds =
                model.encodeUtf8(
                    text = item.target,
                    addBos = false,
                    addText = false,
                    addEos = false,
                )
            require(targetIds.isNotEmpty()) {
                "held-out target encoded to no tokens: ${item.id}"
            }

            val config =
                NativeRuntimeConfig(
                    layers = model.info.layers,
                    batch = 1,
                    dModel = model.info.dModel,
                    dState = model.info.dState,
                    recurrentBackend =
                        NativeBackend.AUTO,
                    packedBackend =
                        NativeBackend.AUTO,
                )

            NativeRuntimeSession
                .create(config)
                .use { session ->
                    session.activate()

                    val prefillStart =
                        System.nanoTime()
                    var logits =
                        session.prefill(
                            model,
                            promptIds,
                        )
                    val prefillEnd =
                        System.nanoTime()
                    prefillNanos +=
                        maxOf(
                            0L,
                            prefillEnd -
                                prefillStart,
                        )

                    targetIds.forEachIndexed {
                            index,
                            targetId,
                        ->
                        require(
                            targetId in
                                logits.indices
                        ) {
                            "held-out target token is outside model vocabulary"
                        }

                        totalNll +=
                            tokenNegativeLogLikelihood(
                                logits,
                                targetId,
                            )
                        if (
                            argmax(logits) ==
                                targetId
                        ) {
                            top1Correct += 1L
                        }
                        targetTokens += 1L

                        if (
                            index !=
                                targetIds.lastIndex
                        ) {
                            logits =
                                session.inferStep(
                                    model,
                                    intArrayOf(
                                        targetId
                                    ),
                                )
                        }
                    }
                }

            targetUtf8Bytes +=
                item.target
                    .toByteArray(Charsets.UTF_8)
                    .size
                    .toLong()
        }

        val finished = System.nanoTime()
        return VN97ModelEvaluationMetrics(
            cases = suite.size,
            targetTokens = targetTokens,
            targetUtf8Bytes =
                targetUtf8Bytes,
            totalNegativeLogLikelihood =
                totalNll,
            top1Correct = top1Correct,
            prefillP95Nanos =
                percentile95(prefillNanos),
            totalEvaluationNanos =
                maxOf(1L, finished - started),
            imageBytes = model.info.imageBytes,
            hasAudioProjection =
                model.info.hasAudioProjection,
            hasVisionProjection =
                model.info.hasVisionProjection,
        )
    }

    fun evaluateCandidateDescriptor(
        fd: Int,
        offset: Long,
        length: Long,
        artifactSha256: ByteArray,
        suite: List<VN97HeldOutCase> =
            VN97HeldOutSuite.cases,
    ): VN97ModelEvaluationMetrics =
        NativeActivatedModel
            .openForEvaluation(
                fd = fd,
                offset = offset,
                length = length,
                expectedModelId =
                    artifactSha256,
            )
            .use { model ->
                evaluateActivatedModel(
                    model = model,
                    suite = suite,
                )
            }

    private fun tokenNegativeLogLikelihood(
        logits: FloatArray,
        targetId: Int,
    ): Double {
        require(logits.isNotEmpty()) {
            "evaluation logits must not be empty"
        }
        require(targetId in logits.indices)
        require(
            logits.all { it.isFinite() }
        ) {
            "evaluation logits must be finite"
        }

        var maximum =
            logits[0].toDouble()
        for (index in 1 until logits.size) {
            maximum =
                maxOf(
                    maximum,
                    logits[index].toDouble(),
                )
        }

        var sum = 0.0
        for (value in logits) {
            sum += exp(
                value.toDouble() -
                    maximum
            )
        }
        require(
            sum.isFinite() &&
                sum > 0.0
        ) {
            "evaluation log-sum-exp is invalid"
        }

        val nll =
            maximum +
                ln(sum) -
                logits[targetId].toDouble()
        require(
            nll.isFinite() &&
                nll >= 0.0
        ) {
            "evaluation NLL is invalid"
        }
        return nll
    }

    private fun argmax(
        values: FloatArray,
    ): Int {
        require(values.isNotEmpty())
        var index = 0
        var best = values[0]
        for (
            candidate in
                1 until values.size
        ) {
            if (
                values[candidate] >
                    best
            ) {
                index = candidate
                best = values[candidate]
            }
        }
        return index
    }

    private fun percentile95(
        values: List<Long>,
    ): Long {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val index =
            ceil(
                sorted.size * 0.95
            ).toInt()
                .coerceIn(
                    1,
                    sorted.size,
                ) - 1
        return sorted[index]
    }
}
