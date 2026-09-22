import ai.vn97.runtime.*

fun main() {
    val utf8 = Utf8StreamAccumulator()
    check(utf8.append(byteArrayOf(0x41)).equals("A"))
    check(utf8.append(byteArrayOf(0xE2.toByte())).isEmpty())
    check(utf8.append(byteArrayOf(0x82.toByte())).isEmpty())
    check(utf8.append(byteArrayOf(0xAC.toByte())) == "€")
    check(utf8.finish().isEmpty())

    val greedy = NativeSamplerConfig(
        temperature = 0.0f,
        topK = 0,
        topP = 1.0f,
        seed = 7L,
    )
    check(
        NativeSampler.sample(
            floatArrayOf(0.0f, 2.0f, 2.0f, -1.0f),
            greedy,
            0L,
        ) == 1
    )

    val topOne = NativeSamplerConfig(
        temperature = 1.0f,
        topK = 1,
        topP = 1.0f,
        seed = 97L,
    )
    repeat(8) { step ->
        check(
            NativeSampler.sample(
                floatArrayOf(0.0f, 3.0f, 2.0f),
                topOne,
                step.toLong(),
            ) == 1
        )
    }

    check(NativeGenerationConfig().maxNewTokens == 256)
    println("M7G_GENERATION_STREAMING_PASS")
}
