package ai.vn97.runtime

/** Bounded greedy continuation. No tools, chat template, or activation authority. */
object VN97Int8TextLoop {
    const val MAX_PROMPT_TOKENS = 128
    const val MAX_NEW_TOKENS = 16
    fun generate(
        prompt: IntArray,
        eos: Int,
        cancelled: () -> Boolean,
        infer: (IntArray) -> FloatArray,
        onToken: (IntArray) -> Unit,
        prefillChunk: Int = 8,
    ): IntArray {
        require(prefillChunk == 1 || prefillChunk == 8)
        require(prompt.size in 1..MAX_PROMPT_TOKENS) { "Câu nhập phải có 1–128 token." }
        require(prompt.all { it in 0 until 50277 } && eos in 0 until 50277)
        val output = mutableListOf<Int>()
        fun checkCancelled() { if (cancelled()) throw java.util.concurrent.CancellationException("Đã dừng thử văn bản") }
        fun choose(row: FloatArray): Int {
            require(row.size == 50288 && row.all { it.isFinite() }) { "Logits sai kích thước hoặc có NaN/Inf" }
            return (0 until 50277).maxByOrNull { row[it] }!!
        }
        var next = -1
        for (offset in prompt.indices step prefillChunk) {
            checkCancelled()
            next = choose(infer(prompt.copyOfRange(offset, minOf(offset + prefillChunk, prompt.size))))
        }
        repeat(MAX_NEW_TOKENS) {
            checkCancelled()
            if (next == eos) return output.toIntArray()
            output.add(next)
            onToken(output.toIntArray())
            if (output.size < MAX_NEW_TOKENS) {
                checkCancelled()
                next = choose(infer(intArrayOf(next)))
            }
        }
        return output.toIntArray()
    }
}
