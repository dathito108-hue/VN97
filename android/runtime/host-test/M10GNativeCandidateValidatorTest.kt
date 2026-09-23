package ai.vn97.runtime

fun main() {
    val info = NativeModelImageCandidateValidator.validate(
        3,
        0,
        128,
        ByteArray(32) { 1 },
    )
    check(info.vocabSize == 10)
    check(info.modelId.contentEquals(ByteArray(32) { 1 }))
    println("M10G_NATIVE_CANDIDATE_VALIDATOR_SYNTAX_PASS")
}
