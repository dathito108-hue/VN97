package ai.vn97.runtime

/**
 * Native structural/identity validation for a pre-activation VN97MI1 candidate.
 * This never returns a model handle and does not grant M9 trust or activation authority.
 */
object NativeModelImageCandidateValidator {
    fun validate(
        fd: Int,
        offset: Long,
        length: Long,
        artifactSha256: ByteArray,
    ): NativeActivatedModelInfo {
        require(fd >= 0) { "candidate fd must be non-negative" }
        require(offset >= 0L) { "candidate offset must be non-negative" }
        require(length > 0L) { "candidate length must be positive" }
        require(artifactSha256.size == 32) {
            "candidate SHA-256 must be exactly 32 bytes"
        }
        require(artifactSha256.any { it.toInt() != 0 }) {
            "candidate SHA-256 must be nonzero"
        }

        val out = LongArray(1)
        checkModelStatus(
            NativeRuntimeBindings.nativeModelOpen(
                fd,
                offset,
                length,
                artifactSha256.copyOf(),
                out,
            ),
            "VN97MI1 candidate open",
        )
        check(out[0] != 0L) {
            "native candidate validator returned a zero handle"
        }
        val handle = out[0]
        return try {
            readCandidateInfo(handle, artifactSha256)
        } finally {
            val status = NativeModelStatus.fromCode(
                NativeRuntimeBindings.nativeModelDestroy(handle)
            )
            if (status != NativeModelStatus.OK &&
                status != NativeModelStatus.INVALID_HANDLE
            ) {
                throw NativeModelException(
                    status,
                    "VN97MI1 candidate destroy",
                )
            }
        }
    }

    private fun readCandidateInfo(
        handle: Long,
        expectedModelId: ByteArray,
    ): NativeActivatedModelInfo {
        val ints = IntArray(9)
        val longs = LongArray(1)
        val modelId = ByteArray(32)
        checkModelStatus(
            NativeRuntimeBindings.nativeModelInfo(
                handle,
                ints,
                longs,
                modelId,
            ),
            "VN97MI1 candidate info",
        )
        require(modelId.contentEquals(expectedModelId)) {
            "native VN97MI1 candidate identity changed after open"
        }
        require(
            ints[0] > 1 &&
                ints[1] > 0 &&
                ints[2] > 0 &&
                ints[3] > 0
        ) {
            "native VN97MI1 candidate contains invalid geometry"
        }
        require(longs[0] > 0L) {
            "native VN97MI1 candidate image length must be positive"
        }
        return NativeActivatedModelInfo(
            modelId = modelId,
            vocabSize = ints[0],
            dModel = ints[1],
            layers = ints[2],
            dState = ints[3],
            embeddingKind = NativeEmbeddingKind.fromCode(ints[4]),
            embeddingRank = ints[5],
            hasTokenizer = ints[6] != 0,
            hasAudioProjection = ints[7] != 0,
            audioFrameSize = ints[8],
            imageBytes = longs[0],
        )
    }

}
