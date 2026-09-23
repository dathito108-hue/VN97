package ai.vn97.runtime

class NativeCognitionContractException(
    message: String,
    cause: Throwable? = null,
) : IllegalArgumentException(message, cause)
