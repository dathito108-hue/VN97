package ai.vn97.app

/** The initial result already consumed one advance; never restart the original turn. */
internal fun <T> advanceVN97BoundedTurn(
    initial: T,
    maxAdvances: Int,
    isYielded: (T) -> Boolean,
    advance: (T) -> T,
): T {
    require(maxAdvances > 0)
    var current = initial
    var count = 1
    while (isYielded(current) && count < maxAdvances) {
        current = advance(current)
        count++
    }
    return current
}
