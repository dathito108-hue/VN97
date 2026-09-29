package ai.vn97.runtime

/** Conservative between-call guard, not a guarantee against an in-flight LMK. */
object VN97TrialMemory {
    const val RESERVE_BYTES = 256L * 1024 * 1024
    data class Sample(val availableBytes: Long, val thresholdBytes: Long,
                      val totalBytes: Long, val lowMemory: Boolean)
    class Pressure : IllegalStateException("Đã lưu kết quả và dừng vì RAM hệ thống xuống thấp. Đóng ứng dụng khác rồi thử lại.")
    fun underPressure(sample: Sample): Boolean = sample.lowMemory ||
        sample.availableBytes < sample.thresholdBytes + RESERVE_BYTES
}
