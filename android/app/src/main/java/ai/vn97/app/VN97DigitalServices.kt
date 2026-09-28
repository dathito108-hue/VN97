package ai.vn97.app

import ai.vn97.platform.VN97DigitalServiceCore
import ai.vn97.platform.VN97LocalDigitalService

enum class VN97DigitalService(val title: String) {
    TABLE_CLEANUP("Làm sạch bảng CSV"),
    DOCUMENT_FORMAT("Định dạng văn bản"),
    CATALOG_PAGE("Tạo trang danh mục từ CSV"),
}

data class VN97DigitalDelivery(
    val fileName: String,
    val content: String,
    val report: String,
    val sourceSha256: String,
    val outputSha256: String,
)

data class VN97ServiceQuote(val priceVnd: Long, val estimatedCostVnd: Long) {
    init {
        require(priceVnd in 0..1_000_000_000_000L)
        require(estimatedCostVnd in 0..1_000_000_000_000L)
    }

    val estimatedMarginVnd: Long get() = priceVnd - estimatedCostVnd
}

/** Local production utilities. No model, network, payment or trading side effects. */
object VN97DigitalServices {
    const val MAX_INPUT_BYTES = VN97DigitalServiceCore.MAX_INPUT_BYTES
    const val MAX_OUTPUT_BYTES = VN97DigitalServiceCore.MAX_OUTPUT_BYTES

    fun produce(service: VN97DigitalService, source: String): VN97DigitalDelivery {
        val delivery = VN97DigitalServiceCore.produce(
            VN97LocalDigitalService.valueOf(service.name),
            source,
        )
        return VN97DigitalDelivery(
            delivery.fileName,
            delivery.content,
            delivery.report,
            delivery.sourceSha256,
            delivery.outputSha256,
        )
    }
}
