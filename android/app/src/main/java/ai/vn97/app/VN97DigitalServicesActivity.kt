package ai.vn97.app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.text.InputFilter
import android.text.InputType
import android.util.AtomicFile
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONArray
import org.json.JSONObject

/** Explicit local workbench: no marketplace posting, customer messaging or payment claims. */
class VN97DigitalServicesActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var serviceView: Spinner
    private lateinit var titleView: EditText
    private lateinit var inputView: EditText
    private lateinit var priceView: EditText
    private lateinit var costView: EditText
    private lateinit var statusView: TextView
    private lateinit var previewView: TextView
    private lateinit var ledgerView: TextView
    private lateinit var paymentProviderView: EditText
    private lateinit var paymentReferenceView: EditText
    private lateinit var paymentGrossView: EditText
    private lateinit var paymentFeeView: EditText
    private lateinit var paymentRefundView: EditText
    private lateinit var paymentCostView: EditText
    private lateinit var produceButton: Button
    private lateinit var exportButton: Button
    private var latest: VN97DigitalDelivery? = null
    private var pendingExport: VN97DigitalDelivery? = null
    private var currentOrderId: String? = null
    private var pendingExportOrderId: String? = null
    private val store: AtomicFile
        get() = AtomicFile(File(noBackupFilesDir, "vn97-digital-service-latest.json"))
    private val ledgerStore: AtomicFile
        get() = AtomicFile(File(noBackupFilesDir, "vn97-service-revenue-ledger.json"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = (16 * resources.displayMetrics.density).toInt()
            setPadding(pad, pad, pad, pad)
        }
        fun label(value: String) {
            root.addView(TextView(this).apply { text = value; textSize = 17f })
        }
        fun field(hintText: String, maxChars: Int, numeric: Boolean = false): EditText {
            return EditText(this).apply {
                hint = hintText
                filters = arrayOf(InputFilter.LengthFilter(maxChars))
                if (numeric) inputType = InputType.TYPE_CLASS_NUMBER
                root.addView(this, ViewGroup.LayoutParams(-1, -2))
            }
        }
        label("VN97 · Dịch vụ số")
        label("Tạo sản phẩm tại máy. Việc gần nhất được lưu riêng trong ứng dụng. " +
            "Các thao tác này chưa tìm khách hàng hoặc thu tiền.")
        serviceView = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@VN97DigitalServicesActivity,
                android.R.layout.simple_spinner_dropdown_item,
                VN97DigitalService.values().map { it.title },
            )
        }
        root.addView(serviceView)
        titleView = field("Tên việc / mã đơn nội bộ", 120)
        label("Giá và chi phí dự kiến, đơn vị VND")
        priceView = field("Giá báo dự kiến", 13, true).apply { setText("0") }
        costView = field("Chi phí dự kiến", 13, true).apply { setText("0") }
        label("Dán văn bản hoặc CSV có hàng tiêu đề. Tối đa 128 KiB, 64 cột, 5.000 dòng.")
        inputView = field("Dữ liệu cần xử lý", VN97DigitalServices.MAX_INPUT_BYTES).apply {
            minLines = 5
            maxLines = 10
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            gravity = android.view.Gravity.TOP
        }
        produceButton = Button(this).apply {
            text = "Tạo sản phẩm và lưu việc"
            setOnClickListener { produce() }
        }
        root.addView(produceButton)
        statusView = TextView(this).apply { text = "Chưa có sản phẩm. Chưa có thanh toán xác minh." }
        root.addView(statusView)
        previewView = TextView(this).apply { setTextIsSelectable(true) }
        root.addView(previewView)
        exportButton = Button(this).apply {
            text = "Xuất sản phẩm đang xem (ZIP)"
            isEnabled = false
            setOnClickListener {
                pendingExport = latest
                pendingExportOrderId = currentOrderId
                startActivityForResult(
                    Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "application/zip"
                        putExtra(Intent.EXTRA_TITLE, "VN97-delivery.zip")
                    },
                    EXPORT_REQUEST,
                )
            }
        }
        root.addView(exportButton)
        ledgerView = TextView(this).apply { setTextIsSelectable(true) }
        root.addView(ledgerView)
        label("Đối soát khoản nhận tự khai (không được tính là doanh thu xác minh)")
        paymentProviderView = field("Nhà cung cấp/kênh, ASCII (vd: bank-manual)", 80)
        paymentReferenceView = field("Mã giao dịch duy nhất", 160)
        paymentGrossView = field("Số tiền nhận", 13, true).apply { setText("0") }
        paymentFeeView = field("Phí", 13, true).apply { setText("0") }
        paymentRefundView = field("Hoàn tiền", 13, true).apply { setText("0") }
        paymentCostView = field("Chi phí thực tế", 13, true).apply { setText("0") }
        root.addView(Button(this).apply {
            text = "Lưu khoản tự khai cho việc hiện tại"
            setOnClickListener { recordManualPayment() }
        })
        root.addView(Button(this).apply {
            text = "Xóa việc đã lưu"
            setOnClickListener {
                worker.execute {
                    store.delete()
                    runOnUiThread {
                        if (!isDestroyed) {
                            latest = null
                            currentOrderId = null
                            pendingExport = null
                            pendingExportOrderId = null
                            inputView.text.clear()
                            titleView.text.clear()
                            previewView.text = ""
                            exportButton.isEnabled = false
                            statusView.text = "Đã xóa việc tại máy. Tệp đã xuất không bị xóa."
                        }
                    }
                }
            }
        })
        setContentView(ScrollView(this).apply { addView(root) })
        restoreLatest()
        refreshLedger()
    }

    private fun produce() {
        val source = inputView.text.toString()
        val title = titleView.text.toString().trim()
        val service = VN97DigitalService.values()[serviceView.selectedItemPosition]
        val orderId = "svc-" + UUID.randomUUID().toString()
        val quote = runCatching {
            require(title.isNotBlank()) { "Hãy đặt tên việc." }
            VN97ServiceQuote(priceView.text.toString().toLong(), costView.text.toString().toLong())
        }.getOrElse {
            statusView.text = "Tên việc và số tiền phải hợp lệ (0–1.000.000.000.000 VND)."
            return
        }
        produceButton.isEnabled = false
        exportButton.isEnabled = false
        worker.execute {
            val result = runCatching {
                val delivery = VN97DigitalServices.produce(service, source)
                val order = VN97ServiceOrder(
                    orderId = orderId,
                    title = title,
                    service = service,
                    quotedPriceVnd = quote.priceVnd,
                    estimatedCostVnd = quote.estimatedCostVnd,
                    outputSha256 = delivery.outputSha256,
                    createdAtEpochMs = System.currentTimeMillis(),
                )
                val ledger = readLedger()
                ledger.recordOrder(order)
                writeLedger(ledger)
                val record = JSONObject().put("schema", 2).put("orderId", orderId).put("title", title)
                    .put("service", service.name).put("source", source)
                    .put("priceVnd", quote.priceVnd).put("costVnd", quote.estimatedCostVnd)
                val bytes = record.toString().toByteArray(Charsets.UTF_8)
                val output = store.startWrite()
                try {
                    output.write(bytes)
                    store.finishWrite(output)
                } catch (failure: Exception) {
                    store.failWrite(output)
                    throw failure
                }
                delivery
            }
            runOnUiThread {
                if (!isDestroyed) {
                    produceButton.isEnabled = true
                    result.fold(
                        onSuccess = {
                            currentOrderId = orderId
                            showDelivery(it, title, quote)
                            refreshLedger()
                        },
                        onFailure = {
                            latest = null
                            previewView.text = ""
                            statusView.text = "Chưa tạo được sản phẩm: ${it.message ?: "lỗi xử lý"}"
                        },
                    )
                }
            }
        }
    }

    private fun showDelivery(delivery: VN97DigitalDelivery, title: String, quote: VN97ServiceQuote) {
        latest = delivery
        exportButton.isEnabled = true
        statusView.text = "$title\n${delivery.report}\n" +
            "Giá dự kiến: ${quote.priceVnd} VND; chi phí dự kiến: ${quote.estimatedCostVnd} VND.\n" +
            "Chênh lệch dự kiến: ${quote.estimatedMarginVnd} VND.\n" +
            "Chưa có thanh toán xác minh. Hãy kiểm tra sản phẩm trước khi bàn giao."
        previewView.text = delivery.content.take(8_000) +
            if (delivery.content.length > 8_000) "\n… Bản xem trước rút gọn; tệp xuất chứa đầy đủ." else ""
    }

    private fun restoreLatest() {
        produceButton.isEnabled = false
        worker.execute {
            val record = runCatching {
                store.openRead().use { stream ->
                    val buffer = ByteArray(1024 * 1024 + 1)
                    var count = 0
                    while (count < buffer.size) {
                        val read = stream.read(buffer, count, buffer.size - count)
                        if (read < 0) break
                        count += read
                    }
                    require(count <= 1024 * 1024)
                    JSONObject(String(buffer, 0, count, Charsets.UTF_8)).also {
                        require(it.getInt("schema") in 1..2)
                    }
                }
            }.getOrNull()
            val restored = runCatching {
                if (record == null) null else VN97DigitalServices.produce(
                    VN97DigitalService.valueOf(record.getString("service")), record.getString("source"),
                )
            }.getOrNull()
            runOnUiThread {
                if (!isDestroyed) {
                    produceButton.isEnabled = true
                    if (record != null && restored != null) {
                        runCatching {
                            val quote = VN97ServiceQuote(record.getLong("priceVnd"), record.getLong("costVnd"))
                            currentOrderId = if (record.getInt("schema") >= 2) {
                                record.getString("orderId")
                            } else null
                            serviceView.setSelection(VN97DigitalService.valueOf(record.getString("service")).ordinal)
                            titleView.setText(record.getString("title"))
                            inputView.setText(record.getString("source"))
                            priceView.setText(quote.priceVnd.toString())
                            costView.setText(quote.estimatedCostVnd.toString())
                            showDelivery(restored, record.getString("title"), quote)
                        }
                    }
                }
            }
        }
    }

    @Deprecated("Legacy activity result API for the existing Activity-based app")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != EXPORT_REQUEST) return
        val delivery = pendingExport
        val orderId = pendingExportOrderId
        pendingExport = null
        pendingExportOrderId = null
        val uri = data?.data
        if (resultCode != RESULT_OK || uri == null || delivery == null) return
        worker.execute {
            val result = runCatching {
                val stream = checkNotNull(contentResolver.openOutputStream(uri, "wt"))
                ZipOutputStream(stream).use { zip ->
                    zip.putNextEntry(ZipEntry(delivery.fileName))
                    zip.write(delivery.content.toByteArray(Charsets.UTF_8))
                    zip.closeEntry()
                    val report = JSONObject().put("schema", "VN97_DIGITAL_DELIVERY_1")
                        .put("orderId", orderId ?: JSONObject.NULL)
                        .put("sourceSha256", delivery.sourceSha256)
                        .put("outputSha256", delivery.outputSha256)
                        .put("fileName", delivery.fileName).put("transformations", delivery.report)
                    zip.putNextEntry(ZipEntry("delivery.json"))
                    zip.write(report.toString(2).toByteArray(Charsets.UTF_8))
                    zip.closeEntry()
                }
            }
            runOnUiThread {
                if (!isDestroyed) {
                    if (result.isSuccess && orderId != null) {
                        worker.execute {
                            runCatching {
                                val ledger = readLedger()
                                ledger.markExported(orderId, System.currentTimeMillis())
                                writeLedger(ledger)
                            }
                            runOnUiThread { if (!isDestroyed) refreshLedger() }
                        }
                    }
                    statusView.text = if (result.isSuccess) {
                        "Đã xuất ZIP và ghi nhận bàn giao tại máy. Chưa có thanh toán xác minh."
                    } else "Xuất tệp thất bại; tệp đích có thể chưa đầy đủ. Hãy xuất lại."
                }
            }
        }
    }

    private fun recordManualPayment() {
        val orderId = currentOrderId
        if (orderId == null) {
            statusView.text = "Hãy tạo hoặc khôi phục một việc có mã đơn trước."
            return
        }
        val payment = runCatching {
            VN97PaymentRecord(
                orderId = orderId,
                provider = paymentProviderView.text.toString().trim(),
                providerAccountFingerprint = "",
                externalEventId = paymentReferenceView.text.toString().trim(),
                grossVnd = paymentGrossView.text.toString().toLong(),
                providerFeeVnd = paymentFeeView.text.toString().toLong(),
                refundVnd = paymentRefundView.text.toString().toLong(),
                realizedCostVnd = paymentCostView.text.toString().toLong(),
                verification = VN97PaymentVerification.MANUAL_UNVERIFIED,
                evidenceSha256 = null,
                settledAtEpochMs = System.currentTimeMillis(),
            )
        }.getOrElse {
            statusView.text = "Khoản tự khai không hợp lệ: ${it.message ?: "kiểm tra mã và số tiền"}."
            return
        }
        worker.execute {
            val result = runCatching {
                val ledger = readLedger()
                ledger.recordPayment(payment)
                writeLedger(ledger)
            }
            runOnUiThread {
                if (!isDestroyed) {
                    statusView.text = if (result.isSuccess) {
                        "Đã lưu khoản tự khai. Khoản này KHÔNG được tính vào doanh thu xác minh."
                    } else {
                        "Không lưu khoản tự khai: ${result.exceptionOrNull()?.message ?: "lỗi lưu trữ"}."
                    }
                    refreshLedger()
                }
            }
        }
    }

    private fun refreshLedger() {
        worker.execute {
            val ledger = runCatching { readLedger() }.getOrElse { VN97ServiceRevenueLedger() }
            val summary = ledger.summary()
            val recent = ledger.orders().takeLast(8).reversed().joinToString("\n") {
                "• ${it.title} · ${it.quotedPriceVnd} VND · " +
                    if (it.exportedAtEpochMs == null) "chưa xuất" else "đã xuất"
            }
            val text = "Sổ việc: ${summary.orderCount} đơn; ${summary.exportedOrderCount} đã xuất.\n" +
                "Doanh thu ròng ĐÃ XÁC MINH: ${summary.verifiedNetVnd} VND " +
                "(${summary.verifiedSettlementCount} settlement).\n" +
                "Khoản tự khai CHƯA XÁC MINH: ${summary.unverifiedClaimNetVnd} VND " +
                "(${summary.unverifiedClaimCount} khoản)." +
                if (recent.isEmpty()) "" else "\nGần đây:\n$recent"
            runOnUiThread { if (!isDestroyed) ledgerView.text = text }
        }
    }

    private fun readLedger(): VN97ServiceRevenueLedger {
        val root = runCatching {
            ledgerStore.openRead().use { JSONObject(readBounded(it, 2 * 1024 * 1024)) }
        }.getOrNull() ?: return VN97ServiceRevenueLedger()
        require(root.getInt("schema") == 1)
        val orders = root.getJSONArray("orders")
        val payments = root.getJSONArray("payments")
        require(orders.length() <= VN97ServiceRevenueLedger.MAX_ORDERS)
        require(payments.length() <= VN97ServiceRevenueLedger.MAX_PAYMENTS)
        return VN97ServiceRevenueLedger(
            orders = (0 until orders.length()).map { index ->
                val value = orders.getJSONObject(index)
                VN97ServiceOrder(
                    orderId = value.getString("orderId"),
                    title = value.getString("title"),
                    service = VN97DigitalService.valueOf(value.getString("service")),
                    quotedPriceVnd = value.getLong("quotedPriceVnd"),
                    estimatedCostVnd = value.getLong("estimatedCostVnd"),
                    outputSha256 = value.getString("outputSha256"),
                    createdAtEpochMs = value.getLong("createdAtEpochMs"),
                    exportedAtEpochMs = if (value.isNull("exportedAtEpochMs")) null else {
                        value.getLong("exportedAtEpochMs")
                    },
                )
            },
            payments = (0 until payments.length()).map { index ->
                val value = payments.getJSONObject(index)
                VN97PaymentRecord(
                    orderId = value.getString("orderId"),
                    provider = value.getString("provider"),
                    providerAccountFingerprint = value.getString("providerAccountFingerprint"),
                    externalEventId = value.getString("externalEventId"),
                    grossVnd = value.getLong("grossVnd"),
                    providerFeeVnd = value.getLong("providerFeeVnd"),
                    refundVnd = value.getLong("refundVnd"),
                    realizedCostVnd = value.getLong("realizedCostVnd"),
                    verification = VN97PaymentVerification.valueOf(value.getString("verification")),
                    evidenceSha256 = if (value.isNull("evidenceSha256")) null else {
                        value.getString("evidenceSha256")
                    },
                    settledAtEpochMs = value.getLong("settledAtEpochMs"),
                )
            },
        )
    }

    private fun writeLedger(ledger: VN97ServiceRevenueLedger) {
        val root = JSONObject().put("schema", 1)
            .put("orders", JSONArray().also { array ->
                ledger.orders().forEach { order ->
                    array.put(JSONObject().put("orderId", order.orderId).put("title", order.title)
                        .put("service", order.service.name).put("quotedPriceVnd", order.quotedPriceVnd)
                        .put("estimatedCostVnd", order.estimatedCostVnd)
                        .put("outputSha256", order.outputSha256)
                        .put("createdAtEpochMs", order.createdAtEpochMs)
                        .put("exportedAtEpochMs", order.exportedAtEpochMs ?: JSONObject.NULL))
                }
            }).put("payments", JSONArray().also { array ->
                ledger.payments().forEach { payment ->
                    array.put(JSONObject().put("orderId", payment.orderId)
                        .put("provider", payment.provider)
                        .put("providerAccountFingerprint", payment.providerAccountFingerprint)
                        .put("externalEventId", payment.externalEventId).put("grossVnd", payment.grossVnd)
                        .put("providerFeeVnd", payment.providerFeeVnd).put("refundVnd", payment.refundVnd)
                        .put("realizedCostVnd", payment.realizedCostVnd)
                        .put("verification", payment.verification.name)
                        .put("evidenceSha256", payment.evidenceSha256 ?: JSONObject.NULL)
                        .put("settledAtEpochMs", payment.settledAtEpochMs))
                }
            })
        val bytes = root.toString().toByteArray(Charsets.UTF_8)
        require(bytes.size <= 2 * 1024 * 1024) { "service ledger exceeds storage bound" }
        val output = ledgerStore.startWrite()
        try {
            output.write(bytes)
            ledgerStore.finishWrite(output)
        } catch (failure: Exception) {
            ledgerStore.failWrite(output)
            throw failure
        }
    }

    private fun readBounded(input: java.io.InputStream, maxBytes: Int): String {
        val output = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(8 * 1024)
        input.use { stream ->
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) break
                require(output.size() + read <= maxBytes) { "stored data exceeds byte bound" }
                output.write(buffer, 0, read)
            }
        }
        return output.toString(Charsets.UTF_8.name())
    }

    override fun onDestroy() {
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val EXPORT_REQUEST = 971
    }
}
