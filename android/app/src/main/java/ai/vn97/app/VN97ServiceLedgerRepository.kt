package ai.vn97.app

import java.io.File
import java.nio.file.Files
import java.nio.file.LinkOption

/** A null read means confirmed absence, never a parse or I/O failure. */
internal class VN97ServiceLedgerRepository(
    private val read: () -> String?,
    private val write: (String) -> Unit,
) {
    fun load(): VN97ServiceRevenueLedger = synchronized(storageLock) { loadLocked() }

    fun update(change: (VN97ServiceRevenueLedger) -> Unit) = synchronized(storageLock) {
        val ledger = loadLocked() // Do not call change or write unless old data was read successfully.
        change(ledger)
        val encoded = VN97ServiceLedgerCodec.encode(ledger)
        write(encoded)
    }

    private fun loadLocked(): VN97ServiceRevenueLedger = try {
        val encoded = read()
        if (encoded == null) VN97ServiceRevenueLedger() else VN97ServiceLedgerCodec.decode(encoded)
    } catch (failure: Exception) {
        throw IllegalStateException("Không đọc được sổ việc; dữ liệu cũ được giữ nguyên.", failure)
    }

    companion object {
        // All Activity instances in this process serialize the entire transaction.
        private val storageLock = Any()

        fun definitelyMissing(base: File): Boolean = listOf(
            base, File(base.path + ".bak"), File(base.path + ".new"),
        ).all { Files.notExists(it.toPath(), LinkOption.NOFOLLOW_LINKS) }
    }
}
