package android.os

import java.io.Closeable
import java.io.File

class ParcelFileDescriptor(
    val fd: Int,
    val statSize: Long,
) : Closeable {
    override fun close() = Unit

    companion object {
        const val MODE_READ_ONLY = 0

        fun open(file: File, mode: Int): ParcelFileDescriptor {
            @Suppress("UNUSED_VARIABLE") val ignored = mode
            return ParcelFileDescriptor(3, file.length())
        }
    }
}
