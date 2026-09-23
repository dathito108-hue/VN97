package android.content

import java.io.File

open class Context(
    val noBackupFilesDir: File,
) {
    val applicationContext: Context get() = this
}
