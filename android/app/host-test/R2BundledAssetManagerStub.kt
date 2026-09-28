package android.content.res

import java.io.File
import java.io.InputStream
import java.io.IOException

class AssetManager(private val root: File) {
    var failOpenPath: String? = null

    fun list(path: String): Array<String>? =
        File(root, path).list() ?: emptyArray()

    fun open(path: String): InputStream {
        if (path == failOpenPath) throw IOException("injected asset read failure")
        return File(root, path).inputStream()
    }
}
