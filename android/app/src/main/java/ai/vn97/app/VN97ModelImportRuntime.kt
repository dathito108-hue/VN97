package ai.vn97.app

import ai.vn97.runtime.NativeActivatedModelInfo
import ai.vn97.runtime.VN97OrtProductionExecutor
import ai.vn97.runtime.VN97R2BridgeBinding
import android.content.Context
import java.io.File

/** Pre-activation check of the installed runtime; does not import or trust runtime bytes. */
internal object VN97ModelImportRuntime {
    fun requireCompatible(context: Context, info: NativeActivatedModelInfo) {
        val root = File(context.noBackupFilesDir, "vn97-r2")
        check(root.isDirectory) {
            "Thiếu bộ thực thi R2/ONNX. Gói lõi và chữ ký chưa đủ để chạy chat; cần bản phân phối VN97 có runtime tương thích. Mô hình hiện tại chưa bị thay thế."
        }
        val executor = VN97OrtProductionExecutor.open(
            context = context,
            runtimeRoot = File(root, "runtime"),
            tuningFile = File(root, "tuning.vn97r2e4.json"),
            maxCachedSessions = 1,
        )
        try {
            VN97R2BridgeBinding.load(File(root, "binding.vn97r2f2.json"))
                .requireCompatible(info, executor.runtimePackage)
        } finally {
            executor.close()
        }
    }
}
