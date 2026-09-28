package ai.vn97.app

import ai.vn97.runtime.NativeActivatedModelInfo
import android.content.Context

/** Legacy CAP/MI1 images are not G06 deployments and cannot activate app inference. */
internal object VN97ModelImportRuntime {
    @Suppress("UNUSED_PARAMETER")
    fun requireCompatible(context: Context, info: NativeActivatedModelInfo): Unit = error(
        "Đã bỏ R2-SSM1 và fast/slow. Gói CAP/MI1 này không phải bộ G06. Cần bộ G06 gồm runtime, tokenizer, tuning, binding và chứng nhận kiểm tra tương thích; mô hình hiện tại chưa bị thay thế."
    )
}
