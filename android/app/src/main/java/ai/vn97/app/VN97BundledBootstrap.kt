package ai.vn97.app

import ai.vn97.runtime.VN97G06Model
import java.io.File

enum class VN97BundledBootstrapOutcome { ABSENT, ALREADY_ACTIVE, ACTIVATED }

/** Trust root is the signed APK; local hashes bind its G06 deployment components. */
class VN97BundledBootstrap(
    private val application: VN97Application,
    @Suppress("UNUSED_PARAMETER") provisioner: VN97AppProvisioner,
) {
    fun activateIfPresent(required: Boolean = false): VN97BundledBootstrapOutcome {
        val changed = application.g06BundledRuntime.installIfPresent(required)
        val model = VN97G06Model.openOrNull(File(application.noBackupFilesDir, VN97G06Model.ROOT_DIR), application)
        if (model == null) {
            check(!required) { "Thiếu bộ G06 đã kiểm chứng trong bản cài đặt." }
            return VN97BundledBootstrapOutcome.ABSENT
        }
        model.close()
        return if (changed) VN97BundledBootstrapOutcome.ACTIVATED else VN97BundledBootstrapOutcome.ALREADY_ACTIVE
    }
}
