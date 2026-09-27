package ai.vn97.app

import ai.vn97.runtime.NativeActivatedModel
import ai.vn97.runtime.VN97R2CognitionInference
import ai.vn97.runtime.VN97R2GenerationConfig
import ai.vn97.runtime.VN97R2ReasoningMode
import java.io.File

class VN97R2RuntimeManager(
    private val application: VN97Application,
) {
    fun openCognition(
        model: NativeActivatedModel,
        mode: VN97R2ReasoningMode = VN97R2ReasoningMode.DEEP,
    ): VN97R2CognitionInference {
        val root = runtimeRoot()
        require(root.isDirectory) {
            "VN97 R2 runtime assets are not installed"
        }
        val tuning = File(root, TUNING_FILENAME)
        require(tuning.isFile) {
            "VN97 R2 device tuning profile is missing"
        }
        return VN97R2CognitionInference.open(
            context = application,
            model = model,
            runtimeRoot = root,
            tuningFile = tuning,
            generationConfig = VN97R2GenerationConfig(mode = mode),
        )
    }

    fun runtimeRoot(): File =
        File(application.noBackupFilesDir, RUNTIME_RELATIVE_ROOT)

    fun supportsAudioModelPath(): Boolean = false

    fun supportsVisionModelPath(): Boolean = false

    companion object {
        const val RUNTIME_RELATIVE_ROOT = "vn97-r2/current"
        const val TUNING_FILENAME = "autotune.vn97r2e4.json"
    }
}
