package ai.vn97.app

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import org.json.JSONObject

data class VN97R2ContinuationIdentity(
    val bindingId: String,
    val runtimeId: String,
    val checkpointSha256: String,
    val tokenizerModelSha256: String,
)

class VN97R2ContinuityGuard(
    private val application: VN97Application,
) {
    private val root =
        File(application.noBackupFilesDir, "vn97-r2/continuity")

    @Synchronized
    fun bind(jobId: Int): VN97R2ContinuationIdentity {
        require(jobId > 0)
        ensureRoot()
        val identity = currentIdentity()
        val body = JSONObject()
            .put("schema", SCHEMA)
            .put("job_id", jobId)
            .put("binding_id", identity.bindingId)
            .put("runtime_id", identity.runtimeId)
            .put("checkpoint_sha256", identity.checkpointSha256)
            .put("tokenizer_model_sha256", identity.tokenizerModelSha256)
            .toString()
            .toByteArray(StandardCharsets.UTF_8)
        val target = file(jobId)
        val temp = File(root, target.name + ".tmp")
        if (temp.exists()) check(temp.delete()) {
            "failed to clear stale R2 continuity temp file"
        }
        temp.writeBytes(body)
        check(temp.renameTo(target)) {
            temp.delete()
            "failed to atomically bind R2 continuation identity"
        }
        return identity
    }

    @Synchronized
    fun requireCurrent(jobId: Int): VN97R2ContinuationIdentity {
        require(jobId > 0)
        val target = file(jobId)
        require(
            target.isFile &&
                !Files.isSymbolicLink(target.toPath())
        ) {
            "R2 continuation identity is missing or unsafe"
        }
        val json = JSONObject(target.readText(StandardCharsets.UTF_8))
        require(
            json.keys().asSequence().toSet() == setOf(
                "schema",
                "job_id",
                "binding_id",
                "runtime_id",
                "checkpoint_sha256",
                "tokenizer_model_sha256",
            )
        ) {
            "R2 continuation identity fields mismatch"
        }
        require(json.getString("schema") == SCHEMA)
        require(json.getInt("job_id") == jobId) {
            "R2 continuation job identity mismatch"
        }
        val persisted = VN97R2ContinuationIdentity(
            bindingId = json.getString("binding_id"),
            runtimeId = json.getString("runtime_id"),
            checkpointSha256 = json.getString("checkpoint_sha256"),
            tokenizerModelSha256 =
                json.getString("tokenizer_model_sha256"),
        )
        val current = currentIdentity()
        require(persisted == current) {
            "R2 runtime/checkpoint/tokenizer changed since continuation was persisted"
        }
        return persisted
    }

    @Synchronized
    fun delete(jobId: Int) {
        require(jobId > 0)
        val target = file(jobId)
        if (target.exists()) check(target.delete()) {
            "failed to delete R2 continuation identity"
        }
    }

    private fun currentIdentity(): VN97R2ContinuationIdentity {
        val binding = application.r2Runtime.currentBinding()
        return VN97R2ContinuationIdentity(
            bindingId = binding.bindingId,
            runtimeId = binding.runtimeId,
            checkpointSha256 = binding.checkpointSha256,
            tokenizerModelSha256 = binding.tokenizerModelSha256,
        )
    }

    private fun ensureRoot() {
        if (root.exists()) {
            require(root.isDirectory && !Files.isSymbolicLink(root.toPath())) {
                "R2 continuity root is unsafe"
            }
        } else {
            check(root.mkdirs()) {
                "failed to create R2 continuity root"
            }
        }
    }

    private fun file(jobId: Int): File =
        File(root, "job-" + jobId + ".vn97r2c1")

    companion object {
        private const val SCHEMA = "VN97R2F4CONT1"
    }
}
