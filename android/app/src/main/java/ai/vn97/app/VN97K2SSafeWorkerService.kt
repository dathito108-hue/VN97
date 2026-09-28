package ai.vn97.app

import ai.vn97.runtime.VN97Mamba2K2SSafeDiagnostics
import android.app.Service
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

class VN97K2SSafeWorkerService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private val incoming = Messenger(
        object : Handler(Looper.getMainLooper()) {
            override fun handleMessage(message: Message) {
                when (message.what) {
                    MSG_RUN -> {
                        val mode = message.data.getString(KEY_MODE)
                        val reply = message.replyTo
                        if (mode.isNullOrBlank() || reply == null) {
                            return
                        }
                        executor.execute {
                            runOne(mode, reply)
                        }
                    }
                    else -> super.handleMessage(message)
                }
            }
        }
    )

    override fun onBind(intent: Intent?): IBinder =
        incoming.binder

    override fun onDestroy() {
        executor.shutdownNow()
        super.onDestroy()
    }

    private fun runOne(
        mode: String,
        reply: Messenger,
    ) {
        val payload = runCatching {
            val runtimeRoot = File(
                noBackupFilesDir,
                RUNTIME_DIR,
            )
            val result = VN97Mamba2K2SSafeDiagnostics().runMode(
                context = applicationContext,
                runtimeRoot = runtimeRoot,
                mode = mode,
            )
            JSONObject()
                .put("worker_completed", true)
                .put("step", result.toJson())
                .toString()
        }.getOrElse { error ->
            JSONObject()
                .put("worker_completed", true)
                .put(
                    "step",
                    JSONObject()
                        .put("schema", "VN97M2K2SSTEP1")
                        .put("mode", mode)
                        .put("worker_internal_failure", true)
                        .put(
                            "error_class",
                            error::class.java.name.take(160),
                        )
                        .put(
                            "error_message",
                            error.message
                                .orEmpty()
                                .replace(
                                    Regex("[\\r\\n\\t]+"),
                                    " ",
                                )
                                .replace(Regex("\\s+"), " ")
                                .trim()
                                .take(480)
                                .ifBlank { "no_message" },
                        )
                )
                .toString()
        }

        val result = Message.obtain(
            null,
            MSG_RESULT,
        ).apply {
            data = Bundle().apply {
                putString(KEY_MODE, mode)
                putString(KEY_JSON, payload)
            }
        }
        runCatching {
            reply.send(result)
        }

        Handler(Looper.getMainLooper()).postDelayed(
            {
                stopSelf()
                Process.killProcess(Process.myPid())
            },
            WORKER_EXIT_DELAY_MS,
        )
    }

    companion object {
        const val MSG_RUN = 1
        const val MSG_RESULT = 2
        const val KEY_MODE = "vn97_k2s_mode"
        const val KEY_JSON = "vn97_k2s_json"

        private const val RUNTIME_DIR = "vn97-r2-k2q-g06"
        private const val WORKER_EXIT_DELAY_MS = 350L
    }
}
