package ai.vn97.app

import ai.vn97.runtime.VN97Mamba2K2SSafeDiagnostics
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import org.json.JSONArray
import org.json.JSONObject

class VN97K2SSafeController(
    private val activity: Activity,
    private val onStatus: (String) -> Unit,
    private val onAggregateReady: (String) -> Unit,
    private val onBusyChanged: (Boolean) -> Unit,
) : AutoCloseable {
    private val preferences =
        activity.getSharedPreferences(
            PREFS,
            Context.MODE_PRIVATE,
        )
    private var bound = false
    private var currentMode: String? = null
    private var resultReceived = false

    private val clientMessenger = Messenger(
        object : Handler(Looper.getMainLooper()) {
            override fun handleMessage(message: Message) {
                if (
                    message.what !=
                    VN97K2SSafeWorkerService.MSG_RESULT
                ) {
                    super.handleMessage(message)
                    return
                }
                val mode = message.data.getString(
                    VN97K2SSafeWorkerService.KEY_MODE
                ) ?: return
                val json = message.data.getString(
                    VN97K2SSafeWorkerService.KEY_JSON
                ) ?: return
                if (mode != currentMode) return

                resultReceived = true
                saveStep(mode, json)
                safeUnbind()
                currentMode = null
                onBusyChanged(false)
                publishProgress()
            }
        }
    )

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(
            name: ComponentName?,
            service: IBinder?,
        ) {
            val mode = currentMode ?: return
            if (service == null) {
                markWorkerDeath(mode, "null_binder")
                return
            }
            val request = Message.obtain(
                null,
                VN97K2SSafeWorkerService.MSG_RUN,
            ).apply {
                replyTo = clientMessenger
                data = android.os.Bundle().apply {
                    putString(
                        VN97K2SSafeWorkerService.KEY_MODE,
                        mode,
                    )
                }
            }
            runCatching {
                Messenger(service).send(request)
            }.onFailure {
                markWorkerDeath(mode, "send_failed")
            }
        }

        override fun onServiceDisconnected(
            name: ComponentName?,
        ) {
            val mode = currentMode ?: return
            if (!resultReceived) {
                markWorkerDeath(
                    mode,
                    "worker_process_disconnected",
                )
            }
        }

        override fun onBindingDied(
            name: ComponentName?,
        ) {
            val mode = currentMode ?: return
            if (!resultReceived) {
                markWorkerDeath(
                    mode,
                    "worker_binding_died",
                )
            }
        }

        override fun onNullBinding(
            name: ComponentName?,
        ) {
            val mode = currentMode ?: return
            markWorkerDeath(mode, "worker_null_binding")
        }
    }

    fun runNext() {
        check(currentMode == null) {
            "K2S worker is already running"
        }
        val next = nextMode()
        if (next == null) {
            publishProgress()
            return
        }

        currentMode = next
        resultReceived = false
        onBusyChanged(true)
        onStatus(
            "K2S isolated step: " + next + "\n" +
                "The heavy ORT session runs in a disposable worker process. " +
                "If Android kills it, the main UI should survive."
        )
        val intent = Intent(
            activity,
            VN97K2SSafeWorkerService::class.java,
        )
        bound = activity.bindService(
            intent,
            connection,
            Context.BIND_AUTO_CREATE,
        )
        if (!bound) {
            markWorkerDeath(next, "bind_failed")
        }
    }

    fun reset() {
        check(currentMode == null) {
            "cannot reset while K2S step is running"
        }
        preferences.edit().clear().apply()
        onAggregateReady("")
        publishProgress()
    }

    fun publishProgress() {
        val completed = completedModes()
        val total = VN97Mamba2K2SSafeDiagnostics.MODES.size
        val next = nextMode()
        val text = buildString {
            append("K2S crash-isolated diagnostics: ")
            append(completed.size)
            append("/")
            append(total)
            append(" complete.")
            if (next != null) {
                append("\nNext: ")
                append(next)
                append("\nTap RUN NEXT K2S STEP.")
            } else {
                append("\nAll K2S steps recorded.")
            }
        }
        onStatus(text)

        if (next == null) {
            onAggregateReady(buildAggregate())
        }
    }

    override fun close() {
        safeUnbind()
        currentMode = null
    }

    private fun markWorkerDeath(
        mode: String,
        reason: String,
    ) {
        if (mode != currentMode) return
        val step = JSONObject()
            .put("schema", "VN97M2K2SSTEP1")
            .put("mode", mode)
            .put("worker_process_died", true)
            .put("reason", reason)
            .put("device_measured", true)
            .put("synthetic", false)
            .put("same_weights_semantics", true)
            .put("graph_changed", false)
            .put("weights_changed", false)
            .put("production_activation_authorized", false)
        val wrapped = JSONObject()
            .put("worker_completed", false)
            .put("step", step)
            .toString()
        saveStep(mode, wrapped)
        resultReceived = true
        safeUnbind()
        currentMode = null
        onBusyChanged(false)
        publishProgress()
    }

    private fun saveStep(
        mode: String,
        json: String,
    ) {
        val parsed = JSONObject(json)
        require(parsed.getJSONObject("step")
            .getString("mode") == mode)
        preferences.edit()
            .putString(STEP_PREFIX + mode, parsed.toString())
            .apply()
    }

    private fun completedModes(): List<String> =
        VN97Mamba2K2SSafeDiagnostics.MODES.filter {
            preferences.contains(STEP_PREFIX + it)
        }

    private fun nextMode(): String? =
        VN97Mamba2K2SSafeDiagnostics.MODES.firstOrNull {
            !preferences.contains(STEP_PREFIX + it)
        }

    private fun buildAggregate(): String {
        val steps = JSONArray()
        VN97Mamba2K2SSafeDiagnostics.MODES.forEach { mode ->
            val raw = requireNotNull(
                preferences.getString(
                    STEP_PREFIX + mode,
                    null,
                )
            )
            steps.put(
                JSONObject(raw).getJSONObject("step")
            )
        }
        return JSONObject()
            .put("schema", "VN97M2K2SSAFE1")
            .put("steps", steps)
            .put("device_measured", true)
            .put("synthetic", false)
            .put("same_weights_semantics", true)
            .put("graph_changed", false)
            .put("weights_changed", false)
            .put("production_activation_authorized", false)
            .toString()
    }

    private fun safeUnbind() {
        if (!bound) return
        runCatching {
            activity.unbindService(connection)
        }
        bound = false
    }

    companion object {
        private const val PREFS = "vn97_k2s_progress"
        private const val STEP_PREFIX = "step_"
    }
}
