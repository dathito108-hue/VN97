package ai.vn97.app

import org.json.JSONObject

internal object VN97TrialRecovery {
    fun interrupted(report: JSONObject, currentPid: Int): Boolean {
        val pid = report.optInt("writer_pid", -1)
        return pid > 0 && pid != currentPid && !report.optBoolean("execution_passed", false) &&
            report.optString("status") in setOf("running", "verifying", "running_or_interrupted")
    }
    fun recover(report: JSONObject, currentPid: Int, exit: JSONObject?): JSONObject {
        if (!interrupted(report, currentPid)) return report
        report.put("status", "interrupted").put("execution_passed", false)
            .put("recovery_note", "Phiên trước đã dừng trước khi hoàn tất. Không tự chạy lại.")
        report.put("process_exit", exit ?: JSONObject().put("reason", "unavailable"))
        return report
    }
}
