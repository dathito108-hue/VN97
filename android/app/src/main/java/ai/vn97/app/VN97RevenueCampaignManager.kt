package ai.vn97.app

import android.content.Context
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import org.json.JSONArray
import org.json.JSONObject

data class VN97RevenueCampaignRequest(
    val endpoint: String,
    val sourceId: String,
    val symbols: Set<String>,
    val userGoal: String,
    val requiresBatteryNotLow: Boolean = true,
    val requiresCharging: Boolean = false,
    val policy: VN97RevenueCampaignPolicy =
        VN97RevenueCampaignPolicy(),
)

data class VN97RevenueCampaignSnapshot(
    val campaignId: String,
    val state: VN97RevenueCampaignState,
    val sessionsStarted: Int,
    val sessionsCompleted: Int,
    val currentSessionId: String?,
    val currentJobId: Int?,
    val qualification: VN97RevenueQualification?,
) {
    val productionMoneyMovementAuthorized: Boolean
        get() = false
}

class VN97RevenueCampaignManager(
    private val application: VN97Application,
) {
    private val preferences =
        application.getSharedPreferences(
            PREFS_NAME,
            Context.MODE_PRIVATE,
        )

    @Synchronized
    fun start(
        request: VN97RevenueCampaignRequest,
    ): VN97RevenueCampaignSnapshot {
        val existing = loadOrNull()
        check(
            existing == null ||
                existing.state != VN97RevenueCampaignState.RUNNING
        ) {
            "a revenue campaign is already running"
        }

        val first =
            application.paperTrading.startSession(
                paperConfig(request)
            )
        val now = System.currentTimeMillis()
        val campaignId =
            campaignId(
                firstSessionId = first.sessionId,
                request = request,
                createdWallTimeMillis = now,
            )
        val record =
            RevenueCampaignRecord(
                campaignId = campaignId,
                modelIdHex = first.modelIdHex,
                endpoint = request.endpoint,
                sourceId = request.sourceId,
                symbols = request.symbols.toList().sorted(),
                userGoal = request.userGoal,
                intervalMillis = request.policy.intervalMillis,
                episodesPerSession =
                    request.policy.episodesPerSession,
                targetSessions = request.policy.targetSessions,
                maxSessions = request.policy.maxSessions,
                requiresBatteryNotLow =
                    request.requiresBatteryNotLow,
                requiresCharging = request.requiresCharging,
                state = VN97RevenueCampaignState.RUNNING,
                sessionIds = listOf(first.sessionId),
                completedSessionIds = emptyList(),
                currentSessionId = first.sessionId,
                currentJobId = first.jobId,
                createdWallTimeMillis = now,
                updatedWallTimeMillis = now,
            )
        try {
            save(record)
        } catch (exc: Throwable) {
            runCatching {
                application.paperTrading.stop(first.jobId)
            }
            throw exc
        }
        return snapshot(record)
    }

    @Synchronized
    fun current(): VN97RevenueCampaignSnapshot? =
        loadOrNull()?.let(::snapshot)

    @Synchronized
    fun reconcileAfterSystemRestart():
        VN97RevenueCampaignSnapshot? {
        val record = loadOrNull() ?: return null
        if (record.state != VN97RevenueCampaignState.RUNNING) {
            return snapshot(record)
        }
        if (record.currentSessionId.isBlank()) {
            return snapshot(
                advanceAfterCompletedEvidence(record)
            )
        }

        val report =
            application.paperTrading.listReports()
                .firstOrNull {
                    it.sessionId == record.currentSessionId
                }
        if (report == null) {
            val failed =
                record.copy(
                    state = VN97RevenueCampaignState.FAILED,
                    currentSessionId = "",
                    currentJobId = 0,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(failed)
            return snapshot(failed)
        }
        if (report.state == VN97PaperTradingSessionState.COMPLETED ||
            report.state == VN97PaperTradingSessionState.FAILED ||
            report.state == VN97PaperTradingSessionState.STOPPED
        ) {
            onPaperSessionTerminal(
                sessionId = report.sessionId,
                state = report.state,
            )
        }
        return current()
    }

    @Synchronized
    fun onPaperSessionTerminal(
        sessionId: String,
        state: VN97PaperTradingSessionState,
    ) {
        val record = loadOrNull() ?: return
        if (record.state != VN97RevenueCampaignState.RUNNING) {
            return
        }
        if (record.currentSessionId != sessionId) {
            return
        }

        if (state == VN97PaperTradingSessionState.STOPPED) {
            val stopped =
                record.copy(
                    state = VN97RevenueCampaignState.STOPPED,
                    currentSessionId = "",
                    currentJobId = 0,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(stopped)
            return
        }
        if (state != VN97PaperTradingSessionState.COMPLETED) {
            val failed =
                record.copy(
                    state = VN97RevenueCampaignState.FAILED,
                    currentSessionId = "",
                    currentJobId = 0,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(failed)
            return
        }

        val completed =
            (record.completedSessionIds + sessionId)
                .distinct()
        val staged =
            record.copy(
                completedSessionIds = completed,
                currentSessionId = "",
                currentJobId = 0,
                updatedWallTimeMillis =
                    monotonicNow(record),
            )
        save(staged)

        advanceAfterCompletedEvidence(staged)
    }

    private fun advanceAfterCompletedEvidence(
        record: RevenueCampaignRecord,
    ): RevenueCampaignRecord {
        check(record.state == VN97RevenueCampaignState.RUNNING)
        check(record.currentSessionId.isBlank())
        if (record.completedSessionIds.isEmpty()) {
            val failed =
                record.copy(
                    state = VN97RevenueCampaignState.FAILED,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(failed)
            return failed
        }
        val qualification =
            qualificationFor(record)
        if (qualification == null) {
            val failed =
                record.copy(
                    state = VN97RevenueCampaignState.FAILED,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(failed)
            return failed
        }
        val directive =
            VN97RevenueCampaignPlanner.afterCompletedSession(
                qualification = qualification,
                completedSessions =
                    record.completedSessionIds.size,
                policy = record.policy(),
            )
        val nextState =
            VN97RevenueCampaignPlanner.stateFor(directive)
        if (
            directive ==
            VN97RevenueCampaignDirective.START_NEXT_SESSION
        ) {
            return startNext(record)
        }
        val terminal =
            record.copy(
                state = nextState,
                updatedWallTimeMillis =
                    monotonicNow(record),
            )
        save(terminal)
        return terminal
    }

    private fun startNext(
        record: RevenueCampaignRecord,
    ): RevenueCampaignRecord {
        check(record.state == VN97RevenueCampaignState.RUNNING)
        check(record.currentSessionId.isBlank())
        check(record.sessionIds.size < record.maxSessions)

        val next =
            application.paperTrading.startSession(
                paperConfig(record)
            )
        if (next.modelIdHex != record.modelIdHex) {
            runCatching {
                application.paperTrading.stop(next.jobId)
            }
            val failed =
                record.copy(
                    state = VN97RevenueCampaignState.FAILED,
                    updatedWallTimeMillis =
                        monotonicNow(record),
                )
            save(failed)
            return failed
        }
        val updated =
            record.copy(
                sessionIds =
                    (record.sessionIds + next.sessionId)
                        .distinct(),
                currentSessionId = next.sessionId,
                currentJobId = next.jobId,
                updatedWallTimeMillis =
                    monotonicNow(record),
            )
        save(updated)
        return updated
    }

    private fun qualificationFor(
        record: RevenueCampaignRecord,
    ): VN97RevenueQualification? {
        if (record.completedSessionIds.isEmpty()) {
            return null
        }
        val aggregate =
            application.paperTrading.performanceEvidence(
                record.completedSessionIds.toSet()
            )
        return VN97RevenueQualificationGate.evaluate(
            aggregate
        )
    }

    private fun snapshot(
        record: RevenueCampaignRecord,
    ): VN97RevenueCampaignSnapshot =
        VN97RevenueCampaignSnapshot(
            campaignId = record.campaignId,
            state = record.state,
            sessionsStarted = record.sessionIds.size,
            sessionsCompleted =
                record.completedSessionIds.size,
            currentSessionId =
                record.currentSessionId.ifBlank { null },
            currentJobId =
                record.currentJobId.takeIf { it > 0 },
            qualification =
                qualificationFor(record),
        )

    private fun paperConfig(
        request: VN97RevenueCampaignRequest,
    ): VN97PaperTradingSessionConfig =
        VN97PaperTradingSessionConfig(
            endpoint = request.endpoint,
            sourceId = request.sourceId,
            symbols = request.symbols,
            userGoal = request.userGoal,
            intervalMillis = request.policy.intervalMillis,
            maxEpisodes = request.policy.episodesPerSession,
            requiresBatteryNotLow =
                request.requiresBatteryNotLow,
            requiresCharging =
                request.requiresCharging,
        )

    private fun paperConfig(
        record: RevenueCampaignRecord,
    ): VN97PaperTradingSessionConfig =
        VN97PaperTradingSessionConfig(
            endpoint = record.endpoint,
            sourceId = record.sourceId,
            symbols = record.symbols.toSet(),
            userGoal = record.userGoal,
            intervalMillis = record.intervalMillis,
            maxEpisodes = record.episodesPerSession,
            requiresBatteryNotLow =
                record.requiresBatteryNotLow,
            requiresCharging =
                record.requiresCharging,
        )

    private fun save(
        record: RevenueCampaignRecord,
    ) {
        check(
            preferences.edit()
                .putString(KEY_RECORD, encode(record))
                .commit()
        ) {
            "failed to persist revenue campaign"
        }
    }

    private fun loadOrNull(): RevenueCampaignRecord? {
        val encoded =
            preferences.getString(KEY_RECORD, null)
                ?: return null
        return decode(encoded)
    }

    private fun encode(
        record: RevenueCampaignRecord,
    ): String =
        JSONObject()
            .put("schema", SCHEMA)
            .put("campaign_id", record.campaignId)
            .put("model_id", record.modelIdHex)
            .put("endpoint", record.endpoint)
            .put("source_id", record.sourceId)
            .put("symbols", JSONArray(record.symbols))
            .put("user_goal", record.userGoal)
            .put("interval_ms", record.intervalMillis)
            .put(
                "episodes_per_session",
                record.episodesPerSession,
            )
            .put("target_sessions", record.targetSessions)
            .put("max_sessions", record.maxSessions)
            .put(
                "battery_not_low",
                record.requiresBatteryNotLow,
            )
            .put("charging", record.requiresCharging)
            .put("state", record.state.name)
            .put("sessions", JSONArray(record.sessionIds))
            .put(
                "completed",
                JSONArray(record.completedSessionIds),
            )
            .put(
                "current_session",
                record.currentSessionId,
            )
            .put("current_job", record.currentJobId)
            .put("created_ms", record.createdWallTimeMillis)
            .put("updated_ms", record.updatedWallTimeMillis)
            .toString()

    private fun decode(
        encoded: String,
    ): RevenueCampaignRecord {
        require(
            encoded.toByteArray(StandardCharsets.UTF_8).size <=
                MAX_RECORD_BYTES
        ) {
            "revenue campaign record exceeds byte bound"
        }
        val json = JSONObject(encoded)
        require(json.getString("schema") == SCHEMA)
        val symbols =
            json.getJSONArray("symbols").stringList()
        val sessions =
            json.getJSONArray("sessions").stringList()
        val completed =
            json.getJSONArray("completed").stringList()
        return RevenueCampaignRecord(
            campaignId = json.getString("campaign_id"),
            modelIdHex = json.getString("model_id"),
            endpoint = json.getString("endpoint"),
            sourceId = json.getString("source_id"),
            symbols = symbols,
            userGoal = json.getString("user_goal"),
            intervalMillis = json.getLong("interval_ms"),
            episodesPerSession =
                json.getInt("episodes_per_session"),
            targetSessions =
                json.getInt("target_sessions"),
            maxSessions = json.getInt("max_sessions"),
            requiresBatteryNotLow =
                json.getBoolean("battery_not_low"),
            requiresCharging =
                json.getBoolean("charging"),
            state =
                VN97RevenueCampaignState.valueOf(
                    json.getString("state")
                ),
            sessionIds = sessions,
            completedSessionIds = completed,
            currentSessionId =
                json.getString("current_session"),
            currentJobId = json.getInt("current_job"),
            createdWallTimeMillis =
                json.getLong("created_ms"),
            updatedWallTimeMillis =
                json.getLong("updated_ms"),
        )
    }

    companion object {
        private const val PREFS_NAME =
            "vn97_revenue_campaign"
        private const val KEY_RECORD = "record"
        private const val SCHEMA = "VN97REV1"
        private const val MAX_RECORD_BYTES = 32 * 1024

        private fun campaignId(
            firstSessionId: String,
            request: VN97RevenueCampaignRequest,
            createdWallTimeMillis: Long,
        ): String {
            val canonical = buildString {
                append(SCHEMA)
                append('\n')
                append(firstSessionId)
                append('\n')
                append(request.endpoint)
                append('\n')
                append(request.sourceId)
                append('\n')
                append(
                    request.symbols
                        .toList()
                        .sorted()
                        .joinToString(",")
                )
                append('\n')
                append(request.userGoal)
                append('\n')
                append(createdWallTimeMillis)
            }
            return MessageDigest.getInstance("SHA-256")
                .digest(
                    canonical.toByteArray(
                        StandardCharsets.UTF_8
                    )
                )
                .joinToString("") {
                    "%02x".format(it.toInt() and 0xff)
                }
        }
    }
}

private data class RevenueCampaignRecord(
    val campaignId: String,
    val modelIdHex: String,
    val endpoint: String,
    val sourceId: String,
    val symbols: List<String>,
    val userGoal: String,
    val intervalMillis: Long,
    val episodesPerSession: Int,
    val targetSessions: Int,
    val maxSessions: Int,
    val requiresBatteryNotLow: Boolean,
    val requiresCharging: Boolean,
    val state: VN97RevenueCampaignState,
    val sessionIds: List<String>,
    val completedSessionIds: List<String>,
    val currentSessionId: String,
    val currentJobId: Int,
    val createdWallTimeMillis: Long,
    val updatedWallTimeMillis: Long,
) {
    init {
        require(campaignId.isHex64())
        require(modelIdHex.isHex64())
        require(endpoint.isNotBlank())
        require(sourceId.isNotBlank())
        require(symbols.isNotEmpty())
        require(symbols == symbols.distinct().sorted())
        require(userGoal.isNotBlank())
        policy()
        require(sessionIds.isNotEmpty())
        require(sessionIds.size <= maxSessions)
        require(sessionIds.all(String::isHex64))
        require(sessionIds.distinct().size == sessionIds.size)
        require(
            completedSessionIds.all { it in sessionIds }
        )
        require(
            completedSessionIds.distinct().size ==
                completedSessionIds.size
        )
        require(currentJobId >= 0)
        if (state == VN97RevenueCampaignState.RUNNING) {
            if (currentSessionId.isNotBlank()) {
                require(currentSessionId in sessionIds)
                require(currentJobId > 0)
            }
        } else {
            require(currentSessionId.isBlank())
            require(currentJobId == 0)
        }
        require(createdWallTimeMillis >= 0L)
        require(updatedWallTimeMillis >= createdWallTimeMillis)
    }

    fun policy(): VN97RevenueCampaignPolicy =
        VN97RevenueCampaignPolicy(
            episodesPerSession = episodesPerSession,
            targetSessions = targetSessions,
            maxSessions = maxSessions,
            intervalMillis = intervalMillis,
        )
}

private fun JSONArray.stringList(): List<String> =
    (0 until length()).map { index ->
        getString(index)
    }

private fun String.isHex64(): Boolean =
    length == 64 &&
        all { it in "0123456789abcdef" }

private fun monotonicNow(
    record: RevenueCampaignRecord,
): Long =
    maxOf(
        System.currentTimeMillis(),
        record.updatedWallTimeMillis,
    )
