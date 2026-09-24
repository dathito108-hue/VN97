package ai.vn97.app

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

class VN97SelfImprovementActivity : Activity() {
    private lateinit var objectiveView: EditText
    private lateinit var selectionView: TextView
    private lateinit var statusView: TextView
    private lateinit var packageButton: Button
    private lateinit var signatureButton: Button
    private lateinit var publisherKeyButton: Button
    private lateinit var reviewButton: Button
    private lateinit var evaluateButton: Button
    private lateinit var promoteButton: Button
    private lateinit var rejectButton: Button

    private var packageUri: Uri? = null
    private var signatureUri: Uri? = null
    private var publisherKeyUri: Uri? = null
    private var busy = false

    private val worker =
        Executors.newSingleThreadExecutor()

    private val app: VN97Application
        get() =
            application as VN97Application

    override fun onCreate(
        savedInstanceState: Bundle?,
    ) {
        super.onCreate(savedInstanceState)

        val density =
            resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation =
                LinearLayout.VERTICAL
            setPadding(
                (16 * density).toInt(),
                (16 * density).toInt(),
                (16 * density).toInt(),
                (24 * density).toInt(),
            )
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(
                root,
                ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams
                        .MATCH_PARENT,
                    ViewGroup.LayoutParams
                        .WRAP_CONTENT,
                ),
            )
        }

        root.addView(
            TextView(this).apply {
                text =
                    "VN97 Controlled Self-Improvement"
                textSize = 24f
            },
            fullWidth(),
        )
        root.addView(
            TextView(this).apply {
                text =
                    "M17A reviews a signed newer canonical VN97 model against the exact active baseline. " +
                        "M17B evaluates baseline and candidate on the fixed VN97HELD1 suite without activation. " +
                        "M17C can promote only a passing, identity-bound candidate after an explicit confirmation here, with durable rollback evidence."
                setTextIsSelectable(true)
            },
            fullWidth(),
        )

        objectiveView =
            EditText(this).apply {
                hint =
                    "Improvement objective"
                maxLines = 5
            }
        root.addView(
            objectiveView,
            fullWidth(),
        )

        selectionView =
            TextView(this).apply {
                setTextIsSelectable(true)
            }
        root.addView(
            selectionView,
            fullWidth(),
        )

        val pickerRow =
            LinearLayout(this).apply {
                orientation =
                    LinearLayout.HORIZONTAL
            }
        packageButton =
            Button(this).apply {
                text = "Candidate"
                setOnClickListener {
                    openDocument(
                        REQUEST_PACKAGE
                    )
                }
            }
        signatureButton =
            Button(this).apply {
                text = "Signature"
                setOnClickListener {
                    openDocument(
                        REQUEST_SIGNATURE
                    )
                }
            }
        publisherKeyButton =
            Button(this).apply {
                text = "Publisher key"
                setOnClickListener {
                    openDocument(
                        REQUEST_PUBLISHER_KEY
                    )
                }
            }
        pickerRow.addView(
            packageButton,
            weighted(),
        )
        pickerRow.addView(
            signatureButton,
            weighted(),
        )
        pickerRow.addView(
            publisherKeyButton,
            weighted(),
        )
        root.addView(
            pickerRow,
            fullWidth(),
        )

        reviewButton =
            Button(this).apply {
                text =
                    "Review improvement candidate"
                setOnClickListener {
                    reviewCandidate()
                }
            }
        root.addView(
            reviewButton,
            fullWidth(),
        )

        evaluateButton =
            Button(this).apply {
                text =
                    "Evaluate held-out candidate"
                setOnClickListener {
                    evaluateCandidate()
                }
            }
        root.addView(
            evaluateButton,
            fullWidth(),
        )

        promoteButton =
            Button(this).apply {
                text =
                    "Promote evaluated candidate"
                setOnClickListener {
                    confirmPromotion()
                }
            }
        root.addView(
            promoteButton,
            fullWidth(),
        )

        rejectButton =
            Button(this).apply {
                text =
                    "Reject candidate"
                setOnClickListener {
                    rejectCandidate()
                }
            }
        root.addView(
            rejectButton,
            fullWidth(),
        )

        statusView =
            TextView(this).apply {
                setTextIsSelectable(true)
            }
        root.addView(
            statusView,
            fullWidth(),
        )

        root.addView(
            TextView(this).apply {
                text =
                    "M17A/B/C authority boundary: no weight training, no autonomous promotion, no code modification, no M6 grant, and no bypass of signed VN97CAP1/VN97SIG1 provisioning. Promotion requires a passing held-out evaluation plus an explicit human confirmation."
                setTextIsSelectable(true)
            },
            fullWidth(),
        )

        setContentView(scroll)
        renderSelections()
        renderPending()
        refreshControls()
    }

    override fun onResume() {
        super.onResume()
        if (::statusView.isInitialized) {
            renderPending()
            refreshControls()
        }
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?,
    ) {
        super.onActivityResult(
            requestCode,
            resultCode,
            data,
        )
        if (resultCode != RESULT_OK) {
            return
        }
        val uri =
            data?.data ?: return
        persistReadPermission(
            uri,
            data.flags,
        )
        when (requestCode) {
            REQUEST_PACKAGE ->
                packageUri = uri
            REQUEST_SIGNATURE ->
                signatureUri = uri
            REQUEST_PUBLISHER_KEY ->
                publisherKeyUri = uri
            else -> return
        }
        renderSelections()
        refreshControls()
    }

    private fun reviewCandidate() {
        if (busy) return
        val candidatePackage =
            packageUri
        val signature =
            signatureUri
        val publisherKey =
            publisherKeyUri
        val objective =
            objectiveView.text
                .toString()
                .trim()
        if (
            candidatePackage == null ||
            signature == null ||
            publisherKey == null ||
            objective.isBlank()
        ) {
            statusView.text =
                "Select candidate package, signature, publisher key, and enter a non-empty improvement objective."
            return
        }

        setBusy(true)
        statusView.text =
            "Verifying signed canonical model candidate and binding it to the active baseline…"
        worker.execute {
            val result = runCatching {
                app.provisioner
                    .reviewSelfImprovement(
                        packageUri =
                            candidatePackage,
                        signatureUri =
                            signature,
                        publisherKeyUri =
                            publisherKey,
                        objective =
                            objective,
                    )
            }
            runOnUiThread {
                result.onSuccess { pair ->
                    val review = pair.first
                    val candidate = pair.second
                    statusView.text =
                        buildString {
                            append(
                                "CONTROLLED CANDIDATE REVIEWED — activation remains blocked."
                            )
                            append(
                                "\ncandidate_id="
                            )
                            append(
                                candidate.candidateId
                            )
                            append(
                                "\nbaseline_version="
                            )
                            append(
                                candidate.spec
                                    .baselineCapabilityVersion
                            )
                            append(
                                "\nbaseline_artifact_sha256="
                            )
                            append(
                                candidate.spec
                                    .baselineArtifactSha256
                            )
                            append(
                                "\ncandidate_version="
                            )
                            append(
                                review.capabilityVersion
                            )
                            append(
                                "\ncandidate_package_sha256="
                            )
                            append(
                                review.packageSha256
                            )
                            append(
                                "\npublisher="
                            )
                            append(
                                review.publisherKeyId
                            )
                            append(
                                "\nplan_sha256="
                            )
                            append(
                                review.planSha256
                            )
                            append(
                                "\nstate="
                            )
                            append(
                                candidate.state.name
                            )
                            append(
                                "\nNext gate: M17B held-out candidate evaluation. Normal activation is denied."
                            )
                        }
                }.onFailure { exc ->
                    statusView.text =
                        "Candidate review rejected: " +
                            (
                                exc.message ?:
                                    exc::class.java
                                        .simpleName
                            )
                }
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun evaluateCandidate() {
        if (busy) return
        setBusy(true)
        statusView.text =
            "Running fixed VN97HELD1 evaluation against exact baseline and non-active candidate…"
        worker.execute {
            val result = runCatching {
                app.provisioner
                    .evaluatePendingImprovementCandidate()
            }
            runOnUiThread {
                result.onSuccess { evaluation ->
                    statusView.text =
                        buildString {
                            append(
                                "HELD-OUT EVALUATION COMPLETE — candidate is still not active."
                            )
                            append(
                                "\nevaluation_id="
                            )
                            append(
                                evaluation.evaluationId
                            )
                            append(
                                "\nsuite_sha256="
                            )
                            append(
                                evaluation.suiteSha256
                            )
                            append(
                                "\npassed="
                            )
                            append(
                                evaluation.decision
                                    .passed
                            )
                            append(
                                "\nbaseline_nll_per_byte="
                            )
                            append(
                                evaluation.baseline
                                    .nllPerUtf8Byte
                            )
                            append(
                                "\ncandidate_nll_per_byte="
                            )
                            append(
                                evaluation.candidate
                                    .nllPerUtf8Byte
                            )
                            append(
                                "\nbaseline_top1="
                            )
                            append(
                                evaluation.baseline
                                    .top1Accuracy
                            )
                            append(
                                "\ncandidate_top1="
                            )
                            append(
                                evaluation.candidate
                                    .top1Accuracy
                            )
                            append(
                                "\nbaseline_prefill_p95_ns="
                            )
                            append(
                                evaluation.baseline
                                    .prefillP95Nanos
                            )
                            append(
                                "\ncandidate_prefill_p95_ns="
                            )
                            append(
                                evaluation.candidate
                                    .prefillP95Nanos
                            )
                            append(
                                "\nreasons="
                            )
                            append(
                                if (
                                    evaluation.decision
                                        .reasons
                                        .isEmpty()
                                ) {
                                    "none"
                                } else {
                                    evaluation.decision
                                        .reasons
                                        .joinToString(
                                            " | "
                                        )
                                }
                            )
                            append(
                                "\nNext gate: M17C controlled promotion. No activation occurred."
                            )
                        }
                }.onFailure { exc ->
                    statusView.text =
                        "Held-out evaluation failed: " +
                            (
                                exc.message ?:
                                    exc::class.java
                                        .simpleName
                            )
                }
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun confirmPromotion() {
        if (busy) return
        val candidate =
            runCatching {
                app.provisioner
                    .pendingImprovementCandidate()
            }.getOrNull()
                ?: run {
                    statusView.text =
                        "No controlled improvement candidate is pending."
                    refreshControls()
                    return
                }
        val evaluation =
            runCatching {
                app.provisioner
                    .pendingImprovementEvaluation()
            }.getOrNull()
        if (
            evaluation == null ||
            !evaluation.decision.passed
        ) {
            statusView.text =
                "Promotion is blocked until the exact candidate has a passing M17B held-out evaluation."
            refreshControls()
            return
        }

        AlertDialog.Builder(this)
            .setTitle(
                "Promote evaluated VN97 model?"
            )
            .setMessage(
                buildString {
                    append(
                        "This explicit approval will replace the active canonical VN97 model with the reviewed candidate. "
                    )
                    append(
                        "M17C will verify the exact activated artifact and automatically roll back to the exact baseline if post-activation verification fails."
                    )
                    append("\n\ncandidate_id=")
                    append(candidate.candidateId)
                    append("\nevaluation_id=")
                    append(evaluation.evaluationId)
                    append("\ncandidate_version=")
                    append(
                        candidate.spec
                            .candidateCapabilityVersion
                    )
                }
            )
            .setNegativeButton(
                "Cancel",
                null,
            )
            .setPositiveButton(
                "Promote candidate",
            ) { _, _ ->
                promoteCandidate()
            }
            .show()
    }

    private fun promoteCandidate() {
        if (busy) return
        setBusy(true)
        statusView.text =
            "Executing explicitly approved M17C promotion with post-activation native verification…"
        worker.execute {
            val result = runCatching {
                app.provisioner
                    .promotePendingImprovementCandidate(
                        userApproved = true
                    )
            }
            runOnUiThread {
                result.onSuccess { promotion ->
                    statusView.text =
                        buildString {
                            append(
                                "CONTROLLED PROMOTION COMPLETE."
                            )
                            append("\npromotion_id=")
                            append(
                                promotion.promotionId
                            )
                            append("\nstate=")
                            append(
                                promotion.state.name
                            )
                            if (
                                promotion
                                    .resultingActivationId
                                    .isNotEmpty()
                            ) {
                                append(
                                    "\nresulting_activation_id="
                                )
                                append(
                                    promotion
                                        .resultingActivationId
                                )
                            }
                            if (
                                promotion
                                    .resultingArtifactSha256
                                    .isNotEmpty()
                            ) {
                                append(
                                    "\nresulting_artifact_sha256="
                                )
                                append(
                                    promotion
                                        .resultingArtifactSha256
                                )
                            }
                            if (
                                promotion
                                    .rollbackRestoredActivationId
                                    .isNotEmpty()
                            ) {
                                append(
                                    "\nrollback_restored_activation_id="
                                )
                                append(
                                    promotion
                                        .rollbackRestoredActivationId
                                )
                            }
                            if (
                                promotion.detail
                                    .isNotEmpty()
                            ) {
                                append("\ndetail=")
                                append(
                                    promotion.detail
                                )
                            }
                        }
                }.onFailure { exc ->
                    statusView.text =
                        "Controlled promotion failed closed: " +
                            (
                                exc.message ?:
                                    exc::class.java
                                        .simpleName
                            )
                }
                setBusy(false)
                renderPending()
                refreshControls()
            }
        }
    }

    private fun rejectCandidate() {
        if (busy) return
        setBusy(true)
        worker.execute {
            val result = runCatching {
                app.provisioner
                    .rejectImprovementCandidate()
            }
            runOnUiThread {
                result.onSuccess { candidate ->
                    statusView.text =
                        "Candidate rejected durably: " +
                            candidate.candidateId
                }.onFailure { exc ->
                    statusView.text =
                        "Candidate rejection failed: " +
                            (
                                exc.message ?:
                                    exc::class.java
                                        .simpleName
                            )
                }
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun renderPending() {
        val candidate =
            runCatching {
                app.provisioner
                    .pendingImprovementCandidate()
            }.getOrNull()
        if (candidate != null) {
            val evaluation =
                runCatching {
                    app.provisioner
                        .pendingImprovementEvaluation()
                }.getOrNull()
            val promotion =
                runCatching {
                    app.provisioner
                        .pendingPromotionAttempt()
                }.getOrNull()
            statusView.text =
                buildString {
                    append(
                        "CONTROLLED CANDIDATE PENDING — activation blocked."
                    )
                    append("\ncandidate_id=")
                    append(candidate.candidateId)
                    append("\nstate=")
                    append(candidate.state.name)
                    append(
                        "\nobjective="
                    )
                    append(
                        candidate.spec.objective
                            .take(2048)
                    )
                    if (evaluation != null) {
                        append(
                            "\nevaluation_id="
                        )
                        append(
                            evaluation.evaluationId
                        )
                        append(
                            "\nevaluation_passed="
                        )
                        append(
                            evaluation.decision
                                .passed
                        )
                    }
                    if (promotion != null) {
                        append(
                            "\npromotion_id="
                        )
                        append(
                            promotion.promotionId
                        )
                        append(
                            "\npromotion_state="
                        )
                        append(
                            promotion.state.name
                        )
                    } else if (
                        evaluation?.decision?.passed ==
                            true
                    ) {
                        append(
                            "\npromotion_ready=true — explicit confirmation required"
                        )
                    }
                }
        } else if (
            statusView.text.isNullOrBlank()
        ) {
            statusView.text =
                "No controlled improvement candidate is pending."
        }
    }

    private fun openDocument(
        requestCode: Int,
    ) {
        if (busy) return
        startActivityForResult(
            Intent(
                Intent.ACTION_OPEN_DOCUMENT
            ).apply {
                addCategory(
                    Intent.CATEGORY_OPENABLE
                )
                type = "*/*"
                addFlags(
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
                addFlags(
                    Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION
                )
            },
            requestCode,
        )
    }

    private fun persistReadPermission(
        uri: Uri,
        flags: Int,
    ) {
        if (
            flags and
                Intent.FLAG_GRANT_READ_URI_PERMISSION ==
                0
        ) {
            return
        }
        runCatching {
            contentResolver
                .takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
        }
    }

    private fun renderSelections() {
        if (!::selectionView.isInitialized) {
            return
        }
        selectionView.text =
            buildString {
                append("Candidate: ")
                append(label(packageUri))
                append("\nSignature: ")
                append(label(signatureUri))
                append("\nPublisher key: ")
                append(label(publisherKeyUri))
            }
    }

    private fun label(
        uri: Uri?,
    ): String =
        uri?.lastPathSegment
            ?.replace('\n', ' ')
            ?.replace('\r', ' ')
            ?.take(160)
            ?: "not selected"

    private fun setBusy(
        value: Boolean,
    ) {
        busy = value
        refreshControls()
    }

    private fun refreshControls() {
        if (!::reviewButton.isInitialized) {
            return
        }
        val pending =
            runCatching {
                app.provisioner
                    .pendingImprovementCandidate() !=
                    null
            }.getOrDefault(false)
        val evaluation =
            if (pending) {
                runCatching {
                    app.provisioner
                        .pendingImprovementEvaluation()
                }.getOrNull()
            } else {
                null
            }
        val promotionPending =
            if (pending) {
                runCatching {
                    app.provisioner
                        .pendingPromotionAttempt() !=
                        null
                }.getOrDefault(false)
            } else {
                false
            }
        packageButton.isEnabled =
            !busy && !pending
        signatureButton.isEnabled =
            !busy && !pending
        publisherKeyButton.isEnabled =
            !busy && !pending
        objectiveView.isEnabled =
            !busy && !pending
        reviewButton.isEnabled =
            !busy &&
                !pending &&
                packageUri != null &&
                signatureUri != null &&
                publisherKeyUri != null &&
                objectiveView.text
                    .toString()
                    .isNotBlank()
        evaluateButton.isEnabled =
            !busy &&
                pending &&
                evaluation == null &&
                !promotionPending
        promoteButton.isEnabled =
            !busy &&
                pending &&
                evaluation?.decision?.passed ==
                    true &&
                !promotionPending
        rejectButton.isEnabled =
            !busy &&
                pending &&
                !promotionPending
    }

    private fun fullWidth() =
        LinearLayout.LayoutParams(
            ViewGroup.LayoutParams
                .MATCH_PARENT,
            ViewGroup.LayoutParams
                .WRAP_CONTENT,
        )

    private fun weighted() =
        LinearLayout.LayoutParams(
            0,
            ViewGroup.LayoutParams
                .WRAP_CONTENT,
            1f,
        )

    companion object {
        private const val REQUEST_PACKAGE =
            5201
        private const val REQUEST_SIGNATURE =
            5202
        private const val REQUEST_PUBLISHER_KEY =
            5203
    }
}
