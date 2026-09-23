package ai.vn97.app

import ai.vn97.runtime.VN97KnowledgeAcquisitionReview
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.util.concurrent.Executors

class VN97CapabilityAcquisitionActivity : Activity() {
    private lateinit var selectionView: TextView
    private lateinit var reviewView: TextView
    private lateinit var remoteUrlView: EditText
    private lateinit var remoteStatusView: TextView
    private lateinit var prepareRemoteButton: Button
    private lateinit var approveRemoteButton: Button
    private lateinit var rejectRemoteButton: Button
    private lateinit var choosePackageButton: Button
    private lateinit var chooseSignatureButton: Button
    private lateinit var choosePublisherKeyButton: Button
    private lateinit var reviewButton: Button
    private lateinit var acquireButton: Button
    private lateinit var clearButton: Button

    private var packageUri: Uri? = null
    private var signatureUri: Uri? = null
    private var publisherKeyUri: Uri? = null
    private var fetchedPackageSha256: String? = null
    private var busy = false

    private val worker = Executors.newSingleThreadExecutor()

    private val app: VN97Application
        get() = application as VN97Application

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val density = resources.displayMetrics.density
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
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
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }

        root.addView(
            TextView(this).apply {
                text = "VN97 Capability Acquisition"
                textSize = 24f
                gravity = Gravity.CENTER_HORIZONTAL
            },
            fullWidth(),
        )
        root.addView(
            TextView(this).apply {
                text =
                    "Signed data-only knowledge import. " +
                        "Imported text is evidence, not code, policy, " +
                        "plugin authority, or permission to control the device."
                setTextIsSelectable(true)
            },
            fullWidth(),
        )

        root.addView(
            TextView(this).apply {
                text = "Governed remote package fetch"
                textSize = 18f
            },
            fullWidth(),
        )
        remoteUrlView = EditText(this).apply {
            hint = "Exact HTTPS URL to a signed VN97CAP1 knowledge package"
            maxLines = 3
        }
        root.addView(remoteUrlView, fullWidth())

        prepareRemoteButton = Button(this).apply {
            text = "Prepare M6 fetch approval"
            setOnClickListener { prepareRemoteFetch() }
        }
        root.addView(prepareRemoteButton, fullWidth())

        val remoteApprovalRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        rejectRemoteButton = Button(this).apply {
            text = "Reject fetch"
            isEnabled = false
            setOnClickListener {
                resolveRemoteFetch(false)
            }
        }
        approveRemoteButton = Button(this).apply {
            text = "Approve & Fetch"
            isEnabled = false
            setOnClickListener {
                resolveRemoteFetch(true)
            }
        }
        remoteApprovalRow.addView(
            rejectRemoteButton,
            weighted(),
        )
        remoteApprovalRow.addView(
            approveRemoteButton,
            weighted(),
        )
        root.addView(remoteApprovalRow, fullWidth())

        remoteStatusView = TextView(this).apply {
            text =
                "No remote fetch pending. Download alone never trusts or imports a package."
            setTextIsSelectable(true)
        }
        root.addView(remoteStatusView, fullWidth())

        selectionView = TextView(this).apply {
            text = "Select VN97CAP1, VN97SIG1, and an independent Ed25519 publisher key."
            setTextIsSelectable(true)
        }
        root.addView(selectionView, fullWidth())

        val pickerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        choosePackageButton = Button(this).apply {
            text = "Package"
            setOnClickListener {
                openDocument(REQUEST_PACKAGE)
            }
        }
        chooseSignatureButton = Button(this).apply {
            text = "Signature"
            setOnClickListener {
                openDocument(REQUEST_SIGNATURE)
            }
        }
        choosePublisherKeyButton = Button(this).apply {
            text = "Publisher key"
            setOnClickListener {
                openDocument(REQUEST_PUBLISHER_KEY)
            }
        }
        pickerRow.addView(choosePackageButton, weighted())
        pickerRow.addView(chooseSignatureButton, weighted())
        pickerRow.addView(choosePublisherKeyButton, weighted())
        root.addView(pickerRow, fullWidth())

        reviewButton = Button(this).apply {
            text = "Review signed knowledge"
            isEnabled = false
            setOnClickListener { reviewSelected() }
        }
        root.addView(reviewButton, fullWidth())

        reviewView = TextView(this).apply {
            text = "No knowledge capability reviewed."
            setTextIsSelectable(true)
        }
        root.addView(reviewView, fullWidth())

        val actionRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        acquireButton = Button(this).apply {
            text = "Trust & Acquire"
            isEnabled = false
            setOnClickListener { acquireReviewed() }
        }
        clearButton = Button(this).apply {
            text = "Clear review"
            isEnabled = false
            setOnClickListener { clearReview() }
        }
        actionRow.addView(clearButton, weighted())
        actionRow.addView(acquireButton, weighted())
        root.addView(actionRow, fullWidth())

        root.addView(
            TextView(this).apply {
                text =
                    "Authority boundary: M16 acquisition creates no M6 grant, " +
                        "dynamic handler, executable plugin, script/native-library loader, " +
                        "or model replacement."
                setTextIsSelectable(true)
            },
            fullWidth(),
        )

        setContentView(scroll)
        renderPendingRemoteFetch()
        renderPendingReview()
        renderSelections()
        refreshControls()
    }

    override fun onResume() {
        super.onResume()
        if (::reviewView.isInitialized) {
            renderPendingRemoteFetch()
            renderPendingReview()
            renderSelections()
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
        val uri = data?.data ?: return
        persistReadPermission(uri, data.flags)
        when (requestCode) {
            REQUEST_PACKAGE -> {
                packageUri = uri
                fetchedPackageSha256 = null
            }
            REQUEST_SIGNATURE -> signatureUri = uri
            REQUEST_PUBLISHER_KEY -> publisherKeyUri = uri
            else -> return
        }
        renderSelections()
        refreshControls()
    }

    private fun openDocument(requestCode: Int) {
        if (busy) return
        startActivityForResult(
            Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "*/*"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
            },
            requestCode,
        )
    }

    private fun persistReadPermission(
        uri: Uri,
        resultFlags: Int,
    ) {
        val granted =
            resultFlags and Intent.FLAG_GRANT_READ_URI_PERMISSION
        if (granted == 0) return
        runCatching {
            contentResolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        }
    }

    private fun reviewSelected() {
        val packageDocument = packageUri
        val remotePackage =
            fetchedPackageSha256
        val signatureDocument = signatureUri
        val publisherKeyDocument = publisherKeyUri
        if (
            (
                packageDocument == null &&
                    remotePackage == null
            ) ||
            signatureDocument == null ||
            publisherKeyDocument == null
        ) {
            reviewView.text =
                "Select or fetch a package, then select signature and publisher key."
            return
        }
        setBusy(true)
        reviewView.text =
            "Verifying package integrity, Ed25519 publisher scope, and VN97KN1 data…"
        worker.execute {
            val result = runCatching {
                if (remotePackage != null) {
                    app.knowledgeAcquisition
                        .reviewFetchedPackage(
                            packageSha256 =
                                remotePackage,
                            signatureUri =
                                signatureDocument,
                            publisherKeyUri =
                                publisherKeyDocument,
                        )
                } else {
                    app.knowledgeAcquisition.review(
                        packageUri =
                            checkNotNull(
                                packageDocument
                            ),
                        signatureUri =
                            signatureDocument,
                        publisherKeyUri =
                            publisherKeyDocument,
                    )
                }
            }
            runOnUiThread {
                result.onSuccess { review ->
                    renderReview(review)
                }.onFailure { exc ->
                    reviewView.text =
                        "Knowledge review rejected: " +
                            (exc.message ?: exc::class.java.simpleName)
                }
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun acquireReviewed() {
        if (app.knowledgeAcquisition.pendingReview() == null) {
            reviewView.text =
                "No reviewed knowledge capability is pending."
            refreshControls()
            return
        }
        setBusy(true)
        reviewView.text =
            "Re-verifying exact reviewed bytes, enrolling publisher trust, and importing into VN97MEM1…"
        worker.execute {
            val result = runCatching {
                app.knowledgeAcquisition.acquireReviewed()
            }
            runOnUiThread {
                result.onSuccess { acquired ->
                    reviewView.text = buildString {
                        append("Knowledge acquisition completed.")
                        append("\ncapability=")
                        append(acquired.capabilityId)
                        append(" v")
                        append(acquired.capabilityVersion)
                        append("\npackage_sha256=")
                        append(acquired.packageSha256)
                        append("\npublisher=")
                        append(acquired.publisherKeyId)
                        append("\nVN97MEM1_records=")
                        append(acquired.recordIds.size)
                        append("\nalready_acquired=")
                        append(acquired.alreadyAcquired)
                        append(
                            "\nAuthority remains evidence_only; no new tool/device permission was created."
                        )
                    }
                }.onFailure { exc ->
                    reviewView.text =
                        "Knowledge acquisition failed closed: " +
                            (exc.message ?: exc::class.java.simpleName)
                }
                setBusy(false)
                renderPendingReview(
                    preserveCompletedText =
                        result.isSuccess
                )
                refreshControls()
            }
        }
    }

    private fun clearReview() {
        if (busy) return
        runCatching {
            app.knowledgeAcquisition.clearReview()
        }.onSuccess {
            reviewView.text =
                "Reviewed capability cleared. No trust or knowledge was acquired by clearing."
        }.onFailure { exc ->
            reviewView.text =
                "Clear review failed: " +
                    (exc.message ?: exc::class.java.simpleName)
        }
        refreshControls()
    }

    private fun renderPendingReview(
        preserveCompletedText: Boolean = false,
    ) {
        val review =
            runCatching {
                app.knowledgeAcquisition.pendingReview()
            }.getOrNull()
        if (review != null) {
            renderReview(review)
        } else if (!preserveCompletedText) {
            reviewView.text =
                "No knowledge capability reviewed."
        }
    }

    private fun renderReview(
        review: VN97KnowledgeAcquisitionReview,
    ) {
        reviewView.text = buildString {
            append("REVIEW ONLY — VN97MEM1 unchanged until Trust & Acquire.")
            append("\ncapability=")
            append(review.capabilityId)
            append(" v")
            append(review.capabilityVersion)
            append("\nrecords=")
            append(review.recordCount)
            append("\npackage_sha256=")
            append(review.packageSha256)
            append("\nsignature_sha256=")
            append(review.signatureSha256)
            append("\npayload_sha256=")
            append(review.payloadSha256)
            append("\npublisher_key_id=")
            append(review.publisherKeyId)
            append("\npublisher_key_sha256=")
            append(review.publisherKeySha256)
            append("\npublisher_previously_trusted=")
            append(review.publisherPreviouslyTrusted)
            append("\nsource_origin=")
            append(review.sourceOrigin.take(1024))
            append("\nsource_license=")
            append(review.sourceLicense.take(128))
            append(
                "\nAccepted scope: signed VN97KN1 knowledge data only."
            )
        }
    }

    private fun renderSelections() {
        selectionView.text = buildString {
            append("Package: ")
            val remote =
                fetchedPackageSha256
            if (remote != null) {
                append("remote:")
                append(remote)
            } else {
                append(selectionLabel(packageUri))
            }
            append("\nSignature: ")
            append(selectionLabel(signatureUri))
            append("\nPublisher key: ")
            append(selectionLabel(publisherKeyUri))
        }
    }

    private fun selectionLabel(uri: Uri?): String =
        uri?.lastPathSegment
            ?.replace('\n', ' ')
            ?.replace('\r', ' ')
            ?.take(160)
            ?: "not selected"

    private fun prepareRemoteFetch() {
        if (busy) return
        val url =
            remoteUrlView.text.toString().trim()
        if (url.isEmpty()) {
            remoteStatusView.text =
                "Enter an exact HTTPS package URL first."
            return
        }
        setBusy(true)
        worker.execute {
            val result = runCatching {
                app.remoteCapabilityFetch.prepare(url)
            }
            runOnUiThread {
                result.onSuccess { approval ->
                    remoteStatusView.text =
                        buildString {
                            append(
                                "M6 APPROVAL REQUIRED — no bytes fetched yet."
                            )
                            append("\nurl=")
                            append(
                                approval.canonicalUrl
                            )
                            append("\nrequest_digest=")
                            append(
                                approval.requestDigest
                            )
                            append("\nexpires_ns=")
                            append(
                                approval.expiresNs
                            )
                            append(
                                "\nReview exact scope before Approve & Fetch."
                            )
                        }
                }.onFailure { exc ->
                    remoteStatusView.text =
                        "Remote fetch preparation rejected: " +
                            (
                                exc.message ?:
                                    exc::class.java.simpleName
                            )
                }
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun resolveRemoteFetch(
        approved: Boolean,
    ) {
        if (busy) return
        if (
            app.remoteCapabilityFetch.pending() ==
                null
        ) {
            remoteStatusView.text =
                "No remote fetch approval is pending."
            refreshControls()
            return
        }
        setBusy(true)
        remoteStatusView.text =
            if (approved) {
                "Fetching exact M6-approved artifact…"
            } else {
                "Rejecting remote fetch…"
            }
        worker.execute {
            val result = runCatching {
                app.remoteCapabilityFetch.resolve(
                    approved
                )
            }
            runOnUiThread {
                result.onSuccess { resolved ->
                    if (!resolved.approved) {
                        remoteStatusView.text =
                            "Remote fetch rejected. No network artifact was accepted."
                    } else {
                        val artifact =
                            checkNotNull(
                                resolved.artifact
                            )
                        fetchedPackageSha256 =
                            artifact.packageSha256
                        packageUri = null
                        remoteStatusView.text =
                            buildString {
                                append(
                                    "Fetched but NOT trusted/imported."
                                )
                                append("\nurl=")
                                append(
                                    artifact.canonicalUrl
                                )
                                append("\npackage_sha256=")
                                append(
                                    artifact.packageSha256
                                )
                                append("\nbytes=")
                                append(
                                    artifact.sizeBytes
                                )
                                append("\ncapability=")
                                append(
                                    artifact.capabilityId
                                )
                                append(" v")
                                append(
                                    artifact.capabilityVersion
                                )
                                append("\nkind=")
                                append(
                                    artifact.kind
                                )
                                append("\nM6_receipt=")
                                append(
                                    resolved.receiptId
                                )
                                append(
                                    "\nNext: select VN97SIG1 + publisher key, then Review signed knowledge."
                                )
                            }
                    }
                }.onFailure { exc ->
                    remoteStatusView.text =
                        "Remote fetch failed closed: " +
                            (
                                exc.message ?:
                                    exc::class.java.simpleName
                            )
                }
                renderSelections()
                setBusy(false)
                refreshControls()
            }
        }
    }

    private fun renderPendingRemoteFetch() {
        if (!::remoteStatusView.isInitialized) {
            return
        }
        val pending =
            runCatching {
                app.remoteCapabilityFetch.pending()
            }.getOrNull()
        if (pending != null) {
            remoteUrlView.setText(
                pending.canonicalUrl
            )
            remoteStatusView.text =
                buildString {
                    append(
                        "M6 APPROVAL REQUIRED — no bytes fetched yet."
                    )
                    append("\nurl=")
                    append(pending.canonicalUrl)
                    append("\nrequest_digest=")
                    append(pending.requestDigest)
                    append("\nexpires_ns=")
                    append(pending.expiresNs)
                }
        }
    }

    private fun setBusy(value: Boolean) {
        busy = value
        refreshControls()
    }

    private fun refreshControls() {
        if (!::reviewButton.isInitialized) return
        val allSelected =
            (
                packageUri != null ||
                    fetchedPackageSha256 != null
            ) &&
                signatureUri != null &&
                publisherKeyUri != null
        val hasReview =
            runCatching {
                app.knowledgeAcquisition.pendingReview() != null
            }.getOrDefault(false)
        val remotePending =
            runCatching {
                app.remoteCapabilityFetch.pending() != null
            }.getOrDefault(false)
        choosePackageButton.isEnabled = !busy
        chooseSignatureButton.isEnabled = !busy
        choosePublisherKeyButton.isEnabled = !busy
        prepareRemoteButton.isEnabled =
            !busy && !remotePending
        approveRemoteButton.isEnabled =
            !busy && remotePending
        rejectRemoteButton.isEnabled =
            !busy && remotePending
        remoteUrlView.isEnabled =
            !busy && !remotePending
        reviewButton.isEnabled =
            !busy && allSelected && !hasReview
        acquireButton.isEnabled = !busy && hasReview
        clearButton.isEnabled = !busy && hasReview
    }

    private fun fullWidth() =
        LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )

    private fun weighted() =
        LinearLayout.LayoutParams(
            0,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            1f,
        )

    companion object {
        private const val REQUEST_PACKAGE = 5101
        private const val REQUEST_SIGNATURE = 5102
        private const val REQUEST_PUBLISHER_KEY = 5103
    }
}
