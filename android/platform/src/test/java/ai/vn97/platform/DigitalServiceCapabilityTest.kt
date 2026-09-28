package ai.vn97.platform

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DigitalServiceCapabilityTest {
    private class FakeActions : M6AndroidActionPort {
        var prepared: Pair<VN97LocalDigitalService, String>? = null

        override fun prepareDigitalService(
            service: VN97LocalDigitalService,
            source: String,
        ): String {
            prepared = service to source
            return "service:draft:${service.name}:digest"
        }

        override fun launchPackage(packageName: String) = error("unused")
        override fun writeClipboard(text: String) = error("unused")
        override fun gameTap(packageName: String, xBasisPoints: Int, yBasisPoints: Int, durationMillis: Long) = error("unused")
        override fun gameSwipe(packageName: String, startXBasisPoints: Int, startYBasisPoints: Int, endXBasisPoints: Int, endYBasisPoints: Int, durationMillis: Long) = error("unused")
        override fun gameMultiTouch(packageName: String, strokes: List<VN97GameTouchStroke>) = error("unused")
        override fun gameBack(packageName: String) = error("unused")
        override fun fetchCapabilityArtifact(url: String) = error("unused")
    }

    @Test
    fun approvedTypedRequestInvokesOnlySelectedLocalService() {
        val actions = FakeActions()
        val capabilities = M6AndroidProductionCapabilities(actions)
        val registry = capabilities.createSealedRegistry()
        val scope = M6CapabilityScope.fromMap(
            mapOf(
                M6AndroidProductionCapabilities.DIGITAL_SERVICE_SCOPE to
                    VN97LocalDigitalService.TABLE_CLEANUP.name
            )
        )
        val request = M6ExternalActionRequest(
            planId = "plan-1",
            stepId = 1,
            objective = "clean customer table",
            capabilityId = M6AndroidProductionCapabilities.DIGITAL_SERVICE_PREPARE,
            scope = scope,
            payloadJson = "{\"source\":\"A,B\\n1,2\"}",
        )
        val descriptor = registry.validate(request)
        assertTrue(descriptor.approvalRequired)
        val outcome = registry.executeAuthorized(
            M6AuthorizedAction(
                request = request,
                principal = "vn97-assistant",
                lease = M6CapabilityLease(
                    leaseId = "lease-1",
                    principal = "vn97-assistant",
                    capabilityId = request.capabilityId,
                    scopeDigest = scope.digest,
                    issuedNs = 1,
                    expiresNs = 2,
                    maxUses = 1,
                ),
                approvalId = "approval-1",
            )
        )
        assertEquals(
            VN97LocalDigitalService.TABLE_CLEANUP to "A,B\n1,2",
            actions.prepared,
        )
        assertEquals(
            "service:draft:TABLE_CLEANUP:digest",
            outcome.result,
        )
    }

    @Test
    fun malformedOrOversizedPayloadIsRejectedBeforeHandler() {
        val actions = FakeActions()
        val registry = M6AndroidProductionCapabilities(actions).createSealedRegistry()
        val scope = M6CapabilityScope.fromMap(
            mapOf(
                M6AndroidProductionCapabilities.DIGITAL_SERVICE_SCOPE to
                    VN97LocalDigitalService.DOCUMENT_FORMAT.name
            )
        )
        fun request(payload: String) = M6ExternalActionRequest(
            planId = "plan-2",
            stepId = 2,
            objective = "format text",
            capabilityId = M6AndroidProductionCapabilities.DIGITAL_SERVICE_PREPARE,
            scope = scope,
            payloadJson = payload,
        )
        listOf(
            "{\"source\":\"x\",\"extra\":true}",
            "{\"source\":\"\\/\"}",
            "{\"source\":\"${"x".repeat(48 * 1024 + 1)}\"}",
        ).forEach { payload ->
            assertTrue(runCatching { registry.validate(request(payload)) }.isFailure)
        }
        assertEquals(null, actions.prepared)
    }
}
