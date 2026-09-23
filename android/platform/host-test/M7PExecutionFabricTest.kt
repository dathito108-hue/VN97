import ai.vn97.platform.*
import ai.vn97.runtime.*

private class RecordingApproval : M6ApprovalControllerPort {
    override fun createPrompt(requestDigest:String,principal:String,presentationJson:String,promptTtlNs:Long,nowNs:Long)=M6ApprovalPrompt("p",principal,requestDigest,presentationJson,nowNs,nowNs+promptTtlNs)
    override fun resolve(prompt:M6ApprovalPrompt,approved:Boolean,approvalTtlNs:Long,nowNs:Long)=if(approved) M6ApprovalToken("id","local-user",prompt.principal,prompt.requestDigest,nowNs,nowNs+approvalTtlNs,"sig") else null
    override fun verify(token:M6ApprovalToken,requestDigest:String,principal:String,nowNs:Long){ if(token.requestDigest!=requestDigest || token.principal!=principal) throw IllegalStateException("bad approval") }
}

fun main(){
    val step=NativePlannerStep(1,NativePlanStepSpec(NativeStepKind.EXTERNAL,"echo"),NativeStepStatus.WAITING_EXTERNAL)
    val controller=NativePlanController(NativePlan("ab".repeat(32),"g",listOf(step),NativePlanStatus.WAITING_EXTERNAL))
    val descriptor=M6CapabilityDescriptor("test.echo",setOf("target"),approvalRequired=true,maxLeaseUses=1)
    val binder=M6ExternalIntentBinder(listOf(descriptor))
    val request=binder.bind(controller,NativeExternalIntent("test.echo",mapOf("target" to "alpha"),"{\"message\":\"hi\"}"))
    check(request.requestDigest=="e9a4311db6a33f19a118ce9c9cb9f7fcffde650594a562e93a5280f36e050d73")
    check(request.scope.digest=="1b22e7f4a338b62840bbd0d9361ae0eda309ead2df2b1e8b455ea0237eb5174e")
    val parityAudit=M6InMemoryActionAudit()
    val parityReceipt=parityAudit.append(
        status=M6ReceiptStatus.SUCCEEDED,
        request=request,
        principal="runtime.user",
        leaseId="00112233445566778899aabbccddeeff",
        approvalId="ffeeddccbbaa99887766554433221100",
        outcome=M6ActionOutcome(true,"external-ok",0.9,listOf(7L)),
        timestampNs=21,
    )
    check(parityReceipt.receiptId=="e59095fc030b9fba6ab644534ff36bf10776db06a7ffd652c3284be4caf03f84") { parityReceipt.receiptId }

    val approvals=RecordingApproval()
    val handoff=M6ExternalApprovalHandoff(approvals)
    val prompt=handoff.createPrompt(request,"runtime.user",1000,10)
    val token=checkNotNull(handoff.resolve(prompt,true,100,20))

    var calls=0
    val registry=M6TypedCapabilityRegistry()
    registry.register(descriptor,M6CapabilityHandler { action -> calls++; check(action.request.requestDigest==request.requestDigest); M6ActionOutcome(true,"external-ok",0.9,listOf(7L)) })
    registry.seal()
    val grant=M6PolicyGrant("runtime.user","test.echo",request.scope.digest,maxLeaseUses=1)
    val audit=M6InMemoryActionAudit()
    val authority=M6DenyByDefaultAuthorityGate(listOf(grant),approvals)
    val fabric=M6ExternalExecutionFabric(registry,authority,audit)
    val result=fabric.executeWaiting(controller,request,"runtime.user",approval=token,nowNs=21)
    check(!result.replayed && calls==1)
    check(controller.plan.status==NativePlanStatus.COMPLETED)
    check(step.result=="external-ok" && step.evidenceRecordIds==listOf(7L))
    check(audit.receipts.single().status==M6ReceiptStatus.SUCCEEDED)

    // Replay same successful request into a restored waiting plan without calling handler.
    val replayStep=NativePlannerStep(1,NativePlanStepSpec(NativeStepKind.EXTERNAL,"echo"),NativeStepStatus.WAITING_EXTERNAL)
    val replayController=NativePlanController(NativePlan(request.planId,"g",listOf(replayStep),NativePlanStatus.WAITING_EXTERNAL))
    val replay=fabric.executeWaiting(replayController,request,"runtime.user",nowNs=22)
    check(replay.replayed && calls==1)
    check(replayStep.result=="external-ok")

    val deniedStep=NativePlannerStep(1,NativePlanStepSpec(NativeStepKind.EXTERNAL,"echo"),NativeStepStatus.WAITING_EXTERNAL)
    val deniedController=NativePlanController(NativePlan(request.planId,"g",listOf(deniedStep),NativePlanStatus.WAITING_EXTERNAL))
    val deniedAuthority=M6DenyByDefaultAuthorityGate(emptyList(),approvals)
    val deniedFabric=M6ExternalExecutionFabric(registry,deniedAuthority,M6InMemoryActionAudit())
    var denied=false
    try { deniedFabric.executeWaiting(deniedController,request,"runtime.user",approval=token,nowNs=21) } catch(_:M6AuthorizationDeniedException){ denied=true }
    check(denied)

    val noApprovalStep=NativePlannerStep(1,NativePlanStepSpec(NativeStepKind.EXTERNAL,"echo"),NativeStepStatus.WAITING_EXTERNAL)
    val noApprovalController=NativePlanController(NativePlan(request.planId,"g",listOf(noApprovalStep),NativePlanStatus.WAITING_EXTERNAL))
    val noApprovalFabric=M6ExternalExecutionFabric(registry,authority,M6InMemoryActionAudit())
    var required=false
    try { noApprovalFabric.executeWaiting(noApprovalController,request,"runtime.user",nowNs=21) } catch(_:M6ApprovalRequiredException){ required=true }
    check(required)

    var surrogateRejected=false
    try {
        M6InMemoryActionAudit().append(
            status=M6ReceiptStatus.FAILED,
            request=request,
            principal="runtime.user",
            outcome=M6ActionOutcome(false,"\uD800"),
            timestampNs=23,
        )
    } catch(_:IllegalArgumentException){ surrogateRejected=true }
    check(surrogateRejected)

    println("M7P_EXECUTION_FABRIC_PASS")
}
