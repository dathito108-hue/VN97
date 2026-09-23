package ai.vn97.runtime

enum class NativeStepKind { REASON, RETRIEVE, VERIFY, RESPOND, EXTERNAL }
enum class NativeStepStatus { PENDING, RUNNING, WAITING_VERIFICATION, WAITING_EXTERNAL, SUCCEEDED, FAILED, CANCELLED }
enum class NativePlanStatus { READY, RUNNING, WAITING_EXTERNAL, PAUSED, COMPLETED, FAILED, CANCELLED, BUDGET_EXHAUSTED }
data class NativePlanStepSpec(val kind: NativeStepKind, val objective:String, val dependencies:List<Int> = emptyList())
class NativePlannerStep(val stepId:Int,val spec:NativePlanStepSpec,var status:NativeStepStatus,var result:String="",var confidence:Double?=null,var evidenceRecordIds:List<Long> = emptyList(),var failureReason:String="")
class NativePlan(val planId:String,val goal:String,val steps:List<NativePlannerStep>,var status:NativePlanStatus) {
    fun step(id:Int):NativePlannerStep = steps[id-1]
}
class NativePlanController(val plan:NativePlan) {
    fun recordExternalResult(id:Int,result:String,confidence:Double,evidenceRecordIds:List<Long> = emptyList()) {
        val s=plan.step(id); check(s.status==NativeStepStatus.WAITING_EXTERNAL); s.result=result;s.confidence=confidence;s.evidenceRecordIds=evidenceRecordIds;s.status=NativeStepStatus.SUCCEEDED;plan.status=if(plan.steps.all{it.status==NativeStepStatus.SUCCEEDED}) NativePlanStatus.COMPLETED else NativePlanStatus.READY
    }
    fun failStep(id:Int,reason:String,retryable:Boolean=true){ val s=plan.step(id);s.failureReason=reason;s.status=if(retryable) NativeStepStatus.PENDING else NativeStepStatus.FAILED;plan.status=if(retryable) NativePlanStatus.READY else NativePlanStatus.FAILED }
}
data class NativeExternalCapabilityView(val capabilityId:String,val requiredScopeKeys:List<String>,val optionalScopeKeys:List<String>,val approvalRequired:Boolean,val maxPayloadUtf8Bytes:Int,val payloadSchemaJson:String="{}")
data class NativeExternalIntentRequest(val planId:String,val goal:String,val stepId:Int,val objective:String,val capabilities:List<NativeExternalCapabilityView>)
data class NativeExternalIntent(val capabilityId:String,val scope:Map<String,String>,val payloadJson:String)
open class NativeTypedCognitionAdapter(private val result: NativeExternalIntent) { open fun proposeExternalIntent(request: NativeExternalIntentRequest): NativeExternalIntent = result }
