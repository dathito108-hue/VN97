import ai.vn97.runtime.*

private inline fun expectPlanner(block: () -> Unit) {
    var failed=false
    try { block() } catch (_: NativePlannerException) { failed=true }
    check(failed)
}

fun main(){
    val spec=NativePlanStepSpec(
        kind=NativeStepKind.REASON,
        objective="think",
        requiresVerification=true,
        minConfidence=0.75,
    )
    val id=NativePlanIdentity.compute("answer", listOf(spec))
    check(id=="55e15d8cfd131608bbb6f97a9e6b565799c7d2eeec3c63dcf82a564b5f87f2ef") { id }

    val controller=NativePlanController.create(
        goal="answer",
        specs=listOf(
            spec,
            NativePlanStepSpec(NativeStepKind.EXTERNAL,"write",dependencies=listOf(1)),
        ),
        createdNs=1L,
    )
    check(controller.nextDirective()?.stepId==1)
    controller.beginStep(1)
    controller.completeStep(1,"candidate",0.8,listOf(7L))
    check(controller.plan.step(1).status==NativeStepStatus.WAITING_VERIFICATION)
    controller.verifyStep(1,false,"check again")
    check(controller.plan.step(1).status==NativeStepStatus.PENDING)
    controller.beginStep(1)
    controller.completeStep(1,"candidate2",0.9,listOf(7L))
    controller.verifyStep(1,true,"ok")
    check(controller.plan.step(1).status==NativeStepStatus.SUCCEEDED)
    check(controller.nextDirective()?.stepId==2)
    val ext=controller.beginStep(2)
    check(ext.externalRequired)
    check(controller.plan.status==NativePlanStatus.WAITING_EXTERNAL)
    controller.recordExternalResult(2,"written",1.0)
    check(controller.plan.status==NativePlanStatus.COMPLETED)

    val pause=NativePlanController.create("g",listOf(NativePlanStepSpec(NativeStepKind.REASON,"r")),createdNs=2L)
    pause.beginStep(1)
    pause.pause("background")
    check(pause.plan.status==NativePlanStatus.PAUSED)
    check(pause.plan.step(1).status==NativeStepStatus.PENDING)
    pause.resume()
    check(pause.plan.status==NativePlanStatus.READY)

    val memory=NativePlanController.create(
        "g",
        listOf(NativePlanStepSpec(NativeStepKind.RETRIEVE,"r")),
        budget=NativeReasoningBudget(maxMemoryQueries=1,maxMemoryHits=3),
        createdNs=3L,
    )
    check(memory.consumeMemoryQuery(9)==3)
    expectPlanner { memory.consumeMemoryQuery(1) }

    val exhausted=NativePlanController.create(
        "g",
        listOf(NativePlanStepSpec(NativeStepKind.REASON,"r")),
        budget=NativeReasoningBudget(maxTransitions=1,maxRetriesPerStep=0),
        createdNs=4L,
    )
    exhausted.beginStep(1)
    expectPlanner { exhausted.completeStep(1,"x",1.0) }
    check(exhausted.plan.status==NativePlanStatus.BUDGET_EXHAUSTED)
    check(exhausted.plan.step(1).status==NativeStepStatus.CANCELLED)

    val floatFixtures = listOf(
        0.0 to "0.0",
        -0.0 to "-0.0",
        1e-4 to "0.0001",
        1e-5 to "1e-05",
        1e15 to "1000000000000000.0",
        1e16 to "1e+16",
        1.23456789e20 to "1.23456789e+20",
        0.75 to "0.75",
    )
    for ((value, expected) in floatFixtures) {
        check(VnStrictJson.canonical(VnStrictJson.double(value)) == expected)
    }
    println("M7K_PLANNER_PASS")
}
