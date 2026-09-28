plugins {
    kotlin("jvm") version "2.2.0"
    application
}
repositories { mavenCentral() }
dependencies { implementation("com.microsoft.onnxruntime:onnxruntime:1.30.0") }
kotlin { jvmToolchain(17) }
kotlin.sourceSets.main {
    kotlin.srcDirs(".", "../src/main/java")
    kotlin.include("BufferRegression.kt", "ai/vn97/runtime/Mamba2OrtTensorStorage.kt", "ai/vn97/runtime/Mamba2OrtExecutionBuffers.kt")
}
application { mainClass.set("ai.vn97.runtime.BufferRegressionKt") }
tasks.named<JavaExec>("run") { args(projectDir.absolutePath) }
