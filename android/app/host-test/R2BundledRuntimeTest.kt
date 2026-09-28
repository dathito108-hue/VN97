package ai.vn97.app

import android.content.res.AssetManager
import java.io.File
import java.nio.file.Files

private fun writeFixture(root: File, relative: String, text: String) {
    val file = File(root, "vn97-r2/$relative")
    file.parentFile.mkdirs()
    file.writeText(text)
}

fun main() {
    val temp = Files.createTempDirectory("vn97-r2-bundle-test").toFile()
    try {
        val assets = File(temp, "assets").apply { mkdirs() }
        val privateRoot = File(temp, "private").apply { mkdirs() }
        val manager = AssetManager(assets)
        val installer = VN97R2BundledRuntime(VN97Application(privateRoot, manager))
        check(!installer.installIfPresent(required = false))
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)

        val files = listOf(
            "binding.vn97r2f2.json", "tuning.vn97r2e4.json",
            "runtime/runtime.vn97ort1.json", "runtime/step.onnx",
            "runtime/chunk-8.onnx",
        )
        files.forEach { writeFixture(assets, it, "fixture:$it:v1") }
        // Both the staging root and nested parents already exist for sibling files.
        check(installer.installIfPresent(required = true))
        val target = File(privateRoot, "vn97-r2")
        files.forEach { check(File(target, it).readText() == "fixture:$it:v1") }
        val identity = File(target, ".apk-assets.sha256").readText()
        check(identity.trim().matches(Regex("[0-9a-f]{64}")))
        check(!installer.installIfPresent(required = true))
        check(File(target, ".apk-assets.sha256").readText() == identity)
        check(!File(privateRoot, ".vn97-r2.staging").exists())

        // Partial new staging data must never replace the working installation.
        writeFixture(assets, "binding.vn97r2f2.json", "fixture:v2")
        manager.failOpenPath = "vn97-r2/runtime/step.onnx"
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)
        files.forEach { check(File(target, it).readText() == "fixture:$it:v1") }
        check(File(target, ".apk-assets.sha256").readText() == identity)
        check(!File(privateRoot, ".vn97-r2.staging").exists())
        manager.failOpenPath = null

        check(installer.installIfPresent(required = true))
        check(File(target, "binding.vn97r2f2.json").readText() == "fixture:v2")
        val secondIdentity = File(target, ".apk-assets.sha256").readText()
        check(secondIdentity != identity)
        check(!File(privateRoot, ".vn97-r2.backup").exists())

        // Missing required graphs fail before the old target is moved.
        File(assets, "vn97-r2/runtime/step.onnx").delete()
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)
        check(File(target, ".apk-assets.sha256").readText() == secondIdentity)
        check(File(target, "runtime/step.onnx").readText() == "fixture:runtime/step.onnx:v1")
        check(!File(privateRoot, ".vn97-r2.staging").exists())
        println("R2 bundled installer: fresh install, repeat, update and failure preservation PASS")
    } finally {
        check(temp.deleteRecursively())
    }
}
